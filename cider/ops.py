"""cider.ops — Low-level primitive API for W8A8 / W4A8 / per-group linear.

These functions return lazy mx.array nodes. Computation happens when
you call mx.eval() — fully compatible with MLX's graph-based execution.
"""

import re
import subprocess
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

# ── Kernel directory (shipped with the package) ─────────────────
_KERNEL_DIR: Optional[str] = None


def kernel_dir() -> str:
    """Return the absolute path to the bundled Metal kernels."""
    global _KERNEL_DIR
    if _KERNEL_DIR is None:
        _KERNEL_DIR = str(Path(__file__).parent / "kernels")
    return _KERNEL_DIR


# ── Extension loader ────────────────────────────────────────────
_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    import sys
    lib_dir = str(Path(__file__).parent / "lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    try:
        import _cider_prim
        _ext = _cider_prim
        return _ext
    except ImportError:
        raise RuntimeError(
            "Cider C++ extension not available. INT8 TensorOps require Apple M5+. "
            "On M4 and below, use standard MLX inference instead."
        )


# ── Hardware detection ──────────────────────────────────────────

def is_available() -> bool:
    """Check if INT8 TensorOps are available (Apple M5+, Metal 4)."""
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return False
    m = re.match(r"Apple M(\d+)", chip)
    if not m or int(m.group(1)) < 5:
        return False
    try:
        _load_ext()
        return True
    except (ImportError, RuntimeError, Exception):
        return False


# ── Weight quantization helpers ─────────────────────────────────

def quantize_weight_int8(
    w: np.ndarray,
) -> tuple:
    """Quantize FP16/FP32 weights to per-row symmetric INT8.

    Args:
        w: Weight matrix [N, K] as numpy array (N=out_features, K=in_features).

    Returns:
        (w_int8, scale_w) where w_int8 is [N, K] int8 and
        scale_w is [N] float32 (one scale per output channel).
    """
    w = w.astype(np.float32)
    row_max = np.max(np.abs(w), axis=1)  # [N]
    scale = row_max / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    w_int8 = np.clip(np.round(w / scale[:, np.newaxis]), -128, 127).astype(np.int8)
    return w_int8, scale.astype(np.float32)


def pack_weight_int4(
    w: np.ndarray,
    zero_point: int = 8,
) -> tuple:
    """Quantize FP16/FP32 weights to packed INT4 (symmetric, per-column).

    Args:
        w: Weight matrix [K, N] as numpy array. K must be even.
        zero_point: INT4 zero point (default 8 for signed range [-8, 7]).

    Returns:
        (packed_w, scale_w) where packed_w is [K//2, N] uint8
        (high nibble = even k, low nibble = odd k) and
        scale_w is [N] float32.
    """
    K, N = w.shape
    assert K % 2 == 0, f"K must be even, got {K}"
    w = w.astype(np.float32)
    col_max = np.max(np.abs(w), axis=0)
    scale = col_max / 7.0
    scale = np.where(scale == 0, 1.0, scale)
    w_q = np.clip(np.round(w / scale[np.newaxis, :]) + zero_point, 0, 15).astype(np.uint8)
    packed = (w_q[0::2, :] << 4) | w_q[1::2, :]
    return packed, scale.astype(np.float32)


# ── Primitive API ───────────────────────────────────────────────

def perchannel_linear(
    x: mx.array,
    w: mx.array,
    scale_w: mx.array,
    bias: Optional[mx.array] = None,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """W8A8 per-channel quantized linear: y = dequant(quant_a(x) @ w_int8) + bias.

    Args:
        x: Input activations [M, K] float16 or bfloat16.
        w: INT8 weights [N, K] int8 (per-row quantized).
        scale_w: Per-row weight scales [N] float32.
        stream: Optional MLX stream.

    Returns:
        Output [M, N] matching input dtype.
    """
    ext = _load_ext()
    out_dtype = x.dtype
    kw = {"stream": stream} if stream is not None else {}
    N = w.shape[0]
    if bias is None:
        bias = mx.zeros((N,), dtype=mx.float16)
    result = ext.perchannel_linear(x, w, scale_w, bias, kernel_dir(), **kw)
    if out_dtype != mx.float16:
        result = result.astype(out_dtype, **kw)
    return result


# Shared placeholder for new_bias (V5 kernel ignores it; Metal needs valid buffer)
_shared_new_bias_cache = {}

def _get_shared_new_bias_placeholder(N: int, num_groups: int):
    key = (N, num_groups)
    if key not in _shared_new_bias_cache:
        _shared_new_bias_cache[key] = mx.zeros((N, num_groups), dtype=mx.float32)
    return _shared_new_bias_cache[key]


def pergroup_linear(
    x: mx.array,
    w: mx.array,
    scale_w: mx.array,
    group_size: int,
    bias: Optional[mx.array] = None,
    new_bias: Optional[mx.array] = None,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """
    mlx native quantize format asymmetric affine
    quantize: q = clip(round((w - biases) / scales), 0, 2^b - 1), b = bits
    dequantize: w = q*scale + bias
    """
    """Per-group INT8 linear with optional bias.

    Dispatches internally:
      M > 1 → per-group GEMM (activation quantize + INT8 TensorOps)
      M == 1 → per-group MV (FP activation, weight dequant on-the-fly)

    Args:
        x: Input activations [M, K] float16 or bfloat16.
        w: INT8 weights [N, K] int8 (per-group symmetric quantized).
        scale_w: Per-group weight scales [num_groups, N] float32 (physically transposed for coalesced GPU access).
        group_size: Group size (64, 128, or 256).
        bias: Optional bias [N] float16. Default zeros.
        stream: Optional MLX stream.

    Returns:
        Output [M, N] matching input dtype.
    """
    ext = _load_ext()
    N = w.shape[0]
    num_groups = scale_w.shape[0] if scale_w.ndim == 2 else 1  # scale_w is [num_groups, N]
    if bias is None:
        bias = mx.zeros((N,), dtype=mx.float16)
    if new_bias is None:
        # V5 kernel ignores new_bias (symmetric quantization), but Metal
        # requires a valid buffer binding.  Use a tiny shared placeholder
        # instead of allocating (N, num_groups) every forward call.
        new_bias = _get_shared_new_bias_placeholder(N, num_groups)
    out_dtype = x.dtype
    kw = {"stream": stream} if stream is not None else {}
    # scale_w layout: [num_groups, N] physically contiguous. Kernel indexes as scale_w[g * N + n].

    result = ext.pergroup_linear(x, w, scale_w, bias, new_bias, group_size, kernel_dir(), **kw)
    if out_dtype != mx.float16:
        result = result.astype(out_dtype, **kw)
    return result


def w4a8_linear(
    x: mx.array,
    packed_w: mx.array,
    scale_w: mx.array,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """W4A8 quantized linear: y = dequant(quant_a(x) @ unpack4(w)).

    Args:
        x: Input activations [M, K] float16.
        packed_w: Packed INT4 weights [K//2, N] uint8.
        scale_w: Per-column weight scales [N] float32.
        stream: Optional MLX stream.

    Returns:
        Output [M, N] float16.
    """
    ext = _load_ext()
    out_dtype = x.dtype
    kw = {"stream": stream} if stream is not None else {}
    result = ext.w4a8_linear(x, packed_w, scale_w, kernel_dir(), **kw)
    if out_dtype != mx.float16:
        result = result.astype(out_dtype, **kw)
    return result


def int8_matmul_int32(
    a: mx.array,
    b: mx.array,
    stream=None,
) -> mx.array:
    """Raw INT8×INT8→INT32 matmul (bit-exact, no dequant).

    Args:
        a: INT8 matrix [M, K].
        b: INT8 matrix [N, K] (transposed weight layout).
        stream: Optional MLX stream.

    Returns:
        INT32 result [M, N].
    """
    ext = _load_ext()
    kw = {"stream": stream} if stream is not None else {}
    return ext.int8_matmul_int32(a, b, kernel_dir(), **kw)
