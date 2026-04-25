"""cider.ops — Low-level primitive API for W8A8 / W4A8 linear.

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
    # The compiled extension lives in lib/ next to this package
    import sys
    lib_dir = str(Path(__file__).parent / "lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    import _cider_prim
    _ext = _cider_prim
    return _ext


# ── Hardware detection ──────────────────────────────────────────

def is_available() -> bool:
    """Check if INT8 TensorOps are available (Apple M5+, Metal 4).

    Returns False on M4 and earlier, or if the extension fails to load.
    """
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
    except Exception:
        return False


# ── Weight quantization helpers ─────────────────────────────────

def quantize_weight_int8(
    w: np.ndarray,
) -> tuple:
    """Quantize FP16/FP32 weights to per-column symmetric INT8.

    Args:
        w: Weight matrix [K, N] as numpy array.

    Returns:
        (w_int8, scale_w) where w_int8 is [K, N] int8 and
        scale_w is [N] float32.
    """
    w = w.astype(np.float32)
    col_max = np.max(np.abs(w), axis=0)  # [N]
    scale = col_max / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    w_int8 = np.clip(np.round(w / scale[np.newaxis, :]), -128, 127).astype(np.int8)
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
    # Quantize to [0, 15] with zero_point
    w_q = np.clip(np.round(w / scale[np.newaxis, :]) + zero_point, 0, 15).astype(np.uint8)
    # Pack: high nibble = even rows, low nibble = odd rows
    packed = (w_q[0::2, :] << 4) | w_q[1::2, :]
    return packed, scale.astype(np.float32)


# ── Primitive API ───────────────────────────────────────────────

def w8a8_linear(
    x: mx.array,
    w: mx.array,
    scale_w: mx.array,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """W8A8 quantized linear: y = dequant(quant_a(x) @ w_int8).

    This returns a lazy mx.array — computation is deferred until
    mx.eval() is called, integrating with MLX's computation graph.

    Args:
        x: Input activations [M, K] float16 or bfloat16.
        w: INT8 weights [K, N] int8 (per-column quantized).
        scale_w: Per-column weight scales [N] float32.
        stream: Optional MLX stream for scheduling.

    Returns:
        Output [M, N] matching input dtype (float16 or bfloat16).
    """
    ext = _load_ext()
    out_dtype = x.dtype
    kw = {"stream": stream} if stream is not None else {}
    result = ext.w8a8_linear(x, w, scale_w, kernel_dir(), **kw)
    # Kernel outputs float16; cast to input dtype if needed (lazy, near-zero cost)
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

    Packed INT4 weights are unpacked to INT8 on-the-fly in the Metal
    kernel. Activation quantization (FP16 -> INT8) is fused in the
    same command buffer.

    This returns a lazy mx.array — computation is deferred until
    mx.eval() is called.

    Args:
        x: Input activations [M, K] float16.
        packed_w: Packed INT4 weights [K//2, N] uint8.
        scale_w: Per-column weight scales [N] float32.
        stream: Optional MLX stream for scheduling.

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

    Pure integer GEMM: C[i,j] = sum_k A[i,k] * B[k,j].
    No activation quantization, no scale dequant.
    Result is exact — suitable for bit-level correctness testing.

    Args:
        a: INT8 matrix [M, K].
        b: INT8 matrix [K, N].
        stream: Optional MLX stream.

    Returns:
        INT32 result [M, N].
    """
    ext = _load_ext()
    kw = {"stream": stream} if stream is not None else {}
    return ext.int8_matmul_int32(a, b, kernel_dir(), **kw)
