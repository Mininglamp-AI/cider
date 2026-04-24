"""
mlx_w8a8.fused_hybrid — Fused W8A8 Hybrid for Qwen3-VL prefill acceleration.

QKV fuse + gate/up fuse → fewer kernel dispatches → better GPU occupancy.
Prefill: W8A8 INT8 TensorOps with fused projections.
Decode: original QuantizedLinear (W8A16) unchanged.

Usage:
    from mlx_w8a8.fused_hybrid import convert_model_fused, set_mode

    model, proc = load(model_path)           # Load W8A16 model
    convert_model_fused(model)               # Patch in-place
    set_mode("prefill")                      # Before prefill chunks
    # ... run prefill ...
    set_mode("decode")                       # Before decode loop
    # ... run decode (uses original W8A16) ...
"""

import time, gc
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# ── Chip capability detection ───────────────────────────────────
def is_w8a8_available() -> bool:
    """Check if the current hardware supports W8A8 INT8 TensorOps.

    Requires Apple M5 or later (Metal 4.0 MPP cooperative_tensor + matmul2d).
    M4 and earlier do NOT support INT8 TensorOps — returns False.

    Returns:
        True if W8A8 acceleration is available and the extension loads correctly.
    """
    import subprocess
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return False

    # M5, M5 Pro, M5 Max, M5 Ultra, and future M6+ all support Metal 4
    # M4 and below do NOT have INT8 TensorOps
    import re
    m = re.match(r"Apple M(\d+)", chip)
    if not m or int(m.group(1)) < 5:
        return False

    # Verify extension actually loads
    try:
        _ensure_ext()
        return True
    except Exception:
        return False


# Lazy imports for extension
_w8a8_prim = None
_KERNEL_DIR = None


def _quantize_per_channel_np(w_f32):
    """Per-channel INT8 quantization. w: [K, N] float32 → (int8 [K,N], scale [N])."""
    col_max = np.max(np.abs(w_f32), axis=0)  # [N]
    scale = col_max / 127.0
    scale = np.where(scale == 0, 1.0, scale)  # avoid div-by-zero
    w_int8 = np.clip(np.round(w_f32 / scale[np.newaxis, :]), -128, 127).astype(np.int8)
    return w_int8, scale.astype(np.float32)


def _ensure_ext():
    global _w8a8_prim, _KERNEL_DIR
    if _w8a8_prim is None:
        import os, sys
        ext_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "ext", "lib")
        if ext_lib not in sys.path:
            sys.path.insert(0, ext_lib)
        import _w8a8_prim as _prim
        _w8a8_prim = _prim
        _KERNEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "kernels")


# ── Global mode switch ──────────────────────────────────────────
_hybrid_mode = "decode"


def set_mode(mode: str):
    """Set the hybrid mode: 'prefill' (W8A8 fused) or 'decode' (W8A16 original)."""
    global _hybrid_mode
    assert mode in ("prefill", "decode"), f"Invalid mode: {mode}"
    _hybrid_mode = mode


def get_mode() -> str:
    return _hybrid_mode


# ── Helpers ─────────────────────────────────────────────────────
def _dequant_ql(ql):
    """Dequantize a QuantizedLinear → FP16 weight [out_features, in_features]."""
    w = mx.dequantize(
        ql.weight, ql.scales,
        ql.biases if hasattr(ql, 'biases') and ql.biases is not None else None,
        ql.group_size, ql.bits,
    )
    mx.eval(w)
    return w


def _make_w8a8(w_fp16_out_in):
    """Convert [out, in] FP16 weight → (INT8 [in, out], FP32 scales [out])."""
    _ensure_ext()
    w_t = mx.transpose(w_fp16_out_in)
    mx.eval(w_t)
    w_np = np.array(w_t, dtype=np.float32)
    w_int8, scales = _quantize_per_channel_np(w_np)
    result = (mx.array(w_int8), mx.array(scales))
    mx.eval(*result)
    return result


def _w8a8_gemm(x, weight_int8, scales):
    """Dispatch W8A8 GEMM via mlx primitive. x: [..., K], weight: [K, N]."""
    _ensure_ext()
    orig_shape = x.shape
    K = orig_shape[-1]
    N = weight_int8.shape[1]
    if x.ndim > 2:
        x_2d = mx.reshape(x, (-1, K))
    else:
        x_2d = x
    y_2d = _w8a8_prim.w8a8_linear(x_2d, weight_int8, scales, _KERNEL_DIR)
    if x.ndim > 2:
        return mx.reshape(y_2d, orig_shape[:-1] + (N,))
    return y_2d


# ── Fused Attention forward ────────────────────────────────────
def _fused_attention_call(self, x, mask=None, cache=None, position_ids=None):
    """Attention forward: fused QKV GEMM in prefill, original in decode."""
    B, L, D = x.shape

    if _hybrid_mode == "prefill" and hasattr(self, '_fused_qkv_w'):
        # Single fused QKV GEMM: [B,L,dim] → [B,L, n_q+n_k+n_v]
        qkv = _w8a8_gemm(x, self._fused_qkv_w, self._fused_qkv_s)
        n_q, n_k, n_v = self._qkv_split
        queries = qkv[..., :n_q]
        keys = qkv[..., n_q:n_q + n_k]
        values = qkv[..., n_q + n_k:]
    else:
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

    queries = self.q_norm(
        queries.reshape(B, L, self.n_heads, self.head_dim)
    ).transpose(0, 2, 1, 3)
    keys = self.k_norm(
        keys.reshape(B, L, self.n_kv_heads, self.head_dim)
    ).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
        0, 2, 1, 3
    )

    kv_seq_len = keys.shape[-2]
    if position_ids is None:
        kv_seq_len += cache.offset + 1
        position_ids = mx.arange(cache.offset, cache.offset + L)
        position_ids = mx.expand_dims(position_ids, axis=0)
        position_ids = mx.tile(position_ids, (3, 1, 1))
    else:
        kv_seq_len += cache.offset + 1 if cache is not None else 0

    from mlx_vlm.models.qwen3_vl.language import (
        apply_multimodal_rotary_pos_emb,
        scaled_dot_product_attention,
    )

    cos, sin = self.rotary_emb(values, position_ids)
    if mask is not None and isinstance(mask, mx.array):
        if isinstance(kv_seq_len, mx.array):
            kv_seq_len = kv_seq_len.max().item()
        mask = mask[..., :int(kv_seq_len)]

    queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)

    output = scaled_dot_product_attention(
        queries, keys, values, cache, scale=self.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

    # o_proj
    if _hybrid_mode == "prefill" and hasattr(self.o_proj, '_w8a8_w'):
        return _w8a8_gemm(output, self.o_proj._w8a8_w, self.o_proj._w8a8_s)
    return self.o_proj(output)


# ── Fused MLP forward ──────────────────────────────────────────
def _fused_mlp_call(self, x):
    """MLP forward: fused gate+up GEMM in prefill, original in decode."""
    if _hybrid_mode == "prefill" and hasattr(self, '_fused_gu_w'):
        gate_up = _w8a8_gemm(x, self._fused_gu_w, self._fused_gu_s)
        split = self._gu_split
        hidden = nn.silu(gate_up[..., :split]) * gate_up[..., split:]
        if hasattr(self.down_proj, '_w8a8_w'):
            return _w8a8_gemm(hidden, self.down_proj._w8a8_w, self.down_proj._w8a8_s)
        return self.down_proj(hidden)
    return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Model conversion ───────────────────────────────────────────
def convert_model_fused(model, verbose=True):
    """Convert Qwen3-VL language model to fused W8A8 hybrid (in-place).

    Fuses QKV projections and gate+up projections for prefill acceleration.
    Decode path remains unchanged (original QuantizedLinear).

    Args:
        model: Loaded Qwen3-VL model (from mlx_vlm.utils.load).
        verbose: Print progress.

    Returns:
        dict with conversion stats.
    """
    _ensure_ext()
    t0 = time.perf_counter()

    layers = model.language_model.model.layers
    n_layers = len(layers)

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        mlp = layer.mlp

        # ── Fuse QKV ──
        w_q = _dequant_ql(attn.q_proj)
        w_k = _dequant_ql(attn.k_proj)
        w_v = _dequant_ql(attn.v_proj)
        w_qkv = mx.concatenate([w_q, w_k, w_v], axis=0)
        mx.eval(w_qkv)
        attn._fused_qkv_w, attn._fused_qkv_s = _make_w8a8(w_qkv)
        attn._qkv_split = (w_q.shape[0], w_k.shape[0], w_v.shape[0])
        del w_q, w_k, w_v, w_qkv

        # ── W8A8 for o_proj ──
        attn.o_proj._w8a8_w, attn.o_proj._w8a8_s = _make_w8a8(_dequant_ql(attn.o_proj))

        # ── Fuse gate+up ──
        w_gate = _dequant_ql(mlp.gate_proj)
        w_up = _dequant_ql(mlp.up_proj)
        w_gu = mx.concatenate([w_gate, w_up], axis=0)
        mx.eval(w_gu)
        mlp._fused_gu_w, mlp._fused_gu_s = _make_w8a8(w_gu)
        mlp._gu_split = w_gate.shape[0]
        del w_gate, w_up, w_gu

        # ── W8A8 for down_proj ──
        mlp.down_proj._w8a8_w, mlp.down_proj._w8a8_s = _make_w8a8(_dequant_ql(mlp.down_proj))

        # Periodic eval to prevent graph explosion
        if (i + 1) % 7 == 0 or i == n_layers - 1:
            mx.eval(model.parameters())
            gc.collect()

    # ── Class swap for __call__ override ──
    from mlx_vlm.models.qwen3_vl.language import Attention as OrigAttention
    from mlx_vlm.models.qwen3_vl.language import MLP as OrigMLP

    class FusedAttention(OrigAttention):
        __call__ = _fused_attention_call

    class FusedMLP(OrigMLP):
        __call__ = _fused_mlp_call

    for layer in layers:
        layer.self_attn.__class__ = FusedAttention
        layer.mlp.__class__ = FusedMLP

    elapsed = time.perf_counter() - t0
    stats = {
        "n_layers": n_layers,
        "fused_qkv": n_layers,
        "fused_gate_up": n_layers,
        "w8a8_single": n_layers * 2,  # o_proj + down_proj
        "elapsed_s": elapsed,
    }
    if verbose:
        print(f"[W8A8 Fused] Converted {n_layers} layers "
              f"({n_layers} QKV fused + {n_layers} gate_up fused + {n_layers*2} single) "
              f"in {elapsed:.1f}s")
    return stats
