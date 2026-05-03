"""cider.nn — CiderLinear: unified INT8 kernel, zero double storage.

Two internal paths (transparent to caller):
  - per_channel (gs=0): perchannel_linear (prefill GEMM + decode MV, both per-channel)
  - per_group (gs∈{64,128,256}): pergroup_linear (prefill GEMM + decode MV, per-group)

Both paths: only one copy of int8 weights in memory.
Conversion: from_float() does symmetric requant.

Usage:
    from cider import convert_model
    convert_model(model)
    # Done. Both prefill and decode use INT8 kernels.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import ops


# ── Backward compat stubs (no-op) ──────────────────────────────
def set_mode(mode: str):
    """No-op. Kept for backward compatibility."""
    pass

def get_mode() -> str:
    """Always returns 'auto'."""
    return "auto"


# ── Per-group symmetric quantization helper ────────────────────
def _symmetric_quantize_pergroup(w_fp: np.ndarray, group_size: int):
    """Quantize [N, K] float weights to per-group symmetric INT8.

    Args:
        w_fp: [N, K] float32 numpy array
        group_size: elements per group (K must be divisible)

    Returns:
        w_int8: [N, K] int8 numpy
        scale_w: [N, num_groups] float32 numpy
    """
    N, K = w_fp.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    num_groups = K // group_size

    w_reshaped = w_fp.reshape(N, num_groups, group_size)
    group_max = np.max(np.abs(w_reshaped), axis=2)  # [N, num_groups]
    scale = group_max / 127.0
    scale = np.where(scale == 0, 1.0, scale)  # avoid div by zero

    w_int8 = np.clip(
        np.round(w_reshaped / scale[:, :, np.newaxis]),
        -128, 127
    ).astype(np.int8).reshape(N, K)

    return w_int8, scale.astype(np.float32)


# ── CiderLinear ─────────────────────────────────────────────────
class CiderLinear(nn.Module):
    """Unified INT8 Linear: both prefill and decode use custom kernels.

    Internal dispatch (transparent to caller):
      per_channel (gs=0): ops.perchannel_linear — per-channel INT8 pipeline
      per_group (gs∈{64,128,256}): ops.pergroup_linear — per-group with group scales

    Both paths auto-dispatch GEMM (M>1) vs MV (M==1) internally.
    No double weight storage. Only int8 weights + scales in memory.
    """

    def __init__(
        self,
        w_int8: mx.array,       # [N, K] int8
        scale_w: mx.array,      # per-channel: [N], per-group: [num_groups, N] (row-major contiguous)
        group_size: int,        # 0 = per-channel, 64/128/256 = per-group
        in_features: int,
        out_features: int,
        bias: mx.array = None,  # [N] float16, default zeros
    ):
        super().__init__()
        self.w_int8 = w_int8
        # Per-group: physically transpose scale_w from [N, num_groups] to [num_groups, N]
        # for coalesced SIMD access in Metal kernels.
        # scale_w for per-group must already be [num_groups, N] physically contiguous.
        # This is ensured by from_float() and convert.py at construction time (numpy transpose).
        self.scale_w = scale_w
        self.group_size = group_size
        self._in_features = in_features
        self._out_features = out_features
        self.bias = bias if bias is not None else mx.zeros((out_features,), dtype=mx.float16)

        if group_size == 0:
            self._mode = "per_channel"
        elif group_size in (64, 128, 256):
            self._mode = "per_group"
        else:
            raise ValueError(f"Unsupported group_size={group_size}. Use 0 (per-channel) or 64/128/256.")

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self._in_features)

        if self._mode == "per_channel":
            y = ops.perchannel_linear(x_2d, self.w_int8, self.scale_w, self.bias)
        else:
            
            y = ops.pergroup_linear(x_2d, self.w_int8, self.scale_w, self.group_size, self.bias)

        y = y.reshape(*orig_shape[:-1], self._out_features)
        return y

    @staticmethod
    def from_float(layer: nn.Module, target_group_size: int = None) -> "CiderLinear":
        """Create from nn.Linear or nn.QuantizedLinear.

        For QuantizedLinear (8-bit, gs∈{64,128,256}): dequant → symmetric requant per-group.
        For QuantizedLinear (non-8-bit or unsupported gs): dequant → per-channel requant.
        For Linear: per-channel requant.

        Args:
            layer: Source nn.Linear or nn.QuantizedLinear
            target_group_size: Override group_size for conversion.
        """
        if isinstance(layer, nn.QuantizedLinear):
            bits = layer.bits
            gs = layer.group_size
            out_f = layer.scales.shape[0]
            in_f = layer.scales.shape[1] * gs
            lin_bias = getattr(layer, "bias", None)

            # Determine target group_size
            if target_group_size is not None:
                tgs = target_group_size
            elif bits == 8 and gs in (64, 128, 256):
                tgs = gs
            else:
                tgs = 0  # per-channel

            # Dequant
            w_fp = mx.dequantize(
                layer.weight, layer.scales,
                getattr(layer, "biases", None),
                gs, bits,
            )
            w_np = np.array(w_fp.astype(mx.float32))

            if tgs == 0:
                # Per-channel symmetric requant
                w_int8_np, scale_np = ops.quantize_weight_int8(w_np)
                return CiderLinear(
                    w_int8=mx.array(w_int8_np),
                    scale_w=mx.array(scale_np),
                    group_size=0,
                    in_features=in_f,
                    out_features=out_f,
                    bias=lin_bias,
                )
            else:
                # Per-group symmetric requant
                w_int8_np, scale_np = _symmetric_quantize_pergroup(w_np, tgs)
                return CiderLinear(
                    w_int8=mx.array(w_int8_np),
                    scale_w=mx.array(scale_np.T.copy()),  # [N,ng] -> [ng,N] contiguous
                    group_size=tgs,
                    in_features=in_f,
                    out_features=out_f,
                    bias=lin_bias,
                )

        elif hasattr(layer, "weight"):
            # FP Linear → per-channel
            out_f, in_f = layer.weight.shape
            lin_bias = getattr(layer, "bias", None)
            tgs = target_group_size or 0
            w_np = np.array(layer.weight.astype(mx.float32))

            if tgs == 0:
                w_int8_np, scale_np = ops.quantize_weight_int8(w_np)
            else:
                w_int8_np, scale_np = _symmetric_quantize_pergroup(w_np, tgs)

            return CiderLinear(
                w_int8=mx.array(w_int8_np),
                scale_w=mx.array(scale_np.T.copy() if tgs > 0 else scale_np),  # [N,ng] -> [ng,N]
                group_size=tgs,
                in_features=in_f,
                out_features=out_f,
                bias=lin_bias,
            )
        else:
            raise TypeError(f"Unsupported layer: {type(layer)}")

    @property
    def input_dims(self) -> int:
        return self._in_features

    @property
    def output_dims(self) -> int:
        return self._out_features

    def __repr__(self):
        return (
            f"CiderLinear(in={self._in_features}, out={self._out_features}, "
            f"mode={self._mode}, gs={self.group_size})"
        )


# Backward compatibility alias
W8A8Linear = CiderLinear


# ── W4A8Linear ──────────────────────────────────────────────────
class W4A8Linear(nn.Module):
    """Packed INT4 weight × INT8 activation linear layer."""

    def __init__(self, packed_weight: mx.array, scale: mx.array, K: int):
        super().__init__()
        self.packed_weight = packed_weight
        self.scale = scale
        self._K = K

    def __call__(self, x: mx.array) -> mx.array:
        return ops.w4a8_linear(x, self.packed_weight, self.scale)

    @staticmethod
    def from_weights(w: np.ndarray, zero_point: int = 8) -> "W4A8Linear":
        """Create from FP16/FP32 numpy weight [K, N]."""
        K = w.shape[0]
        packed, scale = ops.pack_weight_int4(w, zero_point)
        return W4A8Linear(mx.array(packed), mx.array(scale), K)

    @property
    def input_dims(self) -> int:
        return self._K

    @property
    def output_dims(self) -> int:
        return self.packed_weight.shape[1]
