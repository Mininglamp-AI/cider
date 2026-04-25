"""cider.nn — Drop-in replacement layers for mlx.nn.Linear.

CiderLinear is the primary layer: W8A8 prefill + original-weight decode
in a single module, controlled by set_mode("prefill"/"decode").

W4A8Linear provides packed INT4 weight × INT8 activation.

Usage (recommended — automatic model conversion):
    from cider import convert_model, set_mode
    convert_model(model)
    set_mode("prefill")

Usage (manual, single layer):
    from cider import CiderLinear
    layer = CiderLinear.from_float(existing_linear)
    y = layer(x)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import ops

# ── Global mode (shared with convert.py) ────────────────────────
_mode = "decode"


def set_mode(mode: str):
    """Switch between 'prefill' (W8A8) and 'decode' (original weights).

    Call before each inference phase:
        set_mode("prefill")   # W8A8 INT8 TensorOps — faster for long seqs
        set_mode("decode")    # Original weights — optimal for single tokens
    """
    global _mode
    if mode not in ("prefill", "decode"):
        raise ValueError(f"Mode must be 'prefill' or 'decode', got '{mode}'")
    _mode = mode


def get_mode() -> str:
    """Return the current mode ('prefill' or 'decode')."""
    return _mode


# ── CiderLinear ─────────────────────────────────────────────────
class CiderLinear(nn.Module):
    """Hybrid Linear: W8A8 INT8 TensorOps in prefill, original in decode.

    Prefill mode:  INT8 per-token activation quantization + INT8 weight
                   → TensorOps matmul2d → dequant → output dtype matches input.
    Decode mode:   Delegates to the original nn.Linear / QuantizedLinear.
                   Zero overhead — same speed as if CiderLinear didn't exist.

    Weight layout (W8A8 path): [in_features, out_features] int8, per-column.
    Scale: [out_features] float32.
    """

    def __init__(
        self,
        original: nn.Module,
        w8: mx.array,
        s8: mx.array,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self._original = original
        self._w8 = w8                    # [in, out] int8
        self._s8 = s8                    # [out] float32
        self._in_features = in_features
        self._out_features = out_features

    def __call__(self, x: mx.array) -> mx.array:
        if _mode == "prefill":
            orig_shape = x.shape
            y = ops.w8a8_linear(
                x.reshape(-1, self._in_features),
                self._w8,
                self._s8,
            )
            return y.reshape(*orig_shape[:-1], self._out_features)
        return self._original(x)

    # ── Constructors ────────────────────────────────────────────
    @staticmethod
    def from_float(layer: nn.Module) -> "CiderLinear":
        """Create from an existing nn.Linear or nn.QuantizedLinear.

        Extracts and quantizes weights automatically.
        The original layer is kept for decode fallback.
        """
        if isinstance(layer, nn.QuantizedLinear):
            w_fp = mx.dequantize(
                layer.weight, layer.scales,
                getattr(layer, "biases", None),
                layer.group_size, layer.bits,
            )
        elif hasattr(layer, "weight"):
            w_fp = layer.weight
        else:
            raise TypeError(f"Unsupported layer: {type(layer)}")

        out_f, in_f = w_fp.shape[0], w_fp.shape[1]
        wt = w_fp.astype(mx.float32).T       # [in, out]
        mx.eval(wt)
        w_int8, scale = ops.quantize_weight_int8(np.array(wt))
        w8 = mx.array(w_int8)
        s8 = mx.array(scale)
        mx.eval(w8, s8)
        return CiderLinear(layer, w8, s8, in_f, out_f)

    @staticmethod
    def from_weights(w: np.ndarray) -> "CiderLinear":
        """Create from raw numpy weight [K, N] (no decode fallback).

        For standalone use without a backing nn.Linear.
        Decode mode will use a plain FP16 matmul.
        """
        K, N = w.shape
        w_int8, scale = ops.quantize_weight_int8(w.astype(np.float32))
        w8 = mx.array(w_int8)
        s8 = mx.array(scale)
        # Build a plain Linear as decode fallback
        fallback = nn.Linear(K, N, bias=False)
        fallback.weight = mx.array(w.T.astype(np.float32)).astype(mx.float16)
        return CiderLinear(fallback, w8, s8, K, N)

    @property
    def input_dims(self) -> int:
        return self._in_features

    @property
    def output_dims(self) -> int:
        return self._out_features

    def __repr__(self):
        return (
            f"CiderLinear(in={self._in_features}, out={self._out_features}, "
            f"mode={_mode})"
        )


# Backward compatibility alias
W8A8Linear = CiderLinear


# ── W4A8Linear ──────────────────────────────────────────────────
class W4A8Linear(nn.Module):
    """Packed INT4 weight × INT8 activation linear layer.

    Weight layout: [K//2, N] uint8 (packed nibbles).
    Scale: [N] float32, per-column.
    """

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
