"""cider.nn — Drop-in replacement layers for mlx.nn.Linear.

CiderLinear: W8A8 prefill + original-weight decode in a single module.
Automatically detects prefill vs decode by input sequence length:
  - seq_len > 1  → W8A8 INT8 TensorOps (faster prefill)
  - seq_len == 1 → original weights (optimal decode)

No external mode switching needed. One-line conversion:
    from cider import convert_model
    convert_model(model)
    # Done. Prefill auto-accelerated, decode unchanged.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import ops


# ── Backward compat stubs (no-op) ──────────────────────────────
def set_mode(mode: str):
    """No-op. Kept for backward compatibility. Mode is auto-detected."""
    pass

def get_mode() -> str:
    """Always returns 'auto'. Mode is detected per-call by input shape."""
    return "auto"


# ── CiderLinear ─────────────────────────────────────────────────
class CiderLinear(nn.Module):
    """Hybrid Linear: W8A8 INT8 TensorOps for prefill, original for decode.

    Prefill (seq_len > 1):
        INT8 per-token activation quantization + INT8 weight
        → TensorOps matmul2d → dequant → output dtype matches input.
    Decode (seq_len == 1):
        Delegates to the original nn.Linear / QuantizedLinear.
        Zero overhead — same speed as if CiderLinear didn't exist.

    INT8 weights are created once at convert time (persistent).
    Memory cost: ~1x extra (INT8 copy alongside original weights).
    """

    def __init__(
        self,
        original: nn.Module,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self._original = original
        self._in_features = in_features
        self._out_features = out_features
        self._w8 = None   # [in, out] int8
        self._s8 = None   # [out] float32

    def _ensure_w8(self):
        """Quantize original weights to INT8 (called once)."""
        if self._w8 is not None:
            return
        layer = self._original
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

        wt = w_fp.astype(mx.float32).T   # [in, out]
        mx.eval(wt)
        w_int8, scale = ops.quantize_weight_int8(np.array(wt))
        self._w8 = mx.array(w_int8)
        self._s8 = mx.array(scale)
        mx.eval(self._w8, self._s8)

    def __call__(self, x: mx.array) -> mx.array:
        seq_len = x.shape[-2] if x.ndim >= 2 else 1
        if seq_len > 1:
            # Prefill path: W8A8
            self._ensure_w8()
            orig_shape = x.shape
            y = ops.w8a8_linear(
                x.reshape(-1, self._in_features),
                self._w8,
                self._s8,
            )
            return y.reshape(*orig_shape[:-1], self._out_features)
        # Decode path: original weights
        return self._original(x)

    # ── Constructors ────────────────────────────────────────────
    @staticmethod
    def from_float(layer: nn.Module) -> "CiderLinear":
        """Create from an existing nn.Linear or nn.QuantizedLinear."""
        if isinstance(layer, nn.QuantizedLinear):
            out_f = layer.weight.shape[0]
            in_f = layer.weight.shape[1] * 32 // layer.bits
            if hasattr(layer, "scales"):
                out_f = layer.scales.shape[0]
                in_f = layer.scales.shape[1] * layer.group_size
        elif hasattr(layer, "weight"):
            out_f, in_f = layer.weight.shape
        else:
            raise TypeError(f"Unsupported layer: {type(layer)}")

        return CiderLinear(layer, in_f, out_f)

    @staticmethod
    def from_weights(w: np.ndarray) -> "CiderLinear":
        """Create from raw numpy weight [K, N]."""
        K, N = w.shape
        fallback = nn.Linear(K, N, bias=False)
        fallback.weight = mx.array(w.T.astype(np.float32)).astype(mx.float16)
        return CiderLinear(fallback, K, N)

    @property
    def input_dims(self) -> int:
        return self._in_features

    @property
    def output_dims(self) -> int:
        return self._out_features

    def __repr__(self):
        cached = self._w8 is not None
        return (
            f"CiderLinear(in={self._in_features}, out={self._out_features}, "
            f"cached={cached})"
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
