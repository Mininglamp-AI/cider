"""cider.nn — Drop-in replacement layers for mlx.nn.Linear.

W8A8Linear and W4A8Linear are mlx.nn.Module subclasses whose
__call__ returns lazy mx.array nodes, fully compatible with MLX's
computation graph and mx.eval().

Usage:
    # Convert from existing FP16 weights
    layer = W8A8Linear.from_weights(w_fp16)  # [K, N] or [N, K]
    y = layer(x)  # [M, N] float16, lazy
    mx.eval(y)

    # Or use with quantize helpers
    import numpy as np
    from cider import quantize_weight_int8
    w_int8, scale = quantize_weight_int8(w_np)
    layer = W8A8Linear(mx.array(w_int8), mx.array(scale))
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import ops


class W8A8Linear(nn.Module):
    """INT8 weight × INT8 activation linear layer (TensorOps).

    Weight layout: [K, N] int8 (input_dims × output_dims).
    Scale: [N] float32, per-column.
    """

    def __init__(self, weight: mx.array, scale: mx.array):
        super().__init__()
        self.weight = weight    # [K, N] int8 — frozen, no grad
        self.scale = scale      # [N] float32

    def __call__(self, x: mx.array) -> mx.array:
        return ops.w8a8_linear(x, self.weight, self.scale)

    @staticmethod
    def from_weights(w: np.ndarray) -> "W8A8Linear":
        """Create from FP16/FP32 numpy weight [K, N]."""
        w_int8, scale = ops.quantize_weight_int8(w)
        return W8A8Linear(mx.array(w_int8), mx.array(scale))

    @property
    def input_dims(self) -> int:
        return self.weight.shape[0]

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]


class W4A8Linear(nn.Module):
    """Packed INT4 weight × INT8 activation linear layer.

    Weight layout: [K//2, N] uint8 (packed nibbles).
    Scale: [N] float32, per-column.
    """

    def __init__(self, packed_weight: mx.array, scale: mx.array, K: int):
        super().__init__()
        self.packed_weight = packed_weight  # [K//2, N] uint8
        self.scale = scale                  # [N] float32
        self._K = K                         # original K (for shape reporting)

    def __call__(self, x: mx.array) -> mx.array:
        return ops.w4a8_linear(x, self.packed_weight, self.scale)

    @staticmethod
    def from_weights(
        w: np.ndarray,
        zero_point: int = 8,
    ) -> "W4A8Linear":
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
