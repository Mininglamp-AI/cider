"""cider — INT8 TensorOps quantized matmul for Apple M5+ (Metal 4).

Provides W8A8 and W4A8 quantized linear layers as MLX custom primitives,
fully compatible with MLX's lazy evaluation graph.

Supported modes:
  - W8A8: INT8 weights × INT8 activations via TensorOps matmul2d
  - W4A8: Packed INT4 weights × INT8 activations (software unpack + TensorOps)

Quick start:
    import mlx.core as mx
    from cider import W8A8Linear, W4A8Linear

    layer = W8A8Linear.from_fp16(weight_fp16)
    y = layer(x)   # lazy — evaluated when mx.eval(y) is called
"""

__version__ = "0.6.0"

from .ops import (
    w8a8_linear,
    int8_matmul_int32,
    w4a8_linear,
    quantize_weight_int8,
    pack_weight_int4,
    is_available,
    kernel_dir,
)
from .nn import W8A8Linear, W4A8Linear

__all__ = [
    "w8a8_linear",
    "w4a8_linear",
    "int8_matmul_int32",
    "quantize_weight_int8",
    "pack_weight_int4",
    "is_available",
    "kernel_dir",
    "W8A8Linear",
    "W4A8Linear",
]
