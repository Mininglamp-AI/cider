"""cider — INT8 TensorOps quantized matmul for Apple M5+ (Metal 4).

Quick start:
    from cider import convert_model, set_mode

    model, proc = load("model_path")
    convert_model(model)          # Patch all Linear → CiderLinear

    set_mode("prefill")           # W8A8 INT8 TensorOps (~15-19% faster)
    set_mode("decode")            # Original weights (zero overhead)
"""

__version__ = "0.7.0"

# ── High-level API (recommended) ────────────────────────────────
from .convert import convert_model
from .nn import CiderLinear, set_mode, get_mode

# ── W4A8 layer ──────────────────────────────────────────────────
from .nn import W4A8Linear

# ── Backward compatibility ──────────────────────────────────────
from .nn import W8A8Linear  # alias for CiderLinear

# ── Low-level primitives ────────────────────────────────────────
from .ops import (
    w8a8_linear,
    w4a8_linear,
    int8_matmul_int32,
    quantize_weight_int8,
    pack_weight_int4,
    is_available,
    kernel_dir,
)

__all__ = [
    # High-level (start here)
    "convert_model",
    "set_mode",
    "get_mode",
    "CiderLinear",
    # Layers
    "W8A8Linear",       # alias → CiderLinear
    "W4A8Linear",
    # Primitives
    "w8a8_linear",
    "w4a8_linear",
    "int8_matmul_int32",
    "quantize_weight_int8",
    "pack_weight_int4",
    "is_available",
    "kernel_dir",
]
