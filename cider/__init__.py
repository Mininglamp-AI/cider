"""cider — INT8 TensorOps quantized matmul for Apple M5+ (Metal 4).

Quick start:
    from cider import convert_model

    model, proc = load("model_path")
    convert_model(model)          # Done. Prefill auto-accelerated.

    # CiderLinear auto-detects prefill (seq>1) vs decode (seq==1).
    # No set_mode() needed. set_mode/get_mode kept as no-op for compat.

On Apple M4 and below, install succeeds but is_available() returns False.
convert_model() becomes a no-op and primitive calls raise RuntimeError.
"""

__version__ = "0.7.0"

from .ops import is_available

if is_available():
    # ── High-level API (recommended) ────────────────────────────────
    from .convert import convert_model
    from .nn import CiderLinear, set_mode, get_mode

    # ── W4A8 layer ──────────────────────────────────────────────────
    from .nn import W4A8Linear

    # ── Backward compatibility ──────────────────────────────────────
    from .nn import W8A8Linear  # alias for CiderLinear

    # ── Low-level primitives ────────────────────────────────────────
    from .ops import (
        perchannel_linear,
        w4a8_linear,
        pergroup_linear,
        int8_matmul_int32,
        quantize_weight_int8,
        pack_weight_int4,
        kernel_dir,
    )

    __all__ = [
        "convert_model",
        "set_mode",
        "get_mode",
        "CiderLinear",
        "W8A8Linear",
        "W4A8Linear",
        "perchannel_linear",
        "w4a8_linear",
        "pergroup_linear",
        "int8_matmul_int32",
        "quantize_weight_int8",
        "pack_weight_int4",
        "is_available",
        "kernel_dir",
    ]
else:
    # M4 or below: graceful degradation
    def convert_model(*args, **kwargs):
        """No-op on unsupported hardware."""
        import warnings
        warnings.warn(
            "cider.convert_model() is a no-op: INT8 TensorOps require Apple M5+. "
            "Model will use standard MLX inference.",
            RuntimeWarning, stacklevel=2,
        )

    def set_mode(*args, **kwargs):
        pass

    def get_mode():
        return "unavailable"

    __all__ = ["convert_model", "set_mode", "get_mode", "is_available"]
