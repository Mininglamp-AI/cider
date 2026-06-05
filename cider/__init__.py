"""cider — INT8 TensorOps quantized matmul + optimized SDPA for Apple Silicon.

Quick start (attention acceleration):
    import cider
    cider.patch_sdpa()          # One line. mlx_lm/mlx_vlm auto-accelerated.
    cider.autotune_sdpa()       # Optional: sweep blocks for best perf on your GPU.

Quick start (quantization):
    from cider import convert_model
    model, proc = load("model_path")
    convert_model(model)
"""

__version__ = "0.8.0"

# ── Attention acceleration (always available) ───────────────────
from .attention import patch_sdpa, unpatch_sdpa, autotune_sdpa

# ── Quantization (M5+ only) ────────────────────────────────────
from .ops import is_available

if is_available():
    from .convert import convert_model
    from .nn import CiderLinear, set_mode, get_mode, W4A8Linear, W8A8Linear
    from .ops import (
        perchannel_linear, w4a8_linear, pergroup_linear,
        int8_matmul_int32, quantize_weight_int8, pack_weight_int4, kernel_dir,
    )

    __all__ = [
        "patch_sdpa", "unpatch_sdpa", "autotune_sdpa",
        "convert_model", "set_mode", "get_mode",
        "CiderLinear", "W8A8Linear", "W4A8Linear",
        "perchannel_linear", "w4a8_linear", "pergroup_linear",
        "int8_matmul_int32", "quantize_weight_int8", "pack_weight_int4",
        "is_available", "kernel_dir",
    ]
else:
    def convert_model(*args, **kwargs):
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

    __all__ = [
        "patch_sdpa", "unpatch_sdpa", "autotune_sdpa",
        "convert_model", "set_mode", "get_mode", "is_available",
    ]
