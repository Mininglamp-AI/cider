"""cider.attention — Optimized SDPA kernels for Apple Silicon.

Usage:
    import cider
    cider.patch_sdpa()          # monkey-patch mx.fast.scaled_dot_product_attention
    cider.autotune_sdpa()       # optional: sweep blocks for best perf
"""

from .sdpa import (
    scaled_dot_product_attention,
    load_autotune,
    autotune_sdpa,
    patch_sdpa,
    unpatch_sdpa,
)

__all__ = [
    "scaled_dot_product_attention",
    "load_autotune",
    "autotune_sdpa",
    "patch_sdpa",
    "unpatch_sdpa",
]
