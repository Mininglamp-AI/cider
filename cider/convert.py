"""cider.convert — One-line model conversion for W8A8 prefill acceleration.

Usage:
    from cider import convert_model, set_mode

    model, proc = load("model_path")
    convert_model(model)         # Patch all Linear layers in-place

    set_mode("prefill")          # W8A8 INT8 TensorOps (~15-19% faster)
    # ... run prefill ...
    set_mode("decode")           # Original weights (no overhead)
    # ... run decode ...
works with standard MLX LLM/VLM architectures — tested on Qwen, Llama, Qwen3-VL
Supports float16 and bfloat16 models automatically.
"""

import gc
import time

import mlx.nn as nn

from . import ops
from .nn import CiderLinear

# Re-export mode control from nn (single source of truth)
from .nn import set_mode, get_mode  # noqa: F401

# ── Model conversion ───────────────────────────────────────────
_TARGET_TYPES = (nn.Linear, nn.QuantizedLinear)


def _convert_children(module, counter):
    """Walk module.children(), replace Linear/QuantizedLinear in-place."""
    for name, child in module.children().items():
        if isinstance(child, _TARGET_TYPES):
            setattr(module, name, CiderLinear.from_float(child))
            counter[0] += 1
            if counter[0] % 28 == 0:
                gc.collect()
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, _TARGET_TYPES):
                    child[i] = CiderLinear.from_float(item)
                    counter[0] += 1
                    if counter[0] % 28 == 0:
                        gc.collect()
                elif isinstance(item, nn.Module):
                    _convert_children(item, counter)
        elif isinstance(child, nn.Module):
            _convert_children(child, counter)
        # Skip dict/other non-Module children (e.g. rope_scaling)


def convert_model(
    model: nn.Module,
    *,
    verbose: bool = True,
) -> dict:
    """Convert all Linear/QuantizedLinear layers to CiderLinear (in-place).

    After conversion:
    - set_mode("prefill") → all linears use W8A8 INT8 TensorOps
    - set_mode("decode")  → all linears use original weights (no overhead)

    Args:
        model: Any MLX nn.Module (Qwen3-VL, Llama, Mistral, etc.).
        verbose: Print conversion summary.

    Returns:
        dict with stats: n_converted, elapsed_s.

    Example:
        from cider import convert_model, set_mode

        model, proc = load("model_path")
        stats = convert_model(model)

        set_mode("prefill")
        # ... run prefill (W8A8, ~15-19% faster) ...

        set_mode("decode")
        # ... run decode (original weights, optimal for single token) ...
    """
    if not ops.is_available():
        raise RuntimeError(
            "W8A8 INT8 TensorOps not available. Requires Apple M5+ with Metal 4."
        )

    t0 = time.perf_counter()
    counter = [0]
    _convert_children(model, counter)

    elapsed = time.perf_counter() - t0
    n = counter[0]
    stats = {"n_converted": n, "elapsed_s": elapsed}

    if verbose:
        print(
            f"[cider] Converted {n} layers to CiderLinear "
            f"in {elapsed:.1f}s"
        )

    return stats
