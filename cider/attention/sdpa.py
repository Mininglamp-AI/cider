"""cider.attention.sdpa — Compiled SDPA with monkey-patch support.

Uses precompiled Metal kernels via C++ Custom Primitive (zero JIT overhead).
Falls back to MLX default for unsupported cases.
"""

import math
import json
import os
from typing import Optional

import mlx.core as mx


# ── Kernel directory (where .metal/.h files live) ───────────────
def _get_kernel_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "kernels")


# ── C++ primitive import ────────────────────────────────────────
_prim = None

def _get_prim():
    global _prim
    if _prim is None:
        try:
            from cider.lib import _cider_prim
            _prim = _cider_prim
        except ImportError:
            raise RuntimeError(
                "cider C++ extension not built. "
                "Run: pip install -e . (from cider root)"
            )
    return _prim


# ── GPU arch ────────────────────────────────────────────────────
_gpu_arch = None

def _get_gpu_arch():
    global _gpu_arch
    if _gpu_arch is None:
        info = mx.device_info() if hasattr(mx, 'device_info') else mx.metal.device_info()
        _gpu_arch = info.get("architecture", "unknown")
    return _gpu_arch


# ── Routing ─────────────────────────────────────────────────────
def _use_cider(N: int, gqa_factor: int) -> bool:
    """Decide whether Cider kernel beats MLX default.

    Compiled primitive has near-zero dispatch overhead (unlike JIT).
    Our kernel uses contiguous chunk + TILE=4 which benefits all configs.
    Always route through Cider for decode (Q_seq=1) — the calling code
    already handles mask/prefill fallback before reaching here.
    """
    return True


def _use_2pass(N: int, gqa_factor: int) -> bool:
    return N >= 1024


def _select_blocks(N: int, gqa_factor: int, H_q: int, arch: str) -> int:
    cached = _autotune_lookup(N, gqa_factor, H_q, arch)
    if cached is not None:
        return cached
    if gqa_factor >= 8:
        if N >= 8192:
            return 128
        elif N >= 4096:
            return 64
        else:
            return 32
    elif gqa_factor >= 4:
        if N >= 8192:
            return 64
        else:
            return 32
    else:
        if N >= 16384:
            return 128
        elif N >= 4096:
            return 64
        else:
            return 32


# ── Core SDPA ───────────────────────────────────────────────────
def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
    **kwargs,
) -> mx.array:
    """Optimized SDPA for decode (q_seq_len=1).

    Drop-in replacement for mx.fast.scaled_dot_product_attention.
    Falls back to MLX for unsupported cases (mask, non-decode, prefill).
    Accepts **kwargs for forward-compat with new MLX args (e.g. sinks).
    """
    if mask is not None:
        return _mlx_sdpa(q, k, v, scale=scale, mask=mask, **kwargs)

    B, H_q, Q_seq, D = q.shape
    _, H_kv, N, _ = k.shape

    if Q_seq != 1:
        return _mlx_sdpa(q, k, v, scale=scale, **kwargs)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    gqa_factor = H_q // H_kv

    # Fall back to MLX when Cider doesn't help
    if not _use_cider(N, gqa_factor):
        return _mlx_sdpa(q, k, v, scale=scale, **kwargs)

    prim = _get_prim()
    kernel_dir = _get_kernel_dir()

    if not _use_2pass(N, gqa_factor):
        # ── 1-pass ──
        # Flatten to [B*H_q, D] for queries, [B*H_kv, N, D] for K/V
        q_flat = q.reshape(B * H_q, D)
        k_cont = k.reshape(B * H_kv, N, D)
        v_cont = v.reshape(B * H_kv, N, D)

        out_flat = prim.cider_sdpa_1pass(
            q_flat, k_cont, v_cont,
            gqa_factor, scale, kernel_dir,
        )
        return out_flat.reshape(B, H_q, 1, D)

    else:
        # ── 2-pass ──
        arch = _get_gpu_arch()
        blocks = _select_blocks(N, gqa_factor, H_q, arch)

        out = prim.cider_sdpa_2pass(
            q, k, v,
            gqa_factor, blocks, scale, kernel_dir,
        )
        return out


# ── Monkey-patch ────────────────────────────────────────────────
_mlx_sdpa = mx.fast.scaled_dot_product_attention  # save original
_patched = False


def patch_sdpa(verbose: bool = True):
    """Monkey-patch mx.fast.scaled_dot_product_attention with Cider's optimized version.

    Usage:
        import cider
        cider.patch_sdpa()
        # Now all calls to mx.fast.scaled_dot_product_attention use Cider kernels.
        # mlx_lm, mlx_vlm etc. benefit automatically.
    """
    global _patched
    if _patched:
        if verbose:
            print("[cider] SDPA already patched.")
        return

    # Load AutoTune cache if available
    n = load_autotune()

    mx.fast.scaled_dot_product_attention = scaled_dot_product_attention
    _patched = True

    if verbose:
        arch = _get_gpu_arch()
        tune_msg = f", {n} tuned configs loaded" if n > 0 else ""
        print(f"[cider] SDPA patched — compiled kernels, {arch}{tune_msg}")


def unpatch_sdpa(verbose: bool = True):
    """Restore original mx.fast.scaled_dot_product_attention."""
    global _patched
    if not _patched:
        return
    mx.fast.scaled_dot_product_attention = _mlx_sdpa
    _patched = False
    if verbose:
        print("[cider] SDPA unpatched — restored MLX default.")


# ── AutoTune ────────────────────────────────────────────────────
_autotune_cache = {}
_AUTOTUNE_FILE = os.path.expanduser("~/.cider_sdpa_tune.json")
_BLOCKS_CANDIDATES = [32, 64, 128]


def _autotune_lookup(N: int, gqa_factor: int, H_q: int, arch: str) -> Optional[int]:
    """Look up cached blocks value. Tries exact H_q match first, then generic."""
    key = f"{arch}_gqa{gqa_factor}_H{H_q}_N{N}"
    val = _autotune_cache.get(key)
    if val is not None:
        return val
    # Fallback: try without H_q (legacy cache or generic entry)
    key_generic = f"{arch}_gqa{gqa_factor}_N{N}"
    return _autotune_cache.get(key_generic)


def load_autotune() -> int:
    """Load cached AutoTune results from disk."""
    if os.path.exists(_AUTOTUNE_FILE):
        with open(_AUTOTUNE_FILE) as f:
            data = json.load(f)
        _autotune_cache.update(data)
        return len(data)
    return 0


def save_autotune():
    """Save AutoTune cache to disk."""
    with open(_AUTOTUNE_FILE, 'w') as f:
        json.dump(_autotune_cache, f, indent=2)


def _bench_one(q, k, v, scale, blocks, warmup=10, iters=50):
    """Time a single (blocks) config. Returns median microseconds."""
    import time
    prim = _get_prim()
    kernel_dir = _get_kernel_dir()
    gqa_factor = q.shape[1] // k.shape[1]

    # Warmup
    for _ in range(warmup):
        out = prim.cider_sdpa_2pass(
            q, k, v, gqa_factor, blocks, scale, kernel_dir)
        mx.eval(out)

    # Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = prim.cider_sdpa_2pass(
            q, k, v, gqa_factor, blocks, scale, kernel_dir)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    # Return p10 (robust against outliers)
    idx = max(0, len(times) // 10)
    return times[idx]


def autotune_sdpa(
    gqa_factors: Optional[list] = None,
    seq_lens: Optional[list] = None,
    D: int = 128,
    warmup: int = 10,
    iters: int = 50,
    verbose: bool = True,
) -> dict:
    """Run AutoTune sweep for 2-pass SDPA blocks parameter.

    Sweeps blocks in [32, 64, 128] for each (gqa_factor, N) config.
    Results cached to ~/.cider_sdpa_tune.json.

    Args:
        gqa_factors: list of GQA ratios to tune (default: [1,2,4,8])
        seq_lens: list of sequence lengths (default: [1024,2048,4096,8192,16384,32768])
        D: head dimension (default: 128)
        warmup: warmup iterations per config
        iters: benchmark iterations per config
        verbose: print results

    Returns:
        dict mapping config keys to best blocks values
    """
    if gqa_factors is None:
        gqa_factors = [1, 2, 4, 8]
    if seq_lens is None:
        seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]

    # Typical H_q values across popular models:
    #   32: Qwen2.5-7B, LLaMA-3-8B, Gemma-2-9B, SmolLM
    #   96: Qwen3-32B, Qwen2.5-72B
    #  112: LLaMA-3.1-70B
    _TYPICAL_HQ = [32, 96, 112]

    arch = _get_gpu_arch()
    results = {}

    if verbose:
        print(f"[cider] AutoTune starting — arch={arch}, D={D}")
        print(f"  GQA factors: {gqa_factors}")
        print(f"  H_q values:  {_TYPICAL_HQ}")
        print(f"  Seq lengths: {seq_lens}")
        print(f"  Blocks candidates: {_BLOCKS_CANDIDATES}")
        print()

    mx.random.seed(42)

    for gqa in gqa_factors:
        for H_q in _TYPICAL_HQ:
            H_kv = H_q // gqa if gqa > 1 else H_q
            if H_kv < 1:
                continue  # Skip invalid combos (e.g., H_q=32, GQA=96)
            if H_q % gqa != 0 and gqa > 1:
                continue  # H_q must be divisible by gqa_factor
            B = 1
            scale = 1.0 / math.sqrt(D)

            for N in seq_lens:
                if N < 1024:
                    continue  # 1-pass, no blocks parameter

                q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
                k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
                v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
                mx.eval(q, k, v)

                best_blocks = _BLOCKS_CANDIDATES[0]
                best_time = float('inf')
                timings = {}

                for blocks in _BLOCKS_CANDIDATES:
                    t = _bench_one(q, k, v, scale, blocks, warmup, iters)
                    timings[blocks] = t
                    if t < best_time:
                        best_time = t
                        best_blocks = blocks

                key = f"{arch}_gqa{gqa}_H{H_q}_N{N}"
                _autotune_cache[key] = best_blocks
                results[key] = best_blocks

                if verbose:
                    details = " | ".join(
                        f"b{b}={timings[b]:.0f}us{'*' if b == best_blocks else ''}"
                        for b in _BLOCKS_CANDIDATES
                    )
                    print(f"  GQA{gqa} H{H_q:<3} N={N:>6}: {details} -> blocks={best_blocks}")

    # Save to disk
    save_autotune()

    if verbose:
        print(f"\n[cider] AutoTune done — {len(results)} configs saved to {_AUTOTUNE_FILE}")

    return results
