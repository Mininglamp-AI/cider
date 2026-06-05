"""benchmarks/bench_sdpa.py — Cider SDPA vs MLX native: correctness + performance.

Usage:
    python benchmarks/bench_sdpa.py              # Default configs
    python benchmarks/bench_sdpa.py --autotune   # Run AutoTune first
    python benchmarks/bench_sdpa.py --iters 500  # More iterations for stable results
"""
import argparse
import math
import sys
import time

sys.path.insert(0, ".")

import mlx.core as mx


def bench_one(fn, q, k, v, scale, warmup, iters):
    """Benchmark a single SDPA function. Returns p10 latency in microseconds."""
    for _ in range(warmup):
        r = fn(q, k, v, scale=scale)
        mx.eval(r)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn(q, k, v, scale=scale)
        mx.eval(r)
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    p10 = times[max(0, len(times) // 10)]
    median = times[len(times) // 2]
    return p10, median


def main():
    parser = argparse.ArgumentParser(description="Cider SDPA benchmark")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--autotune", action="store_true", help="Run AutoTune before benchmark")
    parser.add_argument("--D", type=int, default=128, help="Head dimension")
    args = parser.parse_args()

    from cider.attention.sdpa import scaled_dot_product_attention, _mlx_sdpa

    # GPU info
    info = mx.metal.device_info() if hasattr(mx.metal, 'device_info') else mx.device_info()
    arch = info.get("architecture", "unknown")
    print(f"GPU: {arch}")
    print(f"MLX: {mx.__version__}")
    print(f"Config: warmup={args.warmup}, iters={args.iters}, D={args.D}")

    if args.autotune:
        import cider
        print("\n[AutoTune]")
        cider.autotune_sdpa(D=args.D, iters=args.iters // 4)
        print()

    configs = [
        # (B, H_q, H_kv, N, label)
        (1, 32, 32, 1024,  "MHA    N=1K"),
        (1, 32, 32, 2048,  "MHA    N=2K"),
        (1, 32, 32, 4096,  "MHA    N=4K"),
        (1, 32, 32, 8192,  "MHA    N=8K"),
        (1, 32, 32, 16384, "MHA    N=16K"),
        (1, 32, 32, 32768, "MHA    N=32K"),
    ]
    for H_q in [32, 64, 96, 128]:
        for gqa_factor in [2, 4, 8, 16]:
            H_kv = H_q // gqa_factor
            for N in [1024, 2048, 4096, 8192, 16384, 32768]:
                label = f"GQA{H_q}_{gqa_factor} N={N // 1024}K"
                configs.append((1, H_q, H_kv, N, label))

    D = args.D

    # ── Correctness ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CORRECTNESS (max |cider - mlx|)")
    print("=" * 72)
    print(f"{'Config':<16} {'MaxDiff':>12} {'Status':>8}")
    print("-" * 40)

    mx.random.seed(42)
    all_correct = True
    for B, Hq, Hkv, N, label in configs:
        q = mx.random.normal((B, Hq, 1, D)).astype(mx.float16)
        k = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
        scale = 1.0 / math.sqrt(D)

        ref = _mlx_sdpa(q, k, v, scale=scale)
        out = scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(ref, out)

        diff = mx.abs(ref - out).max().item()
        ok = diff < 0.01
        if not ok:
            all_correct = False
        status = "PASS" if ok else "FAIL"
        print(f"{label:<16} {diff:>12.6f} {status:>8}")

    print()
    if all_correct:
        print(">>> ALL CORRECT <<<")
    else:
        print(">>> SOME FAILED <<<")
        sys.exit(1)

    # ── Performance ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"PERFORMANCE (p10 latency, {args.iters} iters)")
    print("=" * 72)
    print(f"{'Config':<16} {'MLX(us)':>10} {'Cider(us)':>10} {'Speedup':>10} {'Status':>8}")
    print("-" * 58)

    wins = ties = losses = 0

    mx.random.seed(42)
    for B, Hq, Hkv, N, label in configs:
        q = mx.random.normal((B, Hq, 1, D)).astype(mx.float16)
        k = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
        scale = 1.0 / math.sqrt(D)
        mx.eval(q, k, v)

        mlx_p10, _ = bench_one(_mlx_sdpa, q, k, v, scale, args.warmup, args.iters)
        cider_p10, _ = bench_one(scaled_dot_product_attention, q, k, v, scale, args.warmup, args.iters)

        ratio = mlx_p10 / cider_p10
        if ratio >= 1.02:
            status = "WIN"
            wins += 1
        elif ratio >= 0.98:
            status = "TIE"
            ties += 1
        else:
            status = "LOSS"
            losses += 1

        print(f"{label:<16} {mlx_p10:>10.1f} {cider_p10:>10.1f} {ratio:>9.2f}x {status:>8}")

    print("-" * 58)
    print(f"Summary: {wins} wins / {ties} ties / {losses} losses")
    print()

    if losses == 0:
        print(">>> NO REGRESSIONS <<<")
    else:
        print(f">>> {losses} REGRESSION(S) DETECTED <<<")


if __name__ == "__main__":
    main()
