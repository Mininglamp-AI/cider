#!/usr/bin/env python3
"""Full kernel benchmark: Cider per-channel & per-group vs MLX w8a16 & w4a16.

Tests M = {1, 128, 1024, 4096, 8192} across Qwen3-2B/7B shapes.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
from cider import perchannel_linear, pergroup_linear, quantize_weight_int8

WARMUP = 5
REPEAT = 20

# Qwen3-VL shapes: [N, K]
SHAPES_NK = [
    (3584,  3584),   # qkv square
    (18944, 3584),   # up-proj
    (3584,  18944),  # down-proj
    (2560,  2560),   # 2B square
    (10240, 2560),   # 2B up
    (2560,  10240),  # 2B down
]

M_VALUES = [1, 128, 1024, 4096, 8192]
GROUP_SIZE = 128


def timed(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        y = fn(); mx.eval(y)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        y = fn(); mx.eval(y)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def main():
    print(f"Full Kernel Benchmark (warmup={WARMUP}, repeat={REPEAT}, gs={GROUP_SIZE})")
    print()

    for N, K in SHAPES_NK:
        print(f"=== Shape [N={N}, K={K}] ===")
        print(f"{'M':>6s} | {'PC(ms)':>8s} {'PG(ms)':>8s} {'w8a16':>8s} {'w4a16':>8s} | {'PC/w8':>7s} {'PC/w4':>7s} {'PG/w8':>7s} {'PG/w4':>7s}")
        print("-" * 95)

        # Prepare per-channel weights
        np.random.seed(42)
        w_fp = np.random.randn(N, K).astype(np.float32)
        # Per-channel: scale per row
        absmax = np.abs(w_fp).max(axis=1, keepdims=True).clip(min=1e-8)
        scale_pc = (absmax / 127.0).astype(np.float32)
        w_int8_pc = np.clip(np.round(w_fp / scale_pc), -127, 127).astype(np.int8)
        w_int8_pc_mx = mx.array(w_int8_pc)
        scale_pc_mx = mx.array(scale_pc.squeeze())

        # Per-group weights (symmetric, gs=GROUP_SIZE)
        ng = (K + GROUP_SIZE - 1) // GROUP_SIZE
        w_pg = w_fp.copy()
        scale_pg = np.zeros((N, ng), dtype=np.float32)
        w_int8_pg = np.zeros((N, K), dtype=np.int8)
        for g in range(ng):
            k0 = g * GROUP_SIZE
            k1 = min(k0 + GROUP_SIZE, K)
            blk = w_pg[:, k0:k1]
            amax = np.abs(blk).max(axis=1, keepdims=True).clip(min=1e-8)
            s = (amax / 127.0).astype(np.float32)
            scale_pg[:, g] = s.squeeze()
            w_int8_pg[:, k0:k1] = np.clip(np.round(blk / s), -127, 127).astype(np.int8)
        w_int8_pg_mx = mx.array(w_int8_pg)
        scale_pg_mx = mx.array(scale_pg)

        # MLX w8a16 (bits=8, gs=64)
        w_mx = mx.array(w_fp.astype(np.float16))
        w8_q, w8_s, w8_b = mx.quantize(w_mx, bits=8, group_size=64)

        # MLX w4a16 (bits=4, gs=64)
        w4_q, w4_s, w4_b = mx.quantize(w_mx, bits=4, group_size=64)

        mx.eval(w_int8_pc_mx, scale_pc_mx, w_int8_pg_mx, scale_pg_mx,
                w8_q, w8_s, w8_b, w4_q, w4_s, w4_b)

        for M in M_VALUES:
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)

            t_pc = timed(lambda: perchannel_linear(x, w_int8_pc_mx, scale_pc_mx))
            t_pg = timed(lambda: pergroup_linear(x, w_int8_pg_mx, scale_pg_mx, GROUP_SIZE))
            t_w8 = timed(lambda: mx.quantized_matmul(x, w8_q, w8_s, w8_b, bits=8, group_size=64))
            t_w4 = timed(lambda: mx.quantized_matmul(x, w4_q, w4_s, w4_b, bits=4, group_size=64))

            r_pc_w8 = t_w8 / t_pc if t_pc > 0 else 0
            r_pc_w4 = t_w4 / t_pc if t_pc > 0 else 0
            r_pg_w8 = t_w8 / t_pg if t_pg > 0 else 0
            r_pg_w4 = t_w4 / t_pg if t_pg > 0 else 0

            print(f"{M:>6d} | {t_pc:>6.2f}ms {t_pg:>6.2f}ms {t_w8:>6.2f}ms {t_w4:>6.2f}ms | {r_pc_w8:>5.2f}x {r_pc_w4:>5.2f}x {r_pg_w8:>5.2f}x {r_pg_w4:>5.2f}x")

        print()


if __name__ == "__main__":
    main()
