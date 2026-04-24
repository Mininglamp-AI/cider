#!/usr/bin/env python3
"""Kernel-level benchmark: W8A8 vs W4A8 vs MLX W4A16.

Measures pure kernel latency (excluding weight quantization).
All three paths go through the same mx.eval() pattern for fair comparison.

Usage:
    cd /path/to/cider
    python benchmarks/bench_kernel.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
from cider import w8a8_linear, w4a8_linear, quantize_weight_int8, pack_weight_int4

WARMUP = 5
REPEAT = 20

# Realistic shapes: (M, K, N) covering prefill & decode scenarios
# Qwen3-VL-2B dims: hidden=1536, intermediate=8960, head_dim=128
# Qwen3-VL-7B dims: hidden=3584, intermediate=18944
SHAPES = [
    # --- Square-ish (K=N) ---
    (1,   4096, 4096),
    (4,   4096, 4096),
    (16,  4096, 4096),
    (32,  4096, 4096),
    (64,  4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    # --- Asymmetric K != N (MLP-like) ---
    (1,   3584, 18944),
    (16,  3584, 18944),
    (64,  3584, 18944),
    (128, 3584, 18944),
    (256, 3584, 18944),
    (512, 3584, 18944),
    # --- Asymmetric N < K (down-projection) ---
    (1,   18944, 3584),
    (16,  18944, 3584),
    (64,  18944, 3584),
    (128, 18944, 3584),
    (256, 18944, 3584),
    (512, 18944, 3584),
    # --- Small model dims (2B-class) ---
    (16,  1536, 8960),
    (64,  1536, 8960),
    (128, 1536, 8960),
    (256, 1536, 8960),
    # --- QKV projection (multi-head) ---
    (16,  3584, 512),
    (64,  3584, 512),
    (128, 3584, 512),
    (256, 3584, 512),
    # --- Large batch ---
    (1024, 4096, 4096),
    (2048, 4096, 4096),
]


def bench_w8a8(x, w_int8, scale_w):
    for _ in range(WARMUP):
        y = w8a8_linear(x, w_int8, scale_w)
        mx.eval(y)
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        y = w8a8_linear(x, w_int8, scale_w)
        mx.eval(y)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def bench_w4a8(x, packed_w, scale_w):
    for _ in range(WARMUP):
        y = w4a8_linear(x, packed_w, scale_w)
        mx.eval(y)
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        y = w4a8_linear(x, packed_w, scale_w)
        mx.eval(y)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def bench_mlx_w4a16(x, w_mlx, scales_mlx, biases_mlx):
    for _ in range(WARMUP):
        y = mx.quantized_matmul(x, w_mlx, scales_mlx, biases_mlx)
        mx.eval(y)
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        y = mx.quantized_matmul(x, w_mlx, scales_mlx, biases_mlx)
        mx.eval(y)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def main():
    print(f"Kernel Benchmark (warmup={WARMUP}, repeat={REPEAT})")
    print(f"{'M':>5s} {'K':>6s} {'N':>6s} | {'W8A8':>8s} | {'W4A8':>8s} | {'W4A16':>8s} | {'W8A8/W4A16':>11s} | {'W4A8/W4A16':>11s}")
    print("-" * 82)

    prev_kn = None
    for M, K, N in SHAPES:
        # Prepare weights (re-create when K/N changes)
        if (K, N) != prev_kn:
            np.random.seed(42)
            w_fp32 = np.random.randn(K, N).astype(np.float32)

            w_int8_np, scale_w8_np = quantize_weight_int8(w_fp32)
            w_int8 = mx.array(w_int8_np)
            scale_w8 = mx.array(scale_w8_np)

            packed_np, scale_w4_np = pack_weight_int4(w_fp32)
            packed_w = mx.array(packed_np)
            scale_w4 = mx.array(scale_w4_np)

            # MLX quantize expects (N, K) layout; quantized_matmul transposes internally
            w_mx = mx.array(w_fp32.T.astype(np.float16))
            w_q, scales_q, biases_q = mx.quantize(w_mx, bits=4, group_size=64)
            mx.eval(w_int8, scale_w8, packed_w, scale_w4, w_q, scales_q, biases_q)
            prev_kn = (K, N)

        x = mx.random.normal((M, K)).astype(mx.float16)
        mx.eval(x)

        t_w8a8 = bench_w8a8(x, w_int8, scale_w8)
        t_w4a8 = bench_w4a8(x, packed_w, scale_w4)
        t_w4a16 = bench_mlx_w4a16(x, w_q, scales_q, biases_q)

        sp_w8 = t_w4a16 / t_w8a8 if t_w8a8 > 0 else 0
        sp_w4 = t_w4a16 / t_w4a8 if t_w4a8 > 0 else 0
        print(f"{M:>5d} {K:>6d} {N:>6d} | {t_w8a8:>6.2f}ms | {t_w4a8:>6.2f}ms | {t_w4a16:>6.2f}ms | {sp_w8:>9.2f}x  | {sp_w4:>9.2f}x")


if __name__ == "__main__":
    main()
