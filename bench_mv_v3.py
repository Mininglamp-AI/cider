#!/usr/bin/env python3
"""Benchmark: Plan A v2 vs Plan A v3 (tiled + direct) vs MLX quantized_matmul.

Focus on large-K shapes where K-tile swizzle should help L2 cache.
"""
import sys, time
sys.path.insert(0, '/Users/ws/work/cider')
import mlx.core as mx
import numpy as np
from cider.lib._cider_prim import mv_plan_a, mv_plan_a_tiled, mv_plan_a_direct

KERNEL_DIR = '/Users/ws/work/cider/cider/kernels'
GROUP_SIZE = 64

def make_test_data(N, K, dtype=np.float16):
    np.random.seed(42)
    num_groups = K // GROUP_SIZE
    w_uint8 = np.random.randint(0, 255, (N, K), dtype=np.uint8)
    w_int8 = w_uint8.view(np.int8)
    w_packed = w_uint8.view(np.uint32).reshape(N, K // 4)
    scales = np.random.randn(N, num_groups).astype(np.float32) * 0.01
    biases = np.random.randn(N, num_groups).astype(np.float32) * 0.001
    x = np.random.randn(1, K).astype(dtype)
    return w_int8, w_packed, scales, biases, x

# ── Correctness check for tiled kernel ────────────────────
print('=== Correctness check (tiled vs v2) ===')
for N, K in [(128, 256), (128, 4096), (3584, 18944)]:
    w_int8, w_packed, scales, biases, x = make_test_data(N, K)
    w_mx = mx.array(w_int8)
    s_mx = mx.array(scales)
    b_mx = mx.array(biases)
    x_mx = mx.array(x)
    wp_mx = mx.array(w_packed)

    # Reference: MLX quantized_matmul
    y_ref = mx.quantized_matmul(x_mx, wp_mx, s_mx, b_mx, group_size=GROUP_SIZE, bits=8)
    mx.eval(y_ref)
    y_ref_np = np.array(y_ref).flatten().astype(np.float32)

    # Plan A v2 (original)
    y_a = mv_plan_a(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR)
    mx.eval(y_a)
    y_a_np = np.array(y_a).flatten().astype(np.float32)

    # Plan A tiled (v3) — output is float32
    y_t = mv_plan_a_tiled(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR)
    mx.eval(y_t)
    y_t_np = np.array(y_t).flatten().astype(np.float32)

    # Plan A direct (v3, NUM_SG=4)
    y_d = mv_plan_a_direct(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR)
    mx.eval(y_d)
    y_d_np = np.array(y_d).flatten().astype(np.float32)

    def cos(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)

    print(f'  N={N:5d} K={K:5d}:')
    print(f'    v2 vs MLX:     cos={cos(y_a_np, y_ref_np):.6f} maxd={np.max(np.abs(y_a_np - y_ref_np)):.4f}')
    print(f'    tiled vs MLX:  cos={cos(y_t_np, y_ref_np):.6f} maxd={np.max(np.abs(y_t_np - y_ref_np)):.4f}')
    print(f'    direct vs MLX: cos={cos(y_d_np, y_ref_np):.6f} maxd={np.max(np.abs(y_d_np - y_ref_np)):.4f}')

# ── Speed benchmark ──────────────────────────────────────────
print()
print('=== Speed benchmark (decode M=1) ===')
print(f'{"Shape":>20s}  {"MLX":>8s}  {"v2":>8s}  {"v3-tile":>8s}  {"v3-dir":>8s}  {"tile/MLX":>8s}  {"tile/v2":>8s}')
print('-' * 90)

shapes = [
    (3584, 3584),    # self_attn
    (3584, 18944),   # mlp gate/up (large K — target for tiling!)
    (18944, 3584),   # mlp down (large N)
    (2560, 2560),    
    (2560, 10240),   # medium K
    (10240, 2560),   # medium N
]

for N, K in shapes:
    w_int8, w_packed, scales, biases, x = make_test_data(N, K)
    w_mx = mx.array(w_int8)
    s_mx = mx.array(scales)
    b_mx = mx.array(biases)
    x_mx = mx.array(x)
    wp_mx = mx.array(w_packed)

    # Warmup
    for _ in range(20):
        mx.eval(mx.quantized_matmul(x_mx, wp_mx, s_mx, b_mx, group_size=GROUP_SIZE, bits=8))
        mx.eval(mv_plan_a(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))
        mx.eval(mv_plan_a_tiled(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))
        mx.eval(mv_plan_a_direct(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))

    def bench(fn, iters=100):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            y = fn()
            mx.eval(y)
            times.append(time.perf_counter() - t0)
        return np.median(times) * 1000

    mlx_ms = bench(lambda: mx.quantized_matmul(x_mx, wp_mx, s_mx, b_mx, group_size=GROUP_SIZE, bits=8))
    v2_ms = bench(lambda: mv_plan_a(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))
    tile_ms = bench(lambda: mv_plan_a_tiled(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))
    dir_ms = bench(lambda: mv_plan_a_direct(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))

    print(f'  N={N:5d} K={K:5d}  {mlx_ms:7.3f}ms  {v2_ms:7.3f}ms  {tile_ms:7.3f}ms  {dir_ms:7.3f}ms  {tile_ms/mlx_ms:7.2f}x  {tile_ms/v2_ms:7.2f}x')

print()
print('tile/MLX < 1.0 = tiled faster than MLX')
print('tile/v2  < 1.0 = tiled faster than v2 (non-tiled)')
