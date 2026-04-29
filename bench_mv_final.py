#!/usr/bin/env python3
"""Final benchmark: v2 vs v4 (vectorized) vs MLX."""
import sys, time
sys.path.insert(0, '/Users/ws/work/cider')
import mlx.core as mx
import numpy as np
from cider.lib._cider_prim import mv_plan_a, mv_plan_a_v4

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

# Correctness
print('=== Correctness v4 ===')
for N, K in [(128, 256+256), (3584, 18944)]:
    w_int8, w_packed, scales, biases, x = make_test_data(N, K)
    w_mx = mx.array(w_int8)
    s_mx = mx.array(scales)
    b_mx = mx.array(biases)
    x_mx = mx.array(x)
    wp_mx = mx.array(w_packed)

    y_ref = mx.quantized_matmul(x_mx, wp_mx, s_mx, b_mx, group_size=GROUP_SIZE, bits=8)
    mx.eval(y_ref)
    y_ref_np = np.array(y_ref).flatten().astype(np.float32)

    y_v4 = mv_plan_a_v4(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR)
    mx.eval(y_v4)
    y_v4_np = np.array(y_v4).flatten().astype(np.float32)

    cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    print(f'  N={N} K={K}: cos={cos(y_v4_np, y_ref_np):.6f} maxd={np.max(np.abs(y_v4_np - y_ref_np)):.4f}')

# Speed
print()
print('=== Speed benchmark (decode M=1) ===')
print(f'{"Shape":>20s}  {"MLX":>8s}  {"v2":>8s}  {"v4":>8s}  {"v4/MLX":>8s}')
print('-' * 60)

shapes = [
    (3584, 3584),
    (3584, 18944),
    (18944, 3584),
    (2560, 2560),
    (2560, 10240),
    (10240, 2560),
]

for N, K in shapes:
    w_int8, w_packed, scales, biases, x = make_test_data(N, K)
    w_mx = mx.array(w_int8)
    s_mx = mx.array(scales)
    b_mx = mx.array(biases)
    x_mx = mx.array(x)
    wp_mx = mx.array(w_packed)

    for _ in range(30):
        mx.eval(mx.quantized_matmul(x_mx, wp_mx, s_mx, b_mx, group_size=GROUP_SIZE, bits=8))
        mx.eval(mv_plan_a(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))
        mx.eval(mv_plan_a_v4(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))

    def bench(fn, iters=200):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            mx.eval(fn())
            times.append(time.perf_counter() - t0)
        return np.median(times) * 1000

    mlx_ms = bench(lambda: mx.quantized_matmul(x_mx, wp_mx, s_mx, b_mx, group_size=GROUP_SIZE, bits=8))
    v2_ms = bench(lambda: mv_plan_a(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))
    v4_ms = bench(lambda: mv_plan_a_v4(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))

    print(f'  N={N:5d} K={K:5d}  {mlx_ms:7.3f}ms  {v2_ms:7.3f}ms  {v4_ms:7.3f}ms  {v4_ms/mlx_ms:7.2f}x')
