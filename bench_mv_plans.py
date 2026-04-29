#!/usr/bin/env python3
"""Benchmark: Plan A vs Plan B vs MLX quantized_matmul (decode M=1).

Plan A: weight dequant on-the-fly (uint8 -> float), FP16 activation
Plan B: activation per-token int8 quant, integer dot, per-group accumulate
MLX: mx.quantized_matmul (native W8A16 path)

All read weight as [N, K] int8 (reinterpret from uint32 packed).
"""
import sys, time
sys.path.insert(0, '/Users/ws/work/cider')
import mlx.core as mx
import numpy as np
from cider.lib._cider_prim import mv_plan_a, mv_plan_b

KERNEL_DIR = '/Users/ws/work/cider/cider/kernels'
GROUP_SIZE = 64

def make_test_data(N, K, dtype=np.float16):
    np.random.seed(42)
    num_groups = K // GROUP_SIZE
    # Create packed weights
    w_uint8 = np.random.randint(0, 255, (N, K), dtype=np.uint8)
    # Store as int8 (reinterpret) - same bit pattern
    w_int8 = w_uint8.view(np.int8)
    # Also create packed uint32 for MLX baseline
    w_packed = w_uint8.view(np.uint32).reshape(N, K // 4)
    scales = np.random.randn(N, num_groups).astype(np.float32) * 0.01
    biases = np.random.randn(N, num_groups).astype(np.float32) * 0.001
    x = np.random.randn(1, K).astype(dtype)
    return w_int8, w_packed, scales, biases, x

def reference_np(x, w_uint8, scales, biases):
    """NumPy reference per-group dequant MV."""
    N, K = w_uint8.shape
    num_groups = K // GROUP_SIZE
    x_f32 = x[0].astype(np.float32)
    y = np.zeros(N, dtype=np.float32)
    for g in range(num_groups):
        k0 = g * GROUP_SIZE
        k1 = k0 + GROUP_SIZE
        x_g = x_f32[k0:k1]
        x_sum = np.sum(x_g)
        for n in range(N):
            dot = np.dot(x_g, w_uint8[n, k0:k1].astype(np.float32))
            y[n] += scales[n, g] * dot + biases[n, g] * x_sum
    return y

# ── Correctness check ────────────────────────────────────────
print('=== Correctness check (N=128, K=256) ===')
N_test, K_test = 128, 256
w_int8, w_packed, scales, biases, x = make_test_data(N_test, K_test)
w_uint8 = w_int8.view(np.uint8)

y_ref = reference_np(x, w_uint8, scales, biases)

# Plan A
w_mx = mx.array(w_int8)
s_mx = mx.array(scales)
b_mx = mx.array(biases)
x_mx = mx.array(x)
y_a = mv_plan_a(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR)
mx.eval(y_a)
y_a_np = np.array(y_a).flatten().astype(np.float32)

# Plan B
y_b = mv_plan_b(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR)
mx.eval(y_b)
y_b_np = np.array(y_b).flatten().astype(np.float32)

# MLX baseline
wp_mx = mx.array(w_packed)
y_mlx = mx.quantized_matmul(x_mx, wp_mx, s_mx, b_mx, group_size=GROUP_SIZE, bits=8)
mx.eval(y_mlx)
y_mlx_np = np.array(y_mlx).flatten().astype(np.float32)

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f'  Plan A vs ref: cos={cos_sim(y_a_np, y_ref):.6f}, max_diff={np.max(np.abs(y_a_np - y_ref)):.4f}')
print(f'  Plan B vs ref: cos={cos_sim(y_b_np, y_ref):.6f}, max_diff={np.max(np.abs(y_b_np - y_ref)):.4f}')
print(f'  MLX   vs ref: cos={cos_sim(y_mlx_np, y_ref):.6f}, max_diff={np.max(np.abs(y_mlx_np - y_ref)):.4f}')

# ── Speed benchmark ──────────────────────────────────────────
print()
print('=== Speed benchmark (decode M=1) ===')
print(f'{"Shape":>20s}  {"MLX":>8s}  {"Plan A":>8s}  {"Plan B":>8s}  {"A/MLX":>6s}  {"B/MLX":>6s}')
print('-' * 70)

shapes = [
    (3584, 3584),    # self_attn qkv (Qwen2.5)
    (3584, 18944),   # mlp gate/up
    (18944, 3584),   # mlp down
    (2560, 2560),    # self_attn (smaller)
    (2560, 10240),   # mlp gate
    (10240, 2560),   # mlp down
]

for N, K in shapes:
    w_int8, w_packed, scales, biases, x = make_test_data(N, K)
    
    w_mx = mx.array(w_int8)
    s_mx = mx.array(scales)
    b_mx = mx.array(biases)
    x_mx = mx.array(x)
    wp_mx = mx.array(w_packed)
    
    # Warmup all
    for _ in range(20):
        mx.eval(mx.quantized_matmul(x_mx, wp_mx, s_mx, b_mx, group_size=GROUP_SIZE, bits=8))
        mx.eval(mv_plan_a(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))
        mx.eval(mv_plan_b(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR))
    
    # Benchmark MLX
    times_mlx = []
    for _ in range(100):
        t0 = time.perf_counter()
        y = mx.quantized_matmul(x_mx, wp_mx, s_mx, b_mx, group_size=GROUP_SIZE, bits=8)
        mx.eval(y)
        times_mlx.append(time.perf_counter() - t0)
    
    # Benchmark Plan A
    times_a = []
    for _ in range(100):
        t0 = time.perf_counter()
        y = mv_plan_a(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR)
        mx.eval(y)
        times_a.append(time.perf_counter() - t0)
    
    # Benchmark Plan B
    times_b = []
    for _ in range(100):
        t0 = time.perf_counter()
        y = mv_plan_b(w_mx, s_mx, b_mx, x_mx, KERNEL_DIR)
        mx.eval(y)
        times_b.append(time.perf_counter() - t0)
    
    mlx_ms = np.median(times_mlx) * 1000
    a_ms = np.median(times_a) * 1000
    b_ms = np.median(times_b) * 1000
    
    print(f'  N={N:5d} K={K:5d}  {mlx_ms:7.3f}ms  {a_ms:7.3f}ms  {b_ms:7.3f}ms  {a_ms/mlx_ms:5.2f}x  {b_ms/mlx_ms:5.2f}x')

print()
print('A/MLX < 1.0 = Plan A faster than MLX')
print('B/MLX < 1.0 = Plan B faster than MLX')
