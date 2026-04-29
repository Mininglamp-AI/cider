import sys, time
sys.path.insert(0, '/Users/ws/work/cider')
import mlx.core as mx
import numpy as np
from cider.lib._cider_prim import w8a8_linear

KERNEL_DIR = '/Users/ws/work/cider/cider/kernels'
GROUP_SIZE = 64

def bench_shape(M, N, K, iters=200, warmup=30):
    np.random.seed(42)
    a_fp = np.random.randn(M, K).astype(np.float32) * 0.1
    b_fp = np.random.randn(N, K).astype(np.float32) * 0.1
    sa = np.max(np.abs(a_fp), axis=1, keepdims=True) / 127.0
    sb = np.max(np.abs(b_fp), axis=1, keepdims=True) / 127.0
    b_int8 = np.clip(np.round(b_fp / (sb + 1e-10)), -128, 127).astype(np.int8)
    a_mx = mx.array(a_fp.astype(np.float16))
    b_int8_mx = mx.array(b_int8)
    sb_mx = mx.array(sb.flatten().astype(np.float32))
    
    num_groups = K // GROUP_SIZE
    w_uint8 = np.random.randint(0, 255, (N, K), dtype=np.uint8)
    w_packed = w_uint8.view(np.uint32).reshape(N, K // 4)
    scales_g = np.random.randn(N, num_groups).astype(np.float16) * 0.01
    biases_g = np.random.randn(N, num_groups).astype(np.float16) * 0.001
    wp_mx = mx.array(w_packed)
    sg_mx = mx.array(scales_g)
    bg_mx = mx.array(biases_g)
    x_mx = mx.array(np.random.randn(M, K).astype(np.float16))
    
    for _ in range(warmup):
        mx.eval(w8a8_linear(a_mx, b_int8_mx, sb_mx, KERNEL_DIR))
        mx.eval(mx.quantized_matmul(x_mx, wp_mx, sg_mx, bg_mx, group_size=GROUP_SIZE, bits=8))
    
    def bench(fn):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            mx.eval(fn())
            times.append(time.perf_counter() - t0)
        return np.median(times) * 1000
    
    our_ms = bench(lambda: w8a8_linear(a_mx, b_int8_mx, sb_mx, KERNEL_DIR))
    mlx_ms = bench(lambda: mx.quantized_matmul(x_mx, wp_mx, sg_mx, bg_mx, group_size=GROUP_SIZE, bits=8))
    return our_ms, mlx_ms

print('=== Prefill: INT8 TensorOps GEMM vs MLX quantized_matmul ===')
print(f'     M      N      K     Cider      MLX  Speedup')
print('-' * 55)

shapes = [
    (32, 3584, 3584),
    (32, 3584, 18944),
    (32, 18944, 3584),
    (128, 3584, 3584),
    (128, 3584, 18944),
    (256, 3584, 3584),
    (256, 3584, 18944),
    (512, 3584, 3584),
    (1024, 3584, 3584),
]

for M, N, K in shapes:
    our_ms, mlx_ms = bench_shape(M, N, K)
    speedup = mlx_ms / our_ms
    print(f'{M:6d} {N:6d} {K:6d}  {our_ms:7.3f}ms  {mlx_ms:7.3f}ms  {speedup:7.2f}x')
