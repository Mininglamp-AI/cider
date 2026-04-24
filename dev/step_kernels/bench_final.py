#!/usr/bin/env python3
"""Step-by-step kernel benchmark. Run each step sequentially with gc."""
import sys, os, time, gc
import numpy as np
sys.path.insert(0, "~/work/cider/cider/lib")
import mlx.core as mx
import _cider_prim as ext

DEV = "~/work/cider/dev/step_kernels"
STEPS = ["step1", "step2", "step3", "step4", "step5"]
WARMUP = 5
REPEAT = 50

# Large tile shapes (M%128==0)
LARGE = [
    (128, 4096, 4096),
    (256, 4096, 4096),
    (128, 3584, 18944),
    (256, 3584, 18944),
]

# Small tile shapes (step3+ only)
SMALL = [
    (1, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (64, 4096, 4096),
]

def bench_one(A_mx, B_mx, kernel_dir):
    """Warmup + benchmark, return median ms."""
    for _ in range(WARMUP):
        mx.eval(ext.int8_matmul_int32(A_mx, B_mx, kernel_dir))
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        mx.eval(ext.int8_matmul_int32(A_mx, B_mx, kernel_dir))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]

# === Large tile benchmark ===
print("=" * 90)
print("Part 1: Large Tile (M>=128)")
print("=" * 90)

large_results = {s: {} for s in STEPS}

for M, K, N in LARGE:
    key = f"({M},{K},{N})"
    np.random.seed(42)
    A_np = np.random.randint(-127, 128, (M, K), dtype=np.int8)
    B_np = np.random.randint(-127, 128, (K, N), dtype=np.int8)
    A_mx = mx.array(A_np)
    B_mx = mx.array(B_np)
    mx.eval(A_mx, B_mx)
    del A_np, B_np
    gc.collect()

    for step in STEPS:
        kdir = os.path.join(DEV, step)
        med = bench_one(A_mx, B_mx, kdir)
        large_results[step][key] = med
        print(f"  {step} {key}: {med:.4f} ms")

    del A_mx, B_mx
    mx.clear_cache()
    gc.collect()

# Print large table
print()
hdr = f"{'Shape':<22}"
for s in STEPS:
    hdr += f" | {s:>10}"
print(hdr)
print("-" * 82)
for M, K, N in LARGE:
    key = f"({M},{K},{N})"
    row = f"{key:<22}"
    for s in STEPS:
        v = large_results[s].get(key, -1)
        row += f" | {v:>10.3f}"
    print(row)

# === Small tile benchmark (step3/4/5 only) ===
print()
print("=" * 60)
print("Part 2: Small Tile (M<128, step3/4/5)")
print("=" * 60)

small_steps = ["step3", "step4", "step5"]
small_results = {s: {} for s in small_steps}

for M, K, N in SMALL:
    key = f"({M},{K},{N})"
    np.random.seed(42)
    A_np = np.random.randint(-127, 128, (M, K), dtype=np.int8)
    B_np = np.random.randint(-127, 128, (K, N), dtype=np.int8)
    A_mx = mx.array(A_np)
    B_mx = mx.array(B_np)
    mx.eval(A_mx, B_mx)
    del A_np, B_np
    gc.collect()

    for step in small_steps:
        kdir = os.path.join(DEV, step)
        med = bench_one(A_mx, B_mx, kdir)
        small_results[step][key] = med
        print(f"  {step} {key}: {med:.4f} ms")

    del A_mx, B_mx
    mx.clear_cache()
    gc.collect()

# Print small table
print()
hdr = f"{'Shape':<22}"
for s in small_steps:
    hdr += f" | {s:>10}"
print(hdr)
print("-" * 58)
for M, K, N in SMALL:
    key = f"({M},{K},{N})"
    row = f"{key:<22}"
    for s in small_steps:
        v = small_results[s].get(key, -1)
        row += f" | {v:>10.3f}"
    print(row)

print("\nDone!")
