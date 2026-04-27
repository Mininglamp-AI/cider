# Cider W8A8 Kernel vs MLX INT8 GEMM: A Transparent Comparison

## TL;DR

**MLX's Steel NAX GEMM templates are fully generic** — they can be instantiated with `<int8_t, int8_t, int32_t>` to produce a working INT8×INT8→INT32 matmul using the same `mpp::tensor_ops::matmul2d` hardware instruction as Cider. At the raw matmul level, both kernels achieve **comparable throughput** at compute-bound sizes (within 3–5% at M ≥ 512).

Cider's contribution is **not** a novel matmul kernel. It is a **complete W8A8 inference pipeline** — fused quantization, matmul, and dequantization — that MLX does not provide.

## Background: What MLX Does Today

MLX (as of v0.31) uses INT8 TensorOps **nowhere** in its codebase:

- **Standard matmul** (`steel_gemm_fused_nax.metal`): Instantiated for `float16`, `bfloat16`, `float32` only. No `int8` instantiation exists.
- **Quantized matmul** (`quantized_nax.h`): Dequantizes packed int{2,3,4,5,6,8} weights → FP16, then uses **FP16 TensorOps**. This is weight-only quantization — activations stay in FP16.

MLX *could* instantiate `int8` GEMM — the templates are generic. They simply choose not to, because a raw INT8 matmul alone isn't useful without the surrounding quantize/dequant pipeline.

## The Benchmark

We wrote a standalone INT8 GEMM kernel in MLX's NAX style, compiled via `mx.fast.metal_kernel`, using the same:
- `matmul2d<int8_t, int8_t, int32_t>` TensorOps instruction
- BM=128, BN=128, BK=512 tile configuration
- NAXFrag cooperative tensor layout
- Device memory direct load (no threadgroup staging)

This is **not** a hypothetical comparison — the MLX-style kernel compiles, runs, and produces bit-exact results (verified: `max_diff=0`).

### Three-Way Comparison (Apple M5 Pro, K=4096, N=4096, Warmup=20, Repeat=50)

| M | Cider Full Pipeline | Cider Raw INT8 | MLX-style INT8 | Full/Raw | Raw/MLX |
|---|-------------------|---------------|----------------|----------|---------|
| 64 | 0.50ms | 0.51ms | 0.77ms | 1.01x | 0.67x |
| 128 | 0.53ms | 0.44ms | 0.33ms | 1.14x | 1.33x |
| 256 | 0.45ms | 0.40ms | 0.37ms | 1.11x | 1.08x |
| 512 | 0.59ms | 0.54ms | 0.52ms | 1.07x | 1.03x |
| 1024 | 0.96ms | 0.90ms | 0.87ms | 1.07x | 1.03x |
| 2048 | 1.66ms | 1.55ms | 1.49ms | 1.07x | 1.04x |

**MLP shapes (Llama-3 8B dimensions)**:

| M | K | N | Cider Full | Cider Raw | MLX-style | Full/Raw | Raw/MLX |
|---|---|---|-----------|-----------|-----------|----------|---------|
| 128 | 3584 | 18944 | 0.75ms | 0.77ms | 0.70ms | 0.98x | 1.09x |
| 512 | 3584 | 18944 | 1.66ms | 1.65ms | 1.57ms | 1.00x | 1.05x |
| 128 | 18944 | 3584 | 0.84ms | 1.07ms | 0.76ms | 0.79x | 1.41x |
| 512 | 18944 | 3584 | 1.90ms | 1.82ms | 1.69ms | 1.04x | 1.08x |

Where:
- **Cider Full Pipeline** = `w8a8_linear()`: FP16→INT8 quantize + INT8 matmul + fused INT32→FP16 dequant
- **Cider Raw INT8** = `int8_matmul_int32()`: Pure INT8×INT8→INT32 (no quantize, no dequant)
- **MLX-style INT8** = Standalone NAX kernel via `mx.fast.metal_kernel`: Same pure INT8→INT32

## What the Numbers Tell Us

### 1. Raw INT8 matmul throughput converges at compute-bound sizes

At M ≥ 512 (compute-bound), Cider Raw and MLX-style differ by ≤ 4%. Both execute the same `matmul2d(16, 32, 16)` TensorOps instruction — the hardware doesn't care who wrote the kernel.

### 2. Small-M behavior differs due to dispatch and tile scheduling

- **M=64**: Cider is faster (0.51 vs 0.77ms) — Cider uses a dedicated small-tile kernel (BM=32, 128 threads) while the MLX-style kernel uses the same BM=128 config, wasting occupancy.
- **M=128**: MLX-style is faster (0.33 vs 0.44ms) — At M=BM=128, exactly 1 row of tiles. Cider's L2 swizzle pattern (designed for multi-row grids) has suboptimal dispatch for this boundary case. This is a known optimization opportunity.

### 3. The full pipeline costs only 5–15% over raw matmul

Cider Full includes two extra operations that raw matmul doesn't:
- **Per-token INT8 quantization**: FP16 activations → INT8 + per-token scale (separate kernel)
- **Fused dequantization**: INT32 accumulator × scale_act × scale_weight → FP16 (fused in matmul store)

At M=2048, this overhead is just 7% (1.66ms vs 1.55ms). This overhead is inherent to **any** W8A8 implementation — you cannot skip quantization or dequantization.

## Why Cider Implements Its Own Kernel

Given that MLX's templates could produce an identical INT8 matmul, why does Cider write its own?

### 1. Fused dequantization in the store phase

MLX's NAX GEMM store writes raw typed output (`static_cast<OutType>(acc)`). W8A8 inference requires:

```
C_fp16[i][j] = C_int32[i][j] * scale_act[i] * scale_weight[j]
```

Cider fuses this multiplication into the store phase of the matmul kernel. Without fusion, you'd need a separate elementwise kernel — adding one extra device memory round-trip (~0.1–0.2ms at M=2048).

### 2. Quantize + matmul in one command encoder

Cider packages activation quantization (FP16→INT8) and the INT8 matmul as a single MLX primitive. One C++ `eval()` call dispatches both Metal kernels in the same command encoder. Using MLX's standard matmul op for INT8 would require two separate graph nodes with potential scheduling gaps.

### 3. MLX's matmul dispatch doesn't support INT8

`mlx/backend/common/compiled.cpp` and `Matmul.cpp` dtype switches only handle floating-point types. Even if you instantiated the INT8 GEMM Metal shader, MLX's C++ `matmul()` op wouldn't route to it. You'd need either:
- Modifying MLX source (impractical for users)
- Using `mx.fast.metal_kernel` (works, but adds Python dispatch overhead and loses graph fusion)

Cider solves this by implementing INT8 matmul as a custom primitive (`mlx::core::Primitive` subclass), bypassing MLX's type restrictions while still integrating with its lazy evaluation graph.

## Reproducing

```bash
cd /path/to/cider
python benchmarks/bench_cider_vs_mlx_int8.py
```

Requires: Apple M5+, MLX ≥ 0.31, Cider installed.

The MLX-style kernel is compiled at runtime via `mx.fast.metal_kernel` with Metal 4 + MPP headers. No Xcode installation required.
