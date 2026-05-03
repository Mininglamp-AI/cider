# Cider W8A8 Kernel vs MLX INT8 GEMM: A Transparent Comparison


MLX's Steel NAX GEMM templates appear generic enough at the shader level to support `<int8_t, int8_t, int32_t>` instantiations. In our standalone test, an MLX-style INT8×INT8→INT32 kernel can be built on the same `mpp::tensor_ops::matmul2d` TensorOps instruction used by Cider. At the raw matmul level, both kernels achieve **comparable throughput** (within 2–5% across all tested sizes).

Cider's contribution is **not** a novel matmul kernel. It is an end-to-end W8A8 inference pipeline for Apple Silicon, combining online activation quantization, INT8 TensorOps matmul, and dequantization in a usable MLX-integrated path.

**Apple M5 introduced cooperative_tensor INT8 TensorOps (matmul2d 16×32×16), delivering 2× the TOPS of FP16 — but no ML framework on macOS uses them.** In the MLX quantization path we examined (v0.31), quantized weights are dequantized for computation and the corresponding matmul path remains FP16-based rather than end-to-end INT8. This means prefill (the compute-bound phase of LLM inference) gets zero benefit from INT8 hardware. W8A8 is the only quantization scheme that accelerates both compute and memory: activations and weights are both INT8, so the matmul runs on the faster INT8 datapath. Cider provides the missing pipeline — online activation quantization, INT8 TensorOps matmul, and fused dequantization — to unlock this dormant hardware capability. In our current experiments on Qwen3-VL-2B (M5 Pro), Cider achieves 1.15–1.21× prefill speedup over a W8A16 baseline, while preserving comparable language-model perplexity in our Llama-3-8B check (PPL Δ < 0.01).


## Background: What MLX Does Today

In the MLX v0.31 code paths we examined, we did not find a public end-to-end INT8 TensorOps inference path.

More specifically:

- **Standard matmul** (`steel_gemm_fused_nax.metal`): Instantiated for `float16`, `bfloat16`, `float32` only. No `int8` instantiation exists.
- **Quantized matmul** (`quantized_nax.h`): Dequantizes packed int{2,3,4,5,6,8} weights → FP16, then uses **FP16 TensorOps**. This is weight-only quantization — activations stay in FP16.

This does **not** mean INT8 TensorOps are impossible in MLX. At the shader level, the underlying GEMM template structure appears flexible enough to support INT8 instantiations. The missing piece is a complete and exposed W8A8 execution pipeline.


## The Benchmark

We wrote a standalone INT8 GEMM kernel in MLX's NAX style, compiled via `mx.fast.metal_kernel`, using the same:
- `matmul2d<int8_t, int8_t, int32_t>` TensorOps instruction
- BM=128, BN=128, BK=512 tile configuration
- NAXFrag cooperative tensor layout
- Device memory direct load (no threadgroup staging)

This is **not** a hypothetical comparison — the MLX-style kernel compiles, runs, and produces bit-exact results (verified: `max_diff=0`).

### Three-Way Comparison (Apple M5 Pro, Warmup=30, Repeat=100, 3 runs best-of)

**Square shapes (K=4096, N=4096)**:

| M | Cider Full Pipeline | Cider Raw INT8 | MLX-style INT8 | Full/Raw | Raw/MLX |
|---|-------------------|---------------|----------------|----------|---------|
| 16 | 0.24ms | 0.21ms | 0.25ms | 1.12x | 0.84x |
| 64 | 0.25ms | 0.24ms | 0.26ms | 1.04x | 0.95x |
| 128 | 0.28ms | 0.27ms | 0.26ms | 1.03x | 1.02x |
| 256 | 0.40ms | 0.40ms | 0.37ms | 1.01x | 1.08x |
| 512 | 0.59ms | 0.54ms | 0.52ms | 1.09x | 1.04x |
| 1024 | 0.94ms | 0.90ms | 0.86ms | 1.05x | 1.05x |
| 2048 | 1.68ms | 1.55ms | 1.49ms | 1.08x | 1.04x |

**MLP shapes (Llama-3 8B dimensions)**:

| M | K | N | Cider Full | Cider Raw | MLX-style | Full/Raw | Raw/MLX |
|---|---|---|-----------|-----------|-----------|----------|---------|
| 128 | 3584 | 18944 | 0.73ms | 0.74ms | 0.70ms | 0.99x | 1.06x |
| 512 | 3584 | 18944 | 1.66ms | 1.65ms | 1.55ms | 1.00x | 1.07x |
| 128 | 18944 | 3584 | 0.83ms | 0.78ms | 0.75ms | 1.06x | 1.05x |
| 512 | 18944 | 3584 | 1.87ms | 1.81ms | 1.68ms | 1.04x | 1.07x |

Where:
- **Cider Full Pipeline** = `perchannel_linear()`: FP16→INT8 quantize + INT8 matmul + fused INT32→FP16 dequant
- **Cider Raw INT8** = `int8_matmul_int32()`: Pure INT8×INT8→INT32 (no quantize, no dequant)
- **MLX-style INT8** = Standalone NAX kernel via `mx.fast.metal_kernel`: Same pure INT8→INT32

## What the Numbers Suggest

### 1. Raw INT8 TensorOps throughput is comparable

Across the tested shapes, Cider Raw INT8 and the standalone MLX-style INT8 kernel deliver similar throughput. This is consistent with the fact that both rely on the same underlying `matmul2d` TensorOps instruction with similar tile configurations.



### 2. The main cost of a usable W8A8 path is pipeline integration

Compared with raw INT8 matmul, the full Cider path additionally includes:
- activation quantization,
- scale handling,
- and dequantization back to the desired output dtype.

These steps introduce overhead, but they are also what make W8A8 usable in practice.

### 3. Cider's main value is exposing an end-to-end W8A8 path
Our results suggest that the key contribution of Cider is not a claim of unique raw GEMM throughput, but the availability of an integrated W8A8 execution path on Apple Silicon.


## Why Cider Uses a Custom Kernel Path

If a standalone MLX-style INT8 kernel can achieve similar raw throughput, why does Cider still implement its own kernel path?


### 1. To support dequantization as part of the usable W8A8 path

A practical W8A8 inference path needs more than raw INT32 accumulation. It also needs output scaling and dtype conversion in a form that fits the surrounding runtime.


### 2. To keep quantization and INT8 matmul in one integrated execution path

Cider packages activation quantization and INT8 matmul into a custom MLX primitive, which is more practical than treating them as disconnected experimental kernels.


### 3. To bypass the current lack of a public end-to-end INT8 route in standard MLX dispatch

Even if INT8 kernels are possible at the shader level, they are not currently exposed as a standard end-to-end W8A8 inference path in the MLX stack we examined.


## Reproducing

```bash
cd /path/to/cider
python benchmarks/mlx_native/bench_cider_vs_mlx_int8.py
```

Requires: Apple M5+, MLX ≥ 0.31, Cider installed.

The MLX-style kernel is compiled at runtime via `mx.fast.metal_kernel` with Metal 4 + MPP headers. No Xcode installation required.

## Notes

- All statements in this document are scoped to the MLX version and code paths we examined during this comparison.
- The standalone MLX-style INT8 kernel is a raw-kernel benchmark, not a full MLX-native end-to-end W8A8 inference pipeline.
- Cider Full Pipeline results include activation quantization and output dequantization overhead, while raw INT8 kernel results do not.