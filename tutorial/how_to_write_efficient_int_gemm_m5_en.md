# How to Write an Efficient INT8 GEMM Kernel on Apple M5

> A step-by-step optimization guide: from naive threadgroup staging to 2.78x speedup using Metal 4 TensorOps
>
> All performance numbers in this tutorial are **real measurements** on Apple M5 Pro, bit-exact verified.

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Step 1: First INT8 TensorOps Kernel — Naive Threadgroup Staging](#2-step-1-first-int8-tensorops-kernel--naive-threadgroup-staging)
3. [Step 2: Eliminate Threadgroup Memory — Direct Device Reads](#3-step-2-eliminate-threadgroup-memory--direct-device-reads)
4. [Step 3: Multi-Tile Dispatch — Small Batch Occupancy Fix](#4-step-3-multi-tile-dispatch--small-batch-occupancy-fix)
5. [Step 4: Deep K-Loop (BK=512) — Amortize Loop Overhead](#5-step-4-deep-k-loop-bk512--amortize-loop-overhead)
6. [Step 5: Swizzle Dispatch — L2 Cache Locality](#6-step-5-swizzle-dispatch--l2-cache-locality)
7. [Key Technique: NAXFrag Register Layout](#7-key-technique-naxfrag-register-layout)
8. [Pitfalls & Lessons Learned](#8-pitfalls--lessons-learned)
9. [Final Results](#9-final-results)

---

## 1. Background & Motivation

Apple's M5 chip introduced **Metal 4** with **TensorOps** (`MetalPerformancePrimitives`). This provides a hardware `matmul2d` instruction operating on `cooperative_tensor` types — analogous to NVIDIA's Tensor Cores.

The key operation:

```metal
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>(ct_a, ct_b, ct_c);
// INT8 × INT8 → INT32, 16×32×16 tile
```

**Why INT8?** Modern LLM inference quantizes both weights and activations to INT8 (W8A8), halving memory bandwidth vs FP16 while the TensorOps hardware executes INT8 matmul at higher throughput.

**The problem:** MLX's GEMM kernel (`nax.h`) dequantizes quantized weights back to FP16 before computing — it never uses INT8×INT8→INT32 TensorOps. We write one from scratch.

### Hardware Specs (Apple M5 Pro)

| Feature | Value |
|---------|-------|
| GPU Cores | 20 |
| TensorOps tile | 16×32×16 (M×K×N) |
| Supported types | INT8×INT8→INT32, FP16×FP16→FP32, BF16×BF16→FP32 |
| SIMD width | 32 threads/simdgroup |

### Benchmark Setup

All benchmarks use **INT8×INT8→INT32 matmul** (pure integer, no dequant), verified **bit-exact** against NumPy int64 reference. Median of 50 runs after 5 warmup iterations.

Primary benchmark shape: **(128, 4096, 4096)** — representative of LLM inference prefill.

---

## 2. Step 1: First INT8 TensorOps Kernel — Naive Threadgroup Staging

Our first kernel follows the classic GPU GEMM pattern: load tiles from device memory to threadgroup (shared) memory, then from threadgroup to registers for TensorOps.

**Config:** BM=128, BN=128, BK=128, SK=32, WM=4, WN=4 (512 threads)

```metal
// Cooperative load: device → threadgroup
threadgroup int8_t tg_A[128 * 128];  // BM × BK
threadgroup int8_t tg_B[128 * 128];  // BK × BN

for (uint idx = tid_in_tg; idx < BM*BK; idx += NUM_THREADS) {
    tg_A[row * BK + col] = A[global_row * K + global_col];
}
// ... same for B ...
threadgroup_barrier(mem_flags::mem_threadgroup);

// Load from threadgroup → registers
nax_frag_load_tg(a_frags, &tg_A[...], BK, sc, ...);
nax_frag_load_tg(b_frags, &tg_B[...], BN, sc, ...);

// TensorOps
gemm_op.run(ct_a, ct_b, ct_c);
```

### Step 1 Performance (Real M5 Pro Data)

| Shape (M,K,N) | Step 1 (ms) |
|----------------|------------|
| (128, 4096, 4096) | **1.002** |
| (256, 4096, 4096) | **1.179** |
| (128, 3584, 18944) | **1.817** |
| (256, 3584, 18944) | **3.120** |

This is our baseline. Now we optimize.

---

## 3. Step 2: Eliminate Threadgroup Memory — Direct Device Reads

**Key insight:** Apple Silicon uses **unified memory**. Unlike discrete GPUs where device→shared is a separate memory hierarchy, on Apple GPU the "device" and "threadgroup" address spaces share the same physical memory. The threadgroup staging adds latency without benefit.

```metal
// BEFORE (Step 1): device → threadgroup → register
nax_frag_load_tg(dst, &tg_A[offset], BK, sc, ...);

// AFTER (Step 2): device → register directly
template <typename T>
inline void nax_frag_load(thread T *dst, const device T *src,
                          int ld, short2 sc,
                          short off_m = 0, short off_n = 0) {
    src += (sc.y + off_m) * ld + (sc.x + off_n);
    for (short i = 0; i < 2; i++)
        for (short j = 0; j < kElemCols; j++)
            dst[i * kElemCols + j] = src[(i * kElemRowsJump) * ld + j];
}
```

We eliminate the `threadgroup` buffers, the cooperative load loop, and the `threadgroup_barrier`. Each simdgroup loads its own data directly from device memory.

### Step 2 Performance

| Shape (M,K,N) | Step 1 (ms) | Step 2 (ms) | Speedup |
|----------------|------------|------------|---------|
| (128, 4096, 4096) | 1.002 | **0.438** | **2.29x** |
| (256, 4096, 4096) | 1.179 | **0.429** | **2.75x** |
| (128, 3584, 18944) | 1.817 | **0.815** | **2.23x** |
| (256, 3584, 18944) | 3.120 | **1.125** | **2.77x** |

**The single biggest optimization: 2.3-2.8x speedup.** Removing threadgroup staging is critical on unified memory architectures.

---

## 4. Step 3: Multi-Tile Dispatch — Small Batch Occupancy Fix

Step 1 and 2 use a single tile config: BM=128, BN=128, 512 threads. When M is small (e.g., M=16), the large tile wastes compute — most simdgroups process padding zeros.

**Solution:** Add a small-tile kernel variant:

| Config | BM | BN | Simdgroups | Threads | Best for |
|--------|----|----|-----------|---------|----------|
| Large | 128 | 128 | 16 (4×4) | 512 | M > 64 |
| Small | 32 | 128 | 4 (1×4) | 128 | M ≤ 64 |

```metal
// Same template, different instantiation
template <int BM, int BN, int BK, int SK, int WM, int WN>
void gemm_int32_impl(...) {
    constexpr int SM = BM / WM;  // Sub-tile per simdgroup
    constexpr int SN = BN / WN;
    // ...
}

// Large: BM=128, WM=4, WN=4 → SM=32, SN=32
kernel void int8_matmul_int32(...) {
    gemm_int32_impl<128, 128, 128, 32, 4, 4>(...);
}

// Small: BM=32, WM=1, WN=4 → SM=32, SN=32
kernel void int8_matmul_int32_small(...) {
    gemm_int32_impl<32, 128, 128, 32, 1, 4>(...);
}
```

Host-side selection:

```cpp
bool use_small = (M <= 64);
```

### Step 3 Performance

**Large tile (M≥128):**

| Shape (M,K,N) | Step 2 (ms) | Step 3 (ms) | Speedup |
|----------------|------------|------------|---------|
| (128, 4096, 4096) | 0.438 | **0.408** | **1.07x** |
| (256, 4096, 4096) | 0.429 | **0.410** | **1.05x** |
| (128, 3584, 18944) | 0.815 | **0.797** | **1.02x** |
| (256, 3584, 18944) | 1.125 | **1.130** | 1.00x |

**Small tile (M<128, Step 3 vs Step 2 not applicable — Step 2 has no small tile):**

| Shape (M,K,N) | Step 3 (ms) |
|----------------|------------|
| (1, 4096, 4096) | **0.239** |
| (16, 4096, 4096) | **0.241** |
| (32, 4096, 4096) | **0.240** |
| (64, 4096, 4096) | **0.247** |

For large shapes the gain is marginal (~5%), but **small-tile support is essential** — without it, M<128 would require padding to 128 rows, wasting compute and memory.

---

## 5. Step 4: Deep K-Loop (BK=512) — Amortize Loop Overhead

Increasing BK from 128 to 512 means each outer K-loop iteration processes 4x more data, reducing loop overhead and barrier frequency:

```
BK=128: K=4096 → 32 outer iterations, 32 barriers
BK=512: K=4096 →  8 outer iterations,  8 barriers (4x fewer)
```

The inner loop processes BK/SK = 512/32 = 16 sub-iterations per outer step:

```metal
constexpr int BK = 512;  // was 128

for (int kk0 = 0; kk0 < K/BK; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
        // 16 sub-iterations of load + compute
        int8_t a_frags[TM][TK][kElemsPerFrag];
        int8_t b_frags[TK][TN][kElemsPerFrag];
        volatile int compiler_barrier;  // prevent register spilling

        // Load fragments from device
        for (short mm = 0; mm < TM; mm++)
            for (short kk = 0; kk < TK; kk++)
                nax_frag_load(a_frags[mm][kk], sg_A + kk1, K, sc, ...);

        // TensorOps compute
        for (...) gemm_op.run(ct_a, ct_b, ct_c);
        (void)compiler_barrier;
    }
    sg_A += BK;
    sg_B += BK * N;
}
```

**Why `volatile int compiler_barrier`?** The Metal compiler may hoist all loads above all computes within the inner loop, causing register spill to device memory. The volatile variable forces a scheduling boundary.

### Step 4 Performance

| Shape (M,K,N) | Step 3 (ms) | Step 4 (ms) | Speedup |
|----------------|------------|------------|---------|
| (128, 4096, 4096) | 0.408 | **0.363** | **1.12x** |
| (256, 4096, 4096) | 0.410 | **0.399** | **1.03x** |
| (128, 3584, 18944) | 0.797 | **0.781** | **1.02x** |
| (256, 3584, 18944) | 1.130 | **1.082** | **1.04x** |

The (128,4096,4096) shape benefits most — BK=512 amortizes the 4096-element K dimension more efficiently.

---

## 6. Step 5: Swizzle Dispatch — L2 Cache Locality

Default threadgroup indexing assigns adjacent threadgroups to adjacent tile columns. When N is large, this causes L2 cache thrashing:

```
Default mapping:
  TG(0,0)→tile(0,0)  TG(1,0)→tile(0,1)  TG(2,0)→tile(0,2)  ...
  → All TGs in a row touch different B columns → B evicted from L2 cache
```

**Swizzle** interleaves the mapping so nearby TGs share B tile columns:

```metal
inline void swizzle_decode(uint2 tgid, uint swizzle_log, uint tiles_n,
                           thread uint &tid_y, thread uint &tid_x) {
    uint tile = 1u << swizzle_log;
    tid_y = tgid.y * tile + (tgid.x % tile);
    tid_x = tgid.x / tile;
}
```

Host computes `swizzle_log`:

```cpp
int swizzle_log = 0;
for (int s = 1; s <= 4; s++) {
    if ((tiles_n / (1 << s)) >= 1) swizzle_log = s;
}
```

### Step 5 Performance (= Production Kernel)

| Shape (M,K,N) | Step 4 (ms) | Step 5 (ms) | Speedup |
|----------------|------------|------------|---------|
| (128, 4096, 4096) | 0.363 | **0.361** | **1.01x** |
| (256, 4096, 4096) | 0.399 | **0.397** | **1.01x** |
| (128, 3584, 18944) | 0.781 | **0.794** | 0.98x |
| (256, 3584, 18944) | 1.082 | **1.086** | 1.00x |

At these benchmark sizes, swizzle has minimal effect (~1%). The benefit grows with larger N and more GPU cores competing for L2. On shapes with N>18944 or on M5 Ultra (40 cores), swizzle would show larger gains.

---

## 7. Key Technique: NAXFrag Register Layout

Understanding the `cooperative_tensor` fragment layout is essential for writing correct TensorOps kernels.

The `matmul2d(16,32,16)` instruction operates on a 16×16 logical tile, distributed across 32 threads (one simdgroup). Each thread holds **8 elements** arranged as 2 rows × 4 columns:

```
Thread layout in a 16×16 tile:
  - 32 threads, each contributes 8 elements
  - Elements span rows [fm, fm+8] and columns [fn..fn+3]

short2 nax_get_coord(ushort lid) {
    short qid = short(lid >> 2);       // quad group (0-7)
    short fm = ((qid & 4) | ((short(lid) >> 1) & 3));  // fragment row
    short fn = ((qid & 2) | (short(lid) & 1)) * 4;     // fragment col
    return short2{fn, fm};  // (col, row)
}
```

**Fragment load** reads 8 scattered elements from a contiguous matrix:

```metal
// Each thread reads 2 rows (stride 8) × 4 columns (contiguous)
template <typename T>
inline void nax_frag_load(thread T *dst, const device T *src,
                          int ld, short2 sc, short off_m, short off_n) {
    src += (sc.y + off_m) * ld + (sc.x + off_n);
    for (short i = 0; i < 2; i++)
        for (short j = 0; j < 4; j++)
            dst[i * 4 + j] = src[(i * 8) * ld + j];
}
```

**The `matmul2d` instruction then consumes two fragments:**
- **ct_a**: 8 elements of A (left input)
- **ct_b**: 16 elements of B (right input, spans 2 logical N tiles)
- **ct_c**: 16 elements of C (destination, matches ct_b layout)

```metal
// The instruction signature:
// matmul2d(16, 32, 16): A[16×16] × B[16×32] → C[16×32]
// ct_a: 8 elements (one 16×16 fragment)
// ct_b: 16 elements (two 16×16 fragments packed)
// ct_c: 16 elements (two 16×16 fragments packed)

for (short i = 0; i < 8; i++) ct_a[i] = a_frags[mm][kk][i];
for (short i = 0; i < 8; i++) {
    ct_b[i]     = b_frags[kk][nn][i];
    ct_b[8 + i] = b_frags[kk][nn+1][i];
}
gemm_op.run(ct_a, ct_b, ct_c);
```

---

## 8. Pitfalls & Lessons Learned

### Pitfall 1: Threadgroup Memory on Unified Memory

The most common mistake when porting GPU kernels to Apple Silicon. On discrete GPUs (NVIDIA, AMD), threadgroup/shared memory is physically separate SRAM — staging data there is essential. On Apple Silicon, threadgroup memory is carved from the same unified RAM, so staging adds latency without benefit.

**Rule:** On Apple Silicon, prefer direct device reads unless you need inter-thread data sharing within a threadgroup.

### Pitfall 2: Pipeline Cache Staleness

MLX's `PipelineCache` uses directory path equality to decide whether to recompile:

```cpp
if (kernel_dir_ == kernel_dir) return;  // Stale!
```

If you modify `.metal` source without changing the directory path, you get stale pipelines — correct compilation but wrong runtime results.

**Workaround:** Copy kernels to a fresh directory for each variant.

### Pitfall 3: Compiler Register Spilling

Without the `volatile int compiler_barrier`, the Metal compiler may hoist all fragment loads above all TensorOps calls, creating huge register pressure and spilling to device memory. The volatile acts as a compiler scheduling fence.

### Pitfall 4: `mx.fast.metal_kernel` Incompatibility

MLX's `mx.fast.metal_kernel` API auto-generates a wrapper function around your kernel body. This wrapper is incompatible with TensorOps `cooperative_tensor` operations, which require **all simdgroup threads to participate uniformly**. The auto-generated wrapper may insert non-uniform control flow that breaks cooperative semantics.

**Solution:** Use the C++ primitive API (`mx::Primitive`) with `PipelineCache` for TensorOps kernels.

---

## 9. Final Results

### Cumulative Optimization Summary

All measurements on Apple M5 Pro, shape **(128, 4096, 4096)**, INT8×INT8→INT32, bit-exact verified.

| Step | Technique | Time (ms) | vs Previous | vs Step 1 |
|------|-----------|-----------|-------------|-----------|
| 1 | Naive TG staging | 1.002 | — | 1.00x |
| 2 | Direct device read | 0.438 | **2.29x** | **2.29x** |
| 3 | Multi-tile dispatch | 0.408 | 1.07x | 2.46x |
| 4 | Deep K-loop (BK=512) | 0.363 | 1.12x | 2.76x |
| 5 | Swizzle dispatch | 0.361 | 1.01x | **2.78x** |

### Cross-Shape Performance (Step 5 = Production)

| Shape (M,K,N) | Step 1 (ms) | Step 5 (ms) | Total Speedup |
|----------------|------------|------------|---------------|
| (128, 4096, 4096) | 1.002 | 0.361 | **2.78x** |
| (256, 4096, 4096) | 1.179 | 0.397 | **2.97x** |
| (128, 3584, 18944) | 1.817 | 0.794 | **2.29x** |
| (256, 3584, 18944) | 3.120 | 1.086 | **2.87x** |

### Small Tile Performance (Step 3+)

| Shape (M,K,N) | Step 3 (ms) | Step 5 (ms) |
|----------------|------------|------------|
| (1, 4096, 4096) | 0.239 | 0.214 |
| (16, 4096, 4096) | 0.241 | 0.237 |
| (32, 4096, 4096) | 0.240 | 0.234 |
| (64, 4096, 4096) | 0.247 | 0.250 |

### Key Takeaways

1. **Eliminating threadgroup staging** is the single biggest win (2.3x) on unified memory.
2. **Multi-tile dispatch** is essential for small-M occupancy, marginal for large M.
3. **BK depth** matters — BK=512 vs BK=128 gives ~12% from amortizing loop overhead.
4. **Swizzle** has minimal effect at moderate N but grows with larger grids.
5. **Every number in this tutorial is real** — all kernel variants are in `dev/step_kernels/` and independently verifiable.

---

## Appendix: Reproducing These Results

All step kernel variants are available in the `dev/step_kernels/` directory:

```
dev/step_kernels/
├── step1/w8a8_matmul.metal  # Naive TG staging
├── step2/w8a8_matmul.metal  # Direct device read
├── step3/w8a8_matmul.metal  # Multi-tile (BM=128 + BM=32)
├── step4/w8a8_matmul.metal  # Deep K-loop (BK=512)
├── step5/w8a8_matmul.metal  # Swizzle (= production)
└── bench_final.py           # Benchmark script
```

Run: `python dev/step_kernels/bench_final.py`

## License

MIT. See [Cider] for the complete SDK.
