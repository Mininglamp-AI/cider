# 如何在 Apple M5 芯片上编写高效的 INT8 GEMM Kernel

> 从朴素实现到 2.78 倍加速的逐步优化指南 —— Metal 4 TensorOps 实战
>
> 本文所有性能数据均为 **Apple M5 Pro 真机实测**，bit-exact 精度验证通过。

## 目录

1. [背景与动机](#1-背景与动机)
2. [Step 1：朴素 INT8 TensorOps Kernel —— Threadgroup 暂存](#2-step-1朴素-int8-tensorops-kernel--threadgroup-暂存)
3. [Step 2：消除 Threadgroup —— 直读 Device Memory](#3-step-2消除-threadgroup--直读-device-memory)
4. [Step 3：多 Tile 调度 —— 小 Batch 占用率优化](#4-step-3多-tile-调度--小-batch-占用率优化)
5. [Step 4：深 K 循环 (BK=512) —— 摊薄循环开销](#5-step-4深-k-循环-bk512--摊薄循环开销)
6. [Step 5：Swizzle 调度 —— L2 Cache 局部性](#6-step-5swizzle-调度--l2-cache-局部性)
7. [核心技术：NAXFrag 寄存器布局](#7-核心技术naxfrag-寄存器布局)
8. [踩坑与教训](#8-踩坑与教训)
9. [最终结果](#9-最终结果)

---

## 1. 背景与动机

Apple M5 芯片引入了 **Metal 4** 的 **TensorOps** (`MetalPerformancePrimitives`)，提供硬件级 `matmul2d` 指令，操作 `cooperative_tensor` 类型——类似 NVIDIA 的 Tensor Cores。

核心操作：

```metal
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>(ct_a, ct_b, ct_c);
// INT8 × INT8 → INT32, 16×32×16 tile
```

**为什么用 INT8？** LLM 推理中，W8A8 方案将权重和激活同时量化为 INT8，内存带宽减半，且硬件 INT8 matmul 吞吐量高于 FP16。

**问题：** MLX 的 GEMM kernel (`nax.h`) 会先将量化权重反量化为 FP16 再计算——从未使用 INT8×INT8→INT32 TensorOps 路径。我们需要从头写一个。

### 硬件参数 (Apple M5 Pro)

| 特性 | 值 |
|------|-----|
| GPU 核心数 | 20 |
| TensorOps Tile | 16×32×16 (M×K×N) |
| 支持类型 | INT8×INT8→INT32, FP16×FP16→FP32, BF16×BF16→FP32 |
| SIMD 宽度 | 32 线程/simdgroup |

### Benchmark 配置

所有 benchmark 使用 **INT8×INT8→INT32**（纯整数 matmul），与 NumPy int64 参考值 **bit-exact** 验证。每个 shape 跑 50 次取中位数，5 次 warmup。

主要基准 shape：**(128, 4096, 4096)**——代表 LLM 推理 prefill 场景。

---

## 2. Step 1：朴素 INT8 TensorOps Kernel —— Threadgroup 暂存

第一版 kernel 采用经典 GPU GEMM 模式：先将数据从 device memory 加载到 threadgroup (shared) memory，再从 threadgroup 读到寄存器执行 TensorOps。

**配置：** BM=128, BN=128, BK=128, SK=32, WM=4, WN=4 (512 线程)

```metal
// 协作加载：device → threadgroup
threadgroup int8_t tg_A[128 * 128];  // BM × BK
threadgroup int8_t tg_B[128 * 128];  // BK × BN

for (uint idx = tid_in_tg; idx < BM*BK; idx += NUM_THREADS) {
    tg_A[row * BK + col] = A[global_row * K + global_col];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// 从 threadgroup 读到寄存器
nax_frag_load_tg(a_frags, &tg_A[...], BK, sc, ...);
nax_frag_load_tg(b_frags, &tg_B[...], BN, sc, ...);

// 执行 TensorOps
gemm_op.run(ct_a, ct_b, ct_c);
```

### Step 1 实测数据

| Shape (M,K,N) | Step 1 (ms) |
|----------------|------------|
| (128, 4096, 4096) | **1.002** |
| (256, 4096, 4096) | **1.179** |
| (128, 3584, 18944) | **1.817** |
| (256, 3584, 18944) | **3.120** |

这是我们的起点。

---

## 3. Step 2：消除 Threadgroup —— 直读 Device Memory

**核心发现：** Apple Silicon 使用**统一内存**架构。跟独立 GPU (NVIDIA, AMD) 的 shared memory 是物理独立 SRAM 不同，Apple GPU 的 threadgroup memory 从同一块统一内存划分。Threadgroup 暂存只是增加了延迟，没有任何收益。

```metal
// 之前 (Step 1): device → threadgroup → register
nax_frag_load_tg(dst, &tg_A[offset], BK, sc, ...);

// 之后 (Step 2): device → register 直读
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

去掉 `threadgroup` buffer、协作加载循环和 `threadgroup_barrier`。每个 simdgroup 独立从 device memory 读数据。

### Step 2 实测数据

| Shape (M,K,N) | Step 1 (ms) | Step 2 (ms) | 加速比 |
|----------------|------------|------------|--------|
| (128, 4096, 4096) | 1.002 | **0.438** | **2.29x** |
| (256, 4096, 4096) | 1.179 | **0.429** | **2.75x** |
| (128, 3584, 18944) | 1.817 | **0.815** | **2.23x** |
| (256, 3584, 18944) | 3.120 | **1.125** | **2.77x** |

**全篇最大优化：2.3-2.8 倍加速。** 统一内存架构下去掉 threadgroup 暂存是第一要务。

---

## 4. Step 3：多 Tile 调度 —— 小 Batch 占用率优化

Step 1/2 只有一种 tile 配置：BM=128, 512 线程。当 M 很小时（如 M=16），大 tile 浪费算力——大部分 simdgroup 在算 padding 的零。

**方案：** 增加小 tile kernel 变体：

| 配置 | BM | BN | Simdgroups | 线程数 | 适用 |
|------|----|----|-----------|--------|------|
| 大 Tile | 128 | 128 | 16 (4×4) | 512 | M > 64 |
| 小 Tile | 32 | 128 | 4 (1×4) | 128 | M ≤ 64 |

Host 侧选择：

```cpp
bool use_small = (M <= 64);
```

### Step 3 实测数据

**大 tile (M≥128)：**

| Shape (M,K,N) | Step 2 (ms) | Step 3 (ms) | 加速比 |
|----------------|------------|------------|--------|
| (128, 4096, 4096) | 0.438 | **0.408** | **1.07x** |
| (256, 4096, 4096) | 0.429 | **0.410** | **1.05x** |
| (128, 3584, 18944) | 0.815 | **0.797** | **1.02x** |
| (256, 3584, 18944) | 1.125 | **1.130** | 1.00x |

**小 tile (M<128，Step 3 新增能力)：**

| Shape (M,K,N) | Step 3 (ms) |
|----------------|------------|
| (1, 4096, 4096) | **0.239** |
| (16, 4096, 4096) | **0.241** |
| (32, 4096, 4096) | **0.240** |
| (64, 4096, 4096) | **0.247** |

大 shape 上增益不大 (~5%)，但**小 tile 支持是刚需**——没有它，M<128 只能 pad 到 128 行，浪费算力和内存。

---

## 5. Step 4：深 K 循环 (BK=512) —— 摊薄循环开销

BK 从 128 增加到 512，外层 K 循环迭代次数减少 4 倍：

```
BK=128: K=4096 → 32 次外层迭代, 32 次 barrier
BK=512: K=4096 →  8 次外层迭代,  8 次 barrier (少 4 倍)
```

```metal
constexpr int BK = 512;  // 从 128 改为 512

for (int kk0 = 0; kk0 < K/BK; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
        // 16 次子迭代的 load + compute
        int8_t a_frags[TM][TK][kElemsPerFrag];
        int8_t b_frags[TK][TN][kElemsPerFrag];
        volatile int compiler_barrier;  // 防止寄存器溢出

        // 加载 + 计算
        ...
        (void)compiler_barrier;
    }
}
```

**`volatile int compiler_barrier` 的作用：** Metal 编译器可能将所有 load 提升到所有 compute 之前，导致寄存器爆掉溢出到 device memory。volatile 变量强制编译器插入调度边界。

### Step 4 实测数据

| Shape (M,K,N) | Step 3 (ms) | Step 4 (ms) | 加速比 |
|----------------|------------|------------|--------|
| (128, 4096, 4096) | 0.408 | **0.363** | **1.12x** |
| (256, 4096, 4096) | 0.410 | **0.399** | **1.03x** |
| (128, 3584, 18944) | 0.797 | **0.781** | **1.02x** |
| (256, 3584, 18944) | 1.130 | **1.082** | **1.04x** |

(128,4096,4096) 获益最大——BK=512 更高效地处理 4096 的 K 维度。

---

## 6. Step 5：Swizzle 调度 —— L2 Cache 局部性

默认 threadgroup 索引将相邻 TG 分配到相邻的 tile 列。当 N 很大时，会造成 L2 cache 颠簸：

```
默认映射:
  TG(0,0)→tile(0,0)  TG(1,0)→tile(0,1)  TG(2,0)→tile(0,2) ...
  → 同一行的 TG 访问不同的 B 列 → B 被反复从 DRAM 重新加载
```

**Swizzle** 重映射 TG 索引，使相邻 TG 共享 B 的 tile 列：

```metal
inline void swizzle_decode(uint2 tgid, uint swizzle_log, uint tiles_n,
                           thread uint &tid_y, thread uint &tid_x) {
    uint tile = 1u << swizzle_log;
    tid_y = tgid.y * tile + (tgid.x % tile);
    tid_x = tgid.x / tile;
}
```

### Step 5 实测数据（= 生产 kernel）

| Shape (M,K,N) | Step 4 (ms) | Step 5 (ms) | 加速比 |
|----------------|------------|------------|--------|
| (128, 4096, 4096) | 0.363 | **0.361** | **1.01x** |
| (256, 4096, 4096) | 0.399 | **0.397** | **1.01x** |
| (128, 3584, 18944) | 0.781 | **0.794** | 0.98x |
| (256, 3584, 18944) | 1.082 | **1.086** | 1.00x |

在当前 benchmark 规模下 swizzle 效果不显著 (~1%)。在更大的 N 或更多 GPU 核心（如 M5 Ultra 40 核）下增益会更明显。

---

## 7. 核心技术：NAXFrag 寄存器布局

理解 `cooperative_tensor` 的 fragment 布局是写正确 TensorOps kernel 的关键。

`matmul2d(16,32,16)` 指令操作 16×16 逻辑 tile，由 32 个线程（一个 simdgroup）分布持有。每个线程持有 **8 个元素**，排列为 2 行 × 4 列：

```metal
// 32 线程 → 16×16 tile
// 每个线程：2 行（间距 8）× 4 列（连续）
short2 nax_get_coord(ushort lid) {
    short qid = short(lid >> 2);       // quad group (0-7)
    short fm = ((qid & 4) | ((short(lid) >> 1) & 3));  // 行
    short fn = ((qid & 2) | (short(lid) & 1)) * 4;     // 列
    return short2{fn, fm};  // (col, row)
}
```

**`matmul2d` 的操作数：**
- **ct_a**: 8 个元素（一个 16×16 fragment）
- **ct_b**: 16 个元素（两个 16×16 fragment 拼接，覆盖 16×32）
- **ct_c**: 16 个元素（与 ct_b 同布局）

```metal
for (short i = 0; i < 8; i++) ct_a[i] = a_frags[mm][kk][i];
for (short i = 0; i < 8; i++) {
    ct_b[i]     = b_frags[kk][nn][i];
    ct_b[8 + i] = b_frags[kk][nn+1][i];
}
gemm_op.run(ct_a, ct_b, ct_c);
```

---

## 8. 踩坑与教训

### 坑 1：统一内存上的 Threadgroup 暂存

从独立 GPU 移植 kernel 时最常犯的错误。NVIDIA/AMD 的 shared memory 是物理独立的 SRAM，暂存到那里是必要的。Apple Silicon 的 threadgroup memory 跟 device memory 共享物理内存，暂存反而增加延迟。

**结论：** Apple Silicon 上，除非需要线程间数据共享，否则优先直读 device memory。

### 坑 2：Pipeline Cache 未感知源码变更

MLX 的 `PipelineCache` 用目录路径字符串判断是否需要重编译：

```cpp
if (kernel_dir_ == kernel_dir) return;  // 旧的！
```

修改 `.metal` 源码但不改目录路径，cache 返回旧 pipeline——编译正确但运行结果错误。

**绕过方法：** 每个 kernel 变体用独立目录。

### 坑 3：编译器寄存器溢出

没有 `volatile int compiler_barrier` 时，Metal 编译器可能把所有 fragment load 提到所有 TensorOps 调用之前，导致寄存器压力爆炸、溢出到 device memory。volatile 充当编译器调度屏障。

### 坑 4：`mx.fast.metal_kernel` 与 TensorOps 不兼容

MLX 的 `mx.fast.metal_kernel` API 会自动生成 wrapper 函数。这个 wrapper 与 TensorOps 的 `cooperative_tensor` 操作不兼容——`cooperative_tensor` 要求 simdgroup 内所有线程**统一参与**。

**解决方案：** TensorOps kernel 必须用 C++ primitive API (`mx::Primitive`) + `PipelineCache`。

---

## 9. 最终结果

### 累积优化汇总

Apple M5 Pro 实测，Shape **(128, 4096, 4096)**，INT8×INT8→INT32，bit-exact 验证。

| Step | 技术 | 耗时 (ms) | vs 上一步 | vs Step 1 |
|------|------|-----------|-----------|-----------|
| 1 | 朴素 TG 暂存 | 1.002 | — | 1.00x |
| 2 | 直读 device memory | 0.438 | **2.29x** | **2.29x** |
| 3 | 多 tile 调度 | 0.408 | 1.07x | 2.46x |
| 4 | 深 K 循环 (BK=512) | 0.363 | 1.12x | 2.76x |
| 5 | Swizzle 调度 | 0.361 | 1.01x | **2.78x** |

### 跨 Shape 性能 (Step 5 = 生产 kernel)

| Shape (M,K,N) | Step 1 (ms) | Step 5 (ms) | 总加速比 |
|----------------|------------|------------|----------|
| (128, 4096, 4096) | 1.002 | 0.361 | **2.78x** |
| (256, 4096, 4096) | 1.179 | 0.397 | **2.97x** |
| (128, 3584, 18944) | 1.817 | 0.794 | **2.29x** |
| (256, 3584, 18944) | 3.120 | 1.086 | **2.87x** |

### 小 Tile 性能 (Step 3+)

| Shape (M,K,N) | Step 3 (ms) | Step 5 (ms) |
|----------------|------------|------------|
| (1, 4096, 4096) | 0.239 | 0.214 |
| (16, 4096, 4096) | 0.241 | 0.237 |
| (32, 4096, 4096) | 0.240 | 0.234 |
| (64, 4096, 4096) | 0.247 | 0.250 |

### 核心结论

1. **消除 threadgroup 暂存**是最大单项优化 (2.3x)，统一内存架构的第一要务
2. **多 tile 调度**对小 M 是刚需，对大 M 增益不大
3. **BK 深度**有效——BK=512 vs BK=128 带来 ~12% 提升
4. **Swizzle** 在中等规模 N 下效果不大，大 grid 下更明显
5. **本文所有数据均为真机实测**——所有 kernel 变体在 `dev/step_kernels/` 下可独立复现

---

## 附录：复现方法

所有 step kernel 变体位于 `dev/step_kernels/` 目录：

```
dev/step_kernels/
├── step1/w8a8_matmul.metal  # 朴素 TG 暂存
├── step2/w8a8_matmul.metal  # 直读 device memory
├── step3/w8a8_matmul.metal  # 多 tile (BM=128 + BM=32)
├── step4/w8a8_matmul.metal  # 深 K 循环 (BK=512)
├── step5/w8a8_matmul.metal  # Swizzle (= 生产 kernel)
└── bench_final.py           # Benchmark 脚本
```

运行: `python dev/step_kernels/bench_final.py`

## License

MIT. 完整 SDK: [Cider]
