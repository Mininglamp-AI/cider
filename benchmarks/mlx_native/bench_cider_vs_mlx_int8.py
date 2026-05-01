#!/usr/bin/env python3
"""
Benchmark: Cider W8A8 INT8 matmul vs MLX-style NAX INT8 matmul.

Both use the same mpp::tensor_ops::matmul2d<int8,int8,int32> hardware instruction.
The difference is in data loading/storing patterns.

We write a standalone INT8 GEMM kernel in MLX Steel NAX style,
compiled via mx.fast.metal_kernel, and compare against cider's kernel.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
from cider import perchannel_linear, quantize_weight_int8
from cider.ops import int8_matmul_int32

# =============================================================
# MLX-style INT8 NAX GEMM kernel (standalone)
# Same tile config as cider: BM=128, BN=128, BK=512, WM=4, WN=4
# Uses matmul2d<int8,int8,int32> - identical HW instruction to cider
# =============================================================

MLX_INT8_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_simdgroup>

using namespace metal;

constant constexpr int kElemsPerFrag = 8;
constant constexpr int kElemCols = 4;
constant constexpr int kElemRowsJump = 8;

METAL_FUNC short2 nax_get_coord(ushort lid) {
    short qid = short(lid >> 2);
    short fm = ((qid & 4) | ((short(lid) >> 1) & 3));
    short fn = ((qid & 2) | (short(lid) & 1)) * 4;
    return short2(fn, fm);
}

METAL_FUNC void load_frag_int8(thread int8_t *dst,
                                const device int8_t *base,
                                int stride, short2 sc,
                                short row_off, short col_off) {
    const device int8_t *ptr = base + (sc.y + row_off) * stride + sc.x + col_off;
    for (short i = 0; i < 2; i++) {
        for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = ptr[i * kElemRowsJump * stride + j];
        }
    }
}

METAL_FUNC void store_frag_int32(const thread int32_t *src,
                                  device int32_t *base,
                                  int stride, short2 sc,
                                  short row_off, short col_off,
                                  uint M, uint N,
                                  uint m_base, uint n_base) {
    for (short i = 0; i < 2; i++) {
        uint r = m_base + row_off + sc.y + i * kElemRowsJump;
        for (short j = 0; j < kElemCols; j++) {
            uint c = n_base + col_off + sc.x + j;
            if (r < M && c < N) {
                base[r * N + c] = src[i * kElemCols + j];
            }
        }
    }
}
"""

MLX_INT8_SOURCE = r"""
    uint M_val = M[0], N_val = N[0], K_val = K[0];
    uint tiles_m_val = tilesm[0], tiles_n_val = tilesn[0];

    constexpr int BM = 128, BN = 128, BK = 512;
    constexpr int WM = 4, WN = 4;
    constexpr int SM = BM / WM;
    constexpr int SN = BN / WN;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = 32 / 16;
    constexpr short SK = 32;

    uint2 tgid = uint2(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y);
    uint sgid = simdgroup_index_in_threadgroup;
    uint lid = thread_index_in_simdgroup;

    if (tgid.x >= tiles_n_val || tgid.y >= tiles_m_val) return;

    short2 sc = nax_get_coord(ushort(lid));
    uint sg_row = sgid / WN;
    uint sg_col = sgid % WN;
    uint m_base = tgid.y * BM + sg_row * SM;
    uint n_base = tgid.x * BN + sg_col * SN;

    const device int8_t *sg_A = A + m_base * K_val;
    const device int8_t *sg_B = B + n_base;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    auto ct_a = gemm_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto ct_b = gemm_op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto ct_c = gemm_op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

    int32_t c_frags[TM * TN][kElemsPerFrag];
    for (int f = 0; f < TM * TN; f++)
        for (int i = 0; i < kElemsPerFrag; i++)
            c_frags[f][i] = 0;

    int gemm_k_iters = int(K_val) / BK;
    for (int kk0 = 0; kk0 < gemm_k_iters; kk0++) {
        threadgroup_barrier(mem_flags::mem_none);
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            int8_t a_frags[TM][TK][kElemsPerFrag];
            int8_t b_frags[TK][TN][kElemsPerFrag];
            volatile int compiler_barrier;

            for (short mm = 0; mm < TM; mm++)
                for (short kk = 0; kk < TK; kk++)
                    load_frag_int8(a_frags[mm][kk], sg_A + kk1, int(K_val), sc, short(mm*16), short(kk*16));

            for (short kk = 0; kk < TK; kk++)
                for (short nn = 0; nn < TN; nn++)
                    load_frag_int8(b_frags[kk][nn], sg_B + kk1 * N_val, int(N_val), sc, short(kk*16), short(nn*16));

            for (short mm = 0; mm < TM; mm++) {
                for (short nn = 0; nn < TN; nn += 2) {
                    for (short kk = 0; kk < TK; kk++) {
                        for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = a_frags[mm][kk][i];
                        for (short i = 0; i < kElemsPerFrag; i++) {
                            ct_b[i] = b_frags[kk][nn][i];
                            ct_b[kElemsPerFrag + i] = b_frags[kk][nn+1][i];
                        }
                        short c0 = mm*TN+nn, c1 = c0+1;
                        for (short i = 0; i < kElemsPerFrag; i++) {
                            ct_c[i] = c_frags[c0][i];
                            ct_c[kElemsPerFrag+i] = c_frags[c1][i];
                        }
                        gemm_op.run(ct_a, ct_b, ct_c);
                        for (short i = 0; i < kElemsPerFrag; i++) {
                            c_frags[c0][i] = ct_c[i];
                            c_frags[c1][i] = ct_c[kElemsPerFrag+i];
                        }
                    }
                }
            }
            (void)compiler_barrier;
        }
        sg_A += BK;
        sg_B += BK * N_val;
    }

    for (short mm = 0; mm < TM; mm++)
        for (short nn = 0; nn < TN; nn++)
            store_frag_int32(c_frags[mm*TN+nn], out, int(N_val), sc, short(mm*16), short(nn*16), M_val, N_val, m_base, n_base);
"""

WARMUP = 20
REPEAT = 50

SHAPES = [
    (16,  4096, 4096),
    (64,  4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (2048, 4096, 4096),
    # MLP shapes
    (128, 3584, 18944),
    (512, 3584, 18944),
    (128, 18944, 3584),
    (512, 18944, 3584),
]


def bench_cider_w8a8(x, w_int8, scale_w):
    """Cider full pipeline: quantize_act + int8_matmul + fused_dequant -> FP16"""
    for _ in range(WARMUP):
        y = perchannel_linear(x, w_int8, scale_w)
        mx.eval(y)
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        y = perchannel_linear(x, w_int8, scale_w)
        mx.eval(y)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def bench_cider_int8_raw(A_int8, B_int8):
    """Cider raw INT8 matmul: int8 x int8 -> int32 (no quantize, no dequant)"""
    for _ in range(WARMUP):
        y = int8_matmul_int32(A_int8, B_int8)
        mx.eval(y)
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        y = int8_matmul_int32(A_int8, B_int8)
        mx.eval(y)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def bench_mlx_int8(A_int8, B_int8, M, N, K):
    """MLX-style standalone INT8 matmul: int8 x int8 -> int32 (raw GEMM only)"""
    BM, BN = 128, 128
    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN

    M_buf = mx.array([M], dtype=mx.uint32)
    N_buf = mx.array([N], dtype=mx.uint32)
    K_buf = mx.array([K], dtype=mx.uint32)
    tm_buf = mx.array([tiles_m], dtype=mx.uint32)
    tn_buf = mx.array([tiles_n], dtype=mx.uint32)

    kernel = mx.fast.metal_kernel(
        name='mlx_style_int8_gemm',
        input_names=['A', 'B', 'M', 'N', 'K', 'tilesm', 'tilesn'],
        output_names=['out'],
        source=MLX_INT8_SOURCE,
        header=MLX_INT8_HEADER,
    )

    # grid = total threads, NOT threadgroup count
    # threadgroup=(512,1,1), so grid_x = tiles_n * 512, grid_y = tiles_m * 1
    grid_x = tiles_n * 512
    grid_y = tiles_m

    for _ in range(WARMUP):
        result = kernel(
            inputs=[A_int8, B_int8, M_buf, N_buf, K_buf, tm_buf, tn_buf],
            output_shapes=[(M, N)],
            output_dtypes=[mx.int32],
            grid=(grid_x, grid_y, 1),
            threadgroup=(512, 1, 1),
        )
        mx.eval(result[0])

    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        result = kernel(
            inputs=[A_int8, B_int8, M_buf, N_buf, K_buf, tm_buf, tn_buf],
            output_shapes=[(M, N)],
            output_dtypes=[mx.int32],
            grid=(grid_x, grid_y, 1),
            threadgroup=(512, 1, 1),
        )
        mx.eval(result[0])
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def verify_correctness(K, N):
    """Quick correctness check: MLX-style int8 matmul vs numpy."""
    M = 16
    np.random.seed(123)
    a_np = np.random.randint(-5, 5, (M, K), dtype=np.int8)
    b_np = np.random.randint(-5, 5, (K, N), dtype=np.int8)
    ref = a_np.astype(np.int32) @ b_np.astype(np.int32)

    A_int8 = mx.array(a_np)
    B_int8 = mx.array(b_np)

    BM, BN = 128, 128
    tiles_m = (M + BM - 1) // BM
    tiles_n = (N + BN - 1) // BN

    kernel = mx.fast.metal_kernel(
        name='mlx_style_int8_gemm',
        input_names=['A', 'B', 'M', 'N', 'K', 'tilesm', 'tilesn'],
        output_names=['out'],
        source=MLX_INT8_SOURCE,
        header=MLX_INT8_HEADER,
    )

    result = kernel(
        inputs=[A_int8, B_int8,
                mx.array([M], dtype=mx.uint32),
                mx.array([N], dtype=mx.uint32),
                mx.array([K], dtype=mx.uint32),
                mx.array([tiles_m], dtype=mx.uint32),
                mx.array([tiles_n], dtype=mx.uint32)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.int32],
        grid=(tiles_n * 512, tiles_m, 1),
        threadgroup=(512, 1, 1),
    )
    mx.eval(result[0])
    got = np.array(result[0])

    max_diff = np.max(np.abs(ref - got))
    match = np.array_equal(ref, got)
    print(f"Correctness check ({M}x{K} @ {K}x{N}): max_diff={max_diff}, exact_match={match}")
    if not match:
        print(f"  ref[0,:8] = {ref[0,:8]}")
        print(f"  got[0,:8] = {got[0,:8]}")
    return match


def main():
    print("=" * 70)
    print("Cider W8A8 (full pipeline) vs MLX-style INT8 GEMM (raw matmul)")
    print("=" * 70)
    print(f"Warmup={WARMUP}, Repeat={REPEAT}")
    print()
    print("NOTE: Cider includes quantize_act + INT8 matmul + fused dequant")
    print("      MLX-style is raw INT8 matmul only (no quantize/dequant)")
    print("      So MLX-style SHOULD be faster (does less work)")
    print()

    # Verify correctness first
    print("--- Correctness Check ---")
    ok = verify_correctness(4096, 4096)
    if not ok:
        print("WARNING: correctness check failed! Results may be invalid.")
    print()

    print(f"{'M':>5s} {'K':>6s} {'N':>6s} | {'Cider-full':>10s} | {'Cider-raw':>10s} | {'MLX-INT8':>9s} | {'full/raw':>8s} | {'raw/MLX':>7s}")
    print("-" * 80)

    prev_kn = None
    for M, K, N in SHAPES:
        if (K, N) != prev_kn:
            np.random.seed(42)
            w_fp32 = np.random.randn(K, N).astype(np.float32)
            w_int8_np, scale_w_np = quantize_weight_int8(w_fp32)
            w_int8 = mx.array(w_int8_np)
            scale_w = mx.array(scale_w_np)
            B_int8 = mx.array(w_int8_np)
            mx.eval(w_int8, scale_w, B_int8)
            prev_kn = (K, N)

        x_fp16 = mx.random.normal((M, K)).astype(mx.float16)
        A_int8 = mx.array(np.random.randint(-127, 127, (M, K), dtype=np.int8))
        mx.eval(x_fp16, A_int8)

        t_full = bench_cider_w8a8(x_fp16, w_int8, scale_w)
        t_raw = bench_cider_int8_raw(A_int8, B_int8)
        t_mlx = bench_mlx_int8(A_int8, B_int8, M, N, K)

        r_full_raw = t_full / t_raw if t_raw > 0 else 0
        r_raw_mlx = t_raw / t_mlx if t_mlx > 0 else 0
        print(f"{M:>5d} {K:>6d} {N:>6d} | {t_full:>8.2f}ms | {t_raw:>8.2f}ms | {t_mlx:>7.2f}ms | {r_full_raw:>6.2f}x | {r_raw_mlx:>5.2f}x")


if __name__ == "__main__":
    main()
