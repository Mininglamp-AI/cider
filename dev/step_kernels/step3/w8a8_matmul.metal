// ============================================================
// Step 3: Multi-tile Dispatch (Large BM=128 + Small BM=32)
// - BK=128, SK=32
// - Direct device read
// - Swizzle decode (required by C++ dispatch)
// - Large: BM=128, BN=128, WM=4, WN=4 (512 threads)
// - Small: BM=32, BN=128, WM=1, WN=4 (128 threads)
// ============================================================

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

constant constexpr short kElemsPerFrag = 8;
constant constexpr short kElemCols = 4;
constant constexpr short kElemRowsJump = 8;

inline short2 nax_get_coord(ushort lid) {
  short qid = short(lid >> 2);
  short fm = ((qid & 4) | ((short(lid) >> 1) & 3));
  short fn = ((qid & 2) | (short(lid) & 1)) * 4;
  return short2{fn, fm};
}

inline void swizzle_decode(uint2 tgid, uint swizzle_log, uint tiles_n,
                           thread uint &tid_y, thread uint &tid_x) {
  uint tile = 1u << swizzle_log;
  tid_y = tgid.y * tile + (tgid.x % tile);
  tid_x = tgid.x / tile;
}

template <typename T>
inline void nax_frag_load(thread T *dst, const device T *src, int ld, short2 sc,
                          short off_m = 0, short off_n = 0) {
  src += (sc.y + off_m) * ld + (sc.x + off_n);
  for (short i = 0; i < 2; i++)
    for (short j = 0; j < kElemCols; j++)
      dst[i * kElemCols + j] = src[(i * kElemRowsJump) * ld + j];
}

inline void nax_frag_store_int32(const thread int32_t *src, device int32_t *dst,
                                 int ld, short2 sc, short off_m, short off_n,
                                 uint M, uint N, uint m_base, uint n_base) {
  for (short i = 0; i < 2; i++)
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N)
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] =
            src[i * kElemCols + j];
    }
}

inline void nax_frag_store_dequant(const thread int32_t *src, device half *dst,
                                   int ld, short2 sc, short off_m, short off_n,
                                   uint M, uint N, uint m_base, uint n_base,
                                   const device float *scale_a,
                                   const device float *scale_w) {
  for (short i = 0; i < 2; i++)
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        float val = float(src[i * kElemCols + j]) * scale_a[mi] * scale_w[ni];
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] =
            half(val);
      }
    }
}

// Generic GEMM with swizzle decode
template <int BM, int BN, int BK, int SK, int WM, int WN>
void step3_gemm_int32_impl(
    const device int8_t *A, const device int8_t *B,
    device int32_t *C, uint M, uint N, uint K,
    uint swizzle_log, uint tiles_m, uint tiles_n,
    uint2 tgid, uint sgid, uint lid) {

  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  uint tid_y, tid_x;
  swizzle_decode(tgid, swizzle_log, tiles_n, tid_y, tid_x);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short2 sc = nax_get_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

  const device int8_t *sg_A = A + m_base * K;
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

  int gemm_k_iters = int(K) / BK;
  for (int kk0 = 0; kk0 < gemm_k_iters; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TK][TN][kElemsPerFrag];
      volatile int compiler_barrier;

      for (short mm = 0; mm < TM; mm++)
        for (short kk = 0; kk < TK; kk++)
          nax_frag_load(a_frags[mm][kk], sg_A + kk1, int(K), sc,
                        short(mm * 16), short(kk * 16));
      for (short kk = 0; kk < TK; kk++)
        for (short nn = 0; nn < TN; nn++)
          nax_frag_load(b_frags[kk][nn], sg_B + kk1 * N, int(N), sc,
                        short(kk * 16), short(nn * 16));

      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = a_frags[mm][kk][i];
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[kk][nn][i];
              ct_b[kElemsPerFrag + i] = b_frags[kk][nn + 1][i];
            }
            short c0 = mm * TN + nn, c1 = c0 + 1;
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_c[i] = c_frags[c0][i];
              ct_c[kElemsPerFrag + i] = c_frags[c1][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kElemsPerFrag; i++) {
              c_frags[c0][i] = ct_c[i];
              c_frags[c1][i] = ct_c[kElemsPerFrag + i];
            }
          }
        }
      }
      (void)compiler_barrier;
    }
    sg_A += BK;
    sg_B += BK * N;
  }

  // Remainder K
  int rem_k = int(K) - gemm_k_iters * BK;
  if (rem_k > 0) {
    for (int kk1 = 0; kk1 < rem_k; kk1 += SK) {
      int actual_sk = min(SK, rem_k - kk1);
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TK][TN][kElemsPerFrag];

      for (short mm = 0; mm < TM; mm++)
        for (short kk = 0; kk < TK; kk++) {
          for (short i = 0; i < 2; i++)
            for (short j = 0; j < kElemCols; j++) {
              int ki = kk * 16 + int(sc.x) + j;
              if (ki < actual_sk) {
                int row = int(sc.y) + mm * 16 + i * kElemRowsJump;
                a_frags[mm][kk][i * kElemCols + j] = sg_A[kk1 + row * int(K) + ki];
              } else {
                a_frags[mm][kk][i * kElemCols + j] = 0;
              }
            }
        }
      for (short kk = 0; kk < TK; kk++)
        for (short nn = 0; nn < TN; nn++) {
          for (short i = 0; i < 2; i++)
            for (short j = 0; j < kElemCols; j++) {
              int ki = kk * 16 + int(sc.y) + i * kElemRowsJump;
              if (ki < actual_sk) {
                int col = nn * 16 + int(sc.x) + j;
                b_frags[kk][nn][i * kElemCols + j] = sg_B[(kk1 + ki) * int(N) + col];
              } else {
                b_frags[kk][nn][i * kElemCols + j] = 0;
              }
            }
        }

      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = a_frags[mm][kk][i];
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[kk][nn][i];
              ct_b[kElemsPerFrag + i] = b_frags[kk][nn + 1][i];
            }
            short c0 = mm * TN + nn, c1 = c0 + 1;
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_c[i] = c_frags[c0][i];
              ct_c[kElemsPerFrag + i] = c_frags[c1][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kElemsPerFrag; i++) {
              c_frags[c0][i] = ct_c[i];
              c_frags[c1][i] = ct_c[kElemsPerFrag + i];
            }
          }
        }
      }
    }
  }

  device int32_t *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_int32(c_frags[mm * TN + nn], D, int(N), sc,
                           short(mm * 16), short(nn * 16), M, N, m_base, n_base);
}

// Dequant version
template <int BM, int BN, int BK, int SK, int WM, int WN>
void step3_gemm_dequant_impl(
    const device int8_t *A, const device int8_t *B,
    device half *C, uint M, uint N, uint K,
    const device float *scale_a, const device float *scale_w,
    uint swizzle_log, uint tiles_m, uint tiles_n,
    uint2 tgid, uint sgid, uint lid) {

  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  uint tid_y, tid_x;
  swizzle_decode(tgid, swizzle_log, tiles_n, tid_y, tid_x);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short2 sc = nax_get_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

  const device int8_t *sg_A = A + m_base * K;
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

  int gemm_k_iters = int(K) / BK;
  for (int kk0 = 0; kk0 < gemm_k_iters; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TK][TN][kElemsPerFrag];
      volatile int compiler_barrier;
      for (short mm = 0; mm < TM; mm++)
        for (short kk = 0; kk < TK; kk++)
          nax_frag_load(a_frags[mm][kk], sg_A + kk1, int(K), sc,
                        short(mm * 16), short(kk * 16));
      for (short kk = 0; kk < TK; kk++)
        for (short nn = 0; nn < TN; nn++)
          nax_frag_load(b_frags[kk][nn], sg_B + kk1 * N, int(N), sc,
                        short(kk * 16), short(nn * 16));
      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = a_frags[mm][kk][i];
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[kk][nn][i];
              ct_b[kElemsPerFrag + i] = b_frags[kk][nn + 1][i];
            }
            short c0 = mm * TN + nn, c1 = c0 + 1;
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_c[i] = c_frags[c0][i];
              ct_c[kElemsPerFrag + i] = c_frags[c1][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kElemsPerFrag; i++) {
              c_frags[c0][i] = ct_c[i];
              c_frags[c1][i] = ct_c[kElemsPerFrag + i];
            }
          }
        }
      }
      (void)compiler_barrier;
    }
    sg_A += BK;
    sg_B += BK * N;
  }

  int rem_k = int(K) - gemm_k_iters * BK;
  if (rem_k > 0) {
    for (int kk1 = 0; kk1 < rem_k; kk1 += SK) {
      int actual_sk = min(SK, rem_k - kk1);
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TK][TN][kElemsPerFrag];
      for (short mm = 0; mm < TM; mm++)
        for (short kk = 0; kk < TK; kk++) {
          for (short i = 0; i < 2; i++)
            for (short j = 0; j < kElemCols; j++) {
              int ki = kk * 16 + int(sc.x) + j;
              if (ki < actual_sk) {
                int row = int(sc.y) + mm * 16 + i * kElemRowsJump;
                a_frags[mm][kk][i * kElemCols + j] = sg_A[kk1 + row * int(K) + ki];
              } else {
                a_frags[mm][kk][i * kElemCols + j] = 0;
              }
            }
        }
      for (short kk = 0; kk < TK; kk++)
        for (short nn = 0; nn < TN; nn++) {
          for (short i = 0; i < 2; i++)
            for (short j = 0; j < kElemCols; j++) {
              int ki = kk * 16 + int(sc.y) + i * kElemRowsJump;
              if (ki < actual_sk) {
                int col = nn * 16 + int(sc.x) + j;
                b_frags[kk][nn][i * kElemCols + j] = sg_B[(kk1 + ki) * int(N) + col];
              } else {
                b_frags[kk][nn][i * kElemCols + j] = 0;
              }
            }
        }
      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = a_frags[mm][kk][i];
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[kk][nn][i];
              ct_b[kElemsPerFrag + i] = b_frags[kk][nn + 1][i];
            }
            short c0 = mm * TN + nn, c1 = c0 + 1;
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_c[i] = c_frags[c0][i];
              ct_c[kElemsPerFrag + i] = c_frags[c1][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kElemsPerFrag; i++) {
              c_frags[c0][i] = ct_c[i];
              c_frags[c1][i] = ct_c[kElemsPerFrag + i];
            }
          }
        }
      }
    }
  }

  device half *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_dequant(c_frags[mm * TN + nn], D, int(N), sc,
                             short(mm * 16), short(nn * 16), M, N, m_base, n_base,
                             scale_a, scale_w);
}

// ============================================================
// Entry points — Large tile
// ============================================================
kernel void w8a8_matmul_fused_dequant(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device half *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    const device float *scale_a [[buffer(6)]],
    const device float *scale_w [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]], constant uint &tiles_n [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  step3_gemm_dequant_impl<128, 128, 128, 32, 4, 4>(
      A, B, C, M, N, K, scale_a, scale_w, swizzle_log, tiles_m, tiles_n,
      tgid, sgid, lid);
}

// Small tile entry point (NEW in Step 3!)
kernel void w8a8_matmul_fused_dequant_small(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device half *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    const device float *scale_a [[buffer(6)]],
    const device float *scale_w [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]], constant uint &tiles_n [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  step3_gemm_dequant_impl<32, 128, 128, 32, 1, 4>(
      A, B, C, M, N, K, scale_a, scale_w, swizzle_log, tiles_m, tiles_n,
      tgid, sgid, lid);
}

// INT32 entry points
kernel void int8_matmul_int32(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  step3_gemm_int32_impl<128, 128, 128, 32, 4, 4>(
      A, B, C, M, N, K, swizzle_log, tiles_m, tiles_n, tgid, sgid, lid);
}

kernel void int8_matmul_int32_small(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  step3_gemm_int32_impl<32, 128, 128, 32, 1, 4>(
      A, B, C, M, N, K, swizzle_log, tiles_m, tiles_n, tgid, sgid, lid);
}
