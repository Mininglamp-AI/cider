// ============================================================
// Step 1: Naive INT8 TensorOps GEMM with Threadgroup Memory Staging
// - BM=128, BN=128, BK=128, SK=32
// - Only large tile (WM=4, WN=4, 512 threads)
// - Threadgroup memory staging: load A/B tiles to TG memory first
// - Swizzle decode included (required by C++ dispatch)
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

// Swizzle decode: convert swizzled grid coords to real tile coords
inline void swizzle_decode(uint2 tgid, uint swizzle_log, uint tiles_n,
                           thread uint &tid_y, thread uint &tid_x) {
  uint tile = 1u << swizzle_log;
  uint tgid_y_raw = tgid.y * tile + (tgid.x % tile);
  uint tgid_x_raw = tgid.x / tile;
  tid_y = tgid_y_raw;
  tid_x = tgid_x_raw;
}

template <typename T>
inline void nax_frag_load_tg(thread T *dst, const threadgroup T *src, int ld,
                              short2 sc, short off_m = 0, short off_n = 0) {
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

// Step 1: TG-staged GEMM, BM=128 only
template <int BM, int BN, int BK, int SK, int WM, int WN>
void step1_gemm_int32_impl(
    const device int8_t *A, const device int8_t *B,
    device int32_t *C, uint M, uint N, uint K,
    uint swizzle_log, uint tiles_m, uint tiles_n,
    threadgroup int8_t *tg_A,
    threadgroup int8_t *tg_B,
    uint2 tgid, uint sgid, uint lid, uint tid_in_tg) {

  constexpr int SM = BM / WM;   // 32
  constexpr int SN = BN / WN;   // 32
  constexpr short TM = SM / 16; // 2
  constexpr short TN = SN / 16; // 2
  constexpr short TK = SK / 16; // 2
  constexpr int NUM_THREADS = WM * WN * 32; // 512

  uint tid_y, tid_x;
  swizzle_decode(tgid, swizzle_log, tiles_n, tid_y, tid_x);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short2 sc = nax_get_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

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

  for (uint k_base = 0; k_base < K; k_base += BK) {
    // Cooperative load A[BM, BK] → threadgroup
    uint a_elems = BM * BK;
    for (uint idx = tid_in_tg; idx < a_elems; idx += NUM_THREADS) {
      uint row = idx / BK;
      uint col = idx % BK;
      uint gm = tid_y * BM + row;
      uint gk = k_base + col;
      tg_A[row * BK + col] = (gm < M && gk < K) ? A[gm * K + gk] : int8_t(0);
    }
    // Cooperative load B[BK, BN] → threadgroup
    uint b_elems = BK * BN;
    for (uint idx = tid_in_tg; idx < b_elems; idx += NUM_THREADS) {
      uint row = idx / BN;
      uint col = idx % BN;
      uint gk = k_base + row;
      uint gn = tid_x * BN + col;
      tg_B[row * BN + col] = (gk < K && gn < N) ? B[gk * N + gn] : int8_t(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TK][TN][kElemsPerFrag];

      for (short mm = 0; mm < TM; mm++)
        for (short kk = 0; kk < TK; kk++)
          nax_frag_load_tg(a_frags[mm][kk],
                           tg_A + (sg_row * SM) * BK + kk1,
                           BK, sc, short(mm * 16), short(kk * 16));
      for (short kk = 0; kk < TK; kk++)
        for (short nn = 0; nn < TN; nn++)
          nax_frag_load_tg(b_frags[kk][nn],
                           tg_B + kk1 * BN + (sg_col * SN),
                           BN, sc, short(kk * 16), short(nn * 16));

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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  device int32_t *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_int32(c_frags[mm * TN + nn], D, int(N), sc,
                           short(mm * 16), short(nn * 16), M, N, m_base, n_base);
}

// Dequant version
template <int BM, int BN, int BK, int SK, int WM, int WN>
void step1_gemm_dequant_impl(
    const device int8_t *A, const device int8_t *B,
    device half *C, uint M, uint N, uint K,
    const device float *scale_a, const device float *scale_w,
    uint swizzle_log, uint tiles_m, uint tiles_n,
    threadgroup int8_t *tg_A,
    threadgroup int8_t *tg_B,
    uint2 tgid, uint sgid, uint lid, uint tid_in_tg) {

  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  constexpr int NUM_THREADS = WM * WN * 32;

  uint tid_y, tid_x;
  swizzle_decode(tgid, swizzle_log, tiles_n, tid_y, tid_x);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short2 sc = nax_get_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

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

  for (uint k_base = 0; k_base < K; k_base += BK) {
    uint a_elems = BM * BK;
    for (uint idx = tid_in_tg; idx < a_elems; idx += NUM_THREADS) {
      uint row = idx / BK;
      uint col = idx % BK;
      uint gm = tid_y * BM + row;
      uint gk = k_base + col;
      tg_A[row * BK + col] = (gm < M && gk < K) ? A[gm * K + gk] : int8_t(0);
    }
    uint b_elems = BK * BN;
    for (uint idx = tid_in_tg; idx < b_elems; idx += NUM_THREADS) {
      uint row = idx / BN;
      uint col = idx % BN;
      uint gk = k_base + row;
      uint gn = tid_x * BN + col;
      tg_B[row * BN + col] = (gk < K && gn < N) ? B[gk * N + gn] : int8_t(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TK][TN][kElemsPerFrag];
      for (short mm = 0; mm < TM; mm++)
        for (short kk = 0; kk < TK; kk++)
          nax_frag_load_tg(a_frags[mm][kk],
                           tg_A + (sg_row * SM) * BK + kk1,
                           BK, sc, short(mm * 16), short(kk * 16));
      for (short kk = 0; kk < TK; kk++)
        for (short nn = 0; nn < TN; nn++)
          nax_frag_load_tg(b_frags[kk][nn],
                           tg_B + kk1 * BN + (sg_col * SN),
                           BN, sc, short(kk * 16), short(nn * 16));
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  device half *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_dequant(c_frags[mm * TN + nn], D, int(N), sc,
                             short(mm * 16), short(nn * 16), M, N, m_base, n_base,
                             scale_a, scale_w);
}

// ============================================================
// Kernel entry points
// ============================================================

kernel void w8a8_matmul_fused_dequant(
    const device int8_t *A [[buffer(0)]],
    const device int8_t *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    const device float *scale_a [[buffer(6)]],
    const device float *scale_w [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]],
    constant uint &tiles_n [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]],
    uint tid_in_tg [[thread_index_in_threadgroup]]) {
  threadgroup int8_t tg_A[128 * 128];
  threadgroup int8_t tg_B[128 * 128];
  step1_gemm_dequant_impl<128, 128, 128, 32, 4, 4>(
      A, B, C, M, N, K, scale_a, scale_w, swizzle_log, tiles_m, tiles_n,
      tg_A, tg_B, tgid, sgid, lid, tid_in_tg);
}

kernel void w8a8_matmul_fused_dequant_small(
    const device int8_t *A [[buffer(0)]],
    const device int8_t *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    const device float *scale_a [[buffer(6)]],
    const device float *scale_w [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]],
    constant uint &tiles_n [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]],
    uint tid_in_tg [[thread_index_in_threadgroup]]) {
  // Step 1 has no small tile — fallback to large tile
  threadgroup int8_t tg_A[128 * 128];
  threadgroup int8_t tg_B[128 * 128];
  step1_gemm_dequant_impl<128, 128, 128, 32, 4, 4>(
      A, B, C, M, N, K, scale_a, scale_w, swizzle_log, tiles_m, tiles_n,
      tg_A, tg_B, tgid, sgid, lid, tid_in_tg);
}

kernel void int8_matmul_int32(
    const device int8_t *A [[buffer(0)]],
    const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]],
    uint tid_in_tg [[thread_index_in_threadgroup]]) {
  threadgroup int8_t tg_A[128 * 128];
  threadgroup int8_t tg_B[128 * 128];
  step1_gemm_int32_impl<128, 128, 128, 32, 4, 4>(
      A, B, C, M, N, K, swizzle_log, tiles_m, tiles_n,
      tg_A, tg_B, tgid, sgid, lid, tid_in_tg);
}

kernel void int8_matmul_int32_small(
    const device int8_t *A [[buffer(0)]],
    const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]],
    uint tid_in_tg [[thread_index_in_threadgroup]]) {
  // Step 1 has no small tile — fallback to large tile
  threadgroup int8_t tg_A[128 * 128];
  threadgroup int8_t tg_B[128 * 128];
  step1_gemm_int32_impl<128, 128, 128, 32, 4, 4>(
      A, B, C, M, N, K, swizzle_log, tiles_m, tiles_n,
      tg_A, tg_B, tgid, sgid, lid, tid_in_tg);
}
