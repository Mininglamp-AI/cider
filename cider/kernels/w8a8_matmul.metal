// ============================================================
// W8A8 INT8×INT8→INT32 TensorOps GEMM
// Target: Apple M5 (G17G), Metal 4
//
// Weight layout: B is [N, K] (row-major), transpose_b=true
// Computes: C[M,N] = A[M,K] × B[N,K]^T
//
// Variants:
//   - fused dequant: INT8×INT8→FP16, with per-token/per-channel scales
//   - raw INT32: INT8×INT8→INT32, no scale (pure integer GEMM)
// Multi-config: large (BM=128) and small (BM=32) tiles
// Swizzle dispatch for L2 cache locality
//
// matmul2d(16,32,16) via MPP cooperative_tensor
// ============================================================

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

// ── NAXFrag layout constants ────────────────────────────────────
constant constexpr short kElemsPerFrag = 8;
constant constexpr short kElemCols = 4;
constant constexpr short kElemRowsJump = 8;

// ── NAXFrag coordinate mapping ──────────────────────────────────
inline short2 nax_get_coord(ushort lid) {
  short qid = short(lid >> 2);
  short fm = ((qid & 4) | ((short(lid) >> 1) & 3));
  short fn = ((qid & 2) | (short(lid) & 1)) * 4;
  return short2{fn, fm};
}

// ── Fragment load: device → register ────────────────────────────
template <typename T>
inline void nax_frag_load(thread T *dst, const device T *src, int ld, short2 sc,
                          short off_m = 0, short off_n = 0) {
  src += (sc.y + off_m) * ld + (sc.x + off_n);
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      dst[i * kElemCols + j] = src[(i * kElemRowsJump) * ld + j];
    }
  }
}

// ── Fragment store: raw INT32 (no dequant) ───────────────────
inline void nax_frag_store_int32(const thread int32_t *src, device int32_t *dst,
                                 int ld, short2 sc, short off_m, short off_n,
                                 uint M, uint N, uint m_base, uint n_base) {
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] =
            src[i * kElemCols + j];
      }
    }
  }
}

// ── Fragment store with bounds check and dequant ────────────────
inline void nax_frag_store_dequant(const thread int32_t *src, device half *dst,
                                   int ld, short2 sc, short off_m, short off_n,
                                   uint M, uint N, uint m_base, uint n_base,
                                   const device float *scale_a,
                                   const device float *scale_w) {
  for (short i = 0; i < 2; i++) {
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
}

// ── Generic GEMM kernel (B is [N, K], transpose_b) ─────────────
// Computes: C[M,N] = A[M,K] × B[N,K]^T
// A is [M, K] row-major, B is [N, K] row-major
// B fragments loaded as [N_tile, K_tile] and hardware transposes via transpose_b=true
template <int BM, int BN, int BK, int SK, int WM, int WN>
void w8a8_gemm_impl(const device int8_t *A, const device int8_t *B,
                    device half *C, uint M, uint N, uint K,
                    const device float *scale_a, const device float *scale_w,
                    uint swizzle_log, uint tiles_m, uint tiles_n, uint2 tgid,
                    uint sgid, uint lid) {
  constexpr int SM = BM / WM;   // 32
  constexpr int SN = BN / WN;   // 32
  constexpr short TM = SM / 16; // 2
  constexpr short TN = SN / 16; // 2
  constexpr short TK = SK / 16; // 2

  // Swizzle decode
  uint tid_y = (tgid.y << swizzle_log) + (tgid.x & ((1u << swizzle_log) - 1u));
  uint tid_x = tgid.x >> swizzle_log;

  if (tid_x >= tiles_n || tid_y >= tiles_m) {
    return;
  }

  short2 sc = nax_get_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

  // A: [M, K] row-major — same as before
  const device int8_t *sg_A = A + m_base * K;
  // B: [N, K] row-major — pointer to start of n_base-th row
  const device int8_t *sg_B = B + n_base * K;

  // transpose_b=true: right operand is [N_frag, K_frag], hardware transposes
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      gemm_op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_c =
      gemm_op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b),
                                                 int32_t>();

  int32_t c_frags[TM * TN][kElemsPerFrag];
  for (int f = 0; f < TM * TN; f++) {
    for (int i = 0; i < kElemsPerFrag; i++) {
      c_frags[f][i] = 0;
    }
  }

  // ── Main K loop ─────────────────────────────────────────────
  int gemm_k_iters = int(K) / BK;

  for (int kk0 = 0; kk0 < gemm_k_iters; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);

    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TN][TK][kElemsPerFrag];  // [N_tile, K_tile] for transpose_b
      volatile int compiler_barrier;

      // Load A fragments: [M_tile, K_tile], ld=K
      for (short mm = 0; mm < TM; mm++) {
        for (short kk = 0; kk < TK; kk++) {
          nax_frag_load(a_frags[mm][kk], sg_A + kk1, int(K), sc, short(mm * 16),
                        short(kk * 16));
        }
      }
      // Load B fragments: [N_tile, K_tile], ld=K (B is [N, K] row-major)
      for (short nn = 0; nn < TN; nn++) {
        for (short kk = 0; kk < TK; kk++) {
          nax_frag_load(b_frags[nn][kk], sg_B + kk1, int(K), sc, short(nn * 16),
                        short(kk * 16));
        }
      }
      // Compute: ct_a=[M_frag, K_frag], ct_b=[N_frag, K_frag] (transpose_b)
      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_a[i] = a_frags[mm][kk][i];
            }
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[nn][kk][i];
              ct_b[kElemsPerFrag + i] = b_frags[nn + 1][kk][i];
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
    sg_B += BK;  // B is [N, K]: K advances by BK along columns
  }

  // ── Remainder K ─────────────────────────────────────────────
  int rem_k = int(K) - gemm_k_iters * BK;
  for (int kk1 = 0; kk1 < rem_k; kk1 += 16) {
    int8_t a_frag[TM][kElemsPerFrag];
    int8_t b_frag[TN][kElemsPerFrag];
    short psk = short(max(0, rem_k - kk1));

    // Load A remainder: same as before
    for (short mm = 0; mm < TM; mm++) {
      const device int8_t *ptr = sg_A + kk1 + (sc.y + mm * 16) * K + sc.x;
      for (short i = 0; i < 2; i++) {
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          a_frag[mm][i * kElemCols + j] =
              (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : int8_t(0);
        }
      }
    }

    // Load B remainder: B is [N, K], reading [N_tile, K_rem]
    for (short nn = 0; nn < TN; nn++) {
      const device int8_t *ptr = sg_B + kk1 + (sc.y + nn * 16) * K + sc.x;
      for (short i = 0; i < 2; i++) {
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          b_frag[nn][i * kElemCols + j] =
              (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : int8_t(0);
        }
      }
    }

    for (short mm = 0; mm < TM; mm++) {
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_a[i] = a_frag[mm][i];
      }
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_b[i] = b_frag[0][i];
        ct_b[kElemsPerFrag + i] = b_frag[1][i];
      }
      short c0 = mm * TN, c1 = c0 + 1;
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

  // ── Store with fused dequant ────────────────────────────────
  device half *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++) {
    for (short nn = 0; nn < TN; nn++) {
      nax_frag_store_dequant(c_frags[mm * TN + nn], D, int(N), sc,
                             short(mm * 16), short(nn * 16), M, N, m_base,
                             n_base, scale_a, scale_w);
    }
  }
}

// ============================================================
// Kernel entry points — fused dequant
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
  w8a8_gemm_impl<128, 128, 512, 32, 4, 4>(A, B, C, M, N, K, scale_a, scale_w,
                                          swizzle_log, tiles_m, tiles_n, tgid,
                                          sgid, lid);
}

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
  w8a8_gemm_impl<32, 128, 512, 32, 1, 4>(A, B, C, M, N, K, scale_a, scale_w,
                                         swizzle_log, tiles_m, tiles_n, tgid,
                                         sgid, lid);
}

// ============================================================
// Raw INT32 GEMM impl (B is [N, K], transpose_b=true)
// ============================================================

template <int BM, int BN, int BK, int SK, int WM, int WN>
void w8a8_gemm_int32_impl(const device int8_t *A, const device int8_t *B,
                          device int32_t *C, uint M, uint N, uint K,
                          uint swizzle_log, uint tiles_m, uint tiles_n,
                          uint2 tgid, uint sgid, uint lid) {
  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  uint tid_y = (tgid.y << swizzle_log) + (tgid.x & ((1u << swizzle_log) - 1u));
  uint tid_x = tgid.x >> swizzle_log;
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short2 sc = nax_get_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

  const device int8_t *sg_A = A + m_base * K;
  const device int8_t *sg_B = B + n_base * K;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
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
      int8_t b_frags[TN][TK][kElemsPerFrag];
      volatile int compiler_barrier;

      for (short mm = 0; mm < TM; mm++)
        for (short kk = 0; kk < TK; kk++)
          nax_frag_load(a_frags[mm][kk], sg_A + kk1, int(K), sc, short(mm*16), short(kk*16));

      for (short nn = 0; nn < TN; nn++)
        for (short kk = 0; kk < TK; kk++)
          nax_frag_load(b_frags[nn][kk], sg_B + kk1, int(K), sc, short(nn*16), short(kk*16));

      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = a_frags[mm][kk][i];
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[nn][kk][i];
              ct_b[kElemsPerFrag + i] = b_frags[nn+1][kk][i];
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
    sg_B += BK;
  }

  // Remainder K
  int rem_k = int(K) - gemm_k_iters * BK;
  for (int kk1 = 0; kk1 < rem_k; kk1 += 16) {
    int8_t a_frag[TM][kElemsPerFrag];
    int8_t b_frag[TN][kElemsPerFrag];
    short psk = short(max(0, rem_k - kk1));

    for (short mm = 0; mm < TM; mm++) {
      const device int8_t *ptr = sg_A + kk1 + (sc.y + mm * 16) * K + sc.x;
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          a_frag[mm][i * kElemCols + j] = (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : int8_t(0);
        }
    }
    for (short nn = 0; nn < TN; nn++) {
      const device int8_t *ptr = sg_B + kk1 + (sc.y + nn * 16) * K + sc.x;
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          b_frag[nn][i * kElemCols + j] = (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : int8_t(0);
        }
    }
    for (short mm = 0; mm < TM; mm++) {
      for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = a_frag[mm][i];
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_b[i] = b_frag[0][i];
        ct_b[kElemsPerFrag + i] = b_frag[1][i];
      }
      short c0 = mm * TN, c1 = c0 + 1;
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

  // Store raw INT32
  device int32_t *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_int32(c_frags[mm * TN + nn], D, int(N), sc, short(mm*16), short(nn*16), M, N, m_base, n_base);
}

// ============================================================
// Kernel entry points — raw INT32 output
// ============================================================

kernel void int8_matmul_int32(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  w8a8_gemm_int32_impl<128, 128, 512, 32, 4, 4>(
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
  w8a8_gemm_int32_impl<32, 128, 512, 32, 1, 4>(
      A, B, C, M, N, K, swizzle_log, tiles_m, tiles_n, tgid, sgid, lid);
}
