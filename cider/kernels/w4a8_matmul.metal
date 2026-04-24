// ============================================================
// W4A8 INT4-weight × INT8-activation → FP16 TensorOps GEMM
// Target: Apple M5, Metal 4
//
// V3: Optimized inline unpack with precomputed base pointers.
//     Key insight: for fragment's 8 elements (2 rows × 4 cols),
//     the 2 rows are at k and k+8. For packed [K/2, N]:
//     - Row 0 (k):   byte at (k/2)*N + n, use k&1 to select nibble
//     - Row 1 (k+8): byte at ((k+8)/2)*N + n, use (k+8)&1 to select nibble
//     Since k+8 has same parity as k (8 is even), both rows use same nibble.
//     => Can share nibble selection logic.
//
//     Further: read 4 consecutive bytes at once per row (cols are contiguous
//     in N dimension), then extract 4 nibbles. This maximizes memory bandwidth.
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

// ── Fragment load A from device memory ──────────────────────────
inline void frag_load_a(thread int8_t *dst, const device int8_t *src, int ld,
                        short2 sc, short off_m, short off_n) {
  src += (sc.y + off_m) * ld + (sc.x + off_n);
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      dst[i * kElemCols + j] = src[(i * kElemRowsJump) * ld + j];
    }
  }
}

// ── Fragment load B with optimized W4 unpack ────────────────────
// Fragment maps to 2 rows (k, k+8) × 4 cols (n, n+1, n+2, n+3).
// packed_w layout: [K/2, N] uint8, high nibble = even k, low nibble = odd k.
// Pre-compute row pointers and read 4 consecutive bytes per row.
inline void frag_load_b_w4(thread int8_t *dst, const device uint8_t *packed_w,
                           uint N, uint k_base, uint n_base, short2 sc) {
  uint k0 = k_base + uint(sc.y); // first row
  uint k1 = k0 + kElemRowsJump;  // second row (k+8)
  uint n = n_base + uint(sc.x);  // column start

  // Both k0 and k1 have same parity (differ by 8)
  bool use_low = (k0 & 1u);

  const device uint8_t *row0 = packed_w + (k0 >> 1) * N + n;
  const device uint8_t *row1 = packed_w + (k1 >> 1) * N + n;

  if (use_low) {
    for (short j = 0; j < kElemCols; j++) {
      dst[j] = int8_t(row0[j] & 0xF) - 8;
      dst[kElemCols + j] = int8_t(row1[j] & 0xF) - 8;
    }
  } else {
    for (short j = 0; j < kElemCols; j++) {
      dst[j] = int8_t(row0[j] >> 4) - 8;
      dst[kElemCols + j] = int8_t(row1[j] >> 4) - 8;
    }
  }
}

// ── Fragment store with fused dequant ───────────────────────────
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

// ── W4A8 GEMM V3 ───────────────────────────────────────────────
template <int BM, int BN, int BK, int SK, int WM, int WN>
void w4a8_gemm_impl(const device int8_t *A, const device uint8_t *packed_w,
                    device half *C, uint M, uint N, uint K,
                    const device float *scale_a, const device float *scale_w,
                    uint swizzle_log, uint tiles_m, uint tiles_n, uint2 tgid,
                    uint sgid, uint lid) {
  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

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

  const device int8_t *sg_A = A + m_base * K;

  constexpr auto matmul_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<matmul_desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      gemm_op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_c =
      gemm_op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b),
                                                 int32_t>();

  int32_t c_frags[TM * TN][kElemsPerFrag];
  for (int f = 0; f < TM * TN; f++) {
    for (int e = 0; e < kElemsPerFrag; e++) {
      c_frags[f][e] = 0;
    }
  }
  int gemm_k_iters = int(K) / BK;

  for (int kk0 = 0; kk0 < gemm_k_iters; kk0++) {
    uint k_offset = uint(kk0) * BK;

    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TK][TN][kElemsPerFrag];
      volatile int compiler_barrier;

      for (short mm = 0; mm < TM; mm++) {
        for (short kk = 0; kk < TK; kk++) {
          frag_load_a(a_frags[mm][kk], sg_A + k_offset + kk1, int(K), sc,
                      short(mm * 16), short(kk * 16));
        }
      }

      for (short kk = 0; kk < TK; kk++) {
        for (short nn = 0; nn < TN; nn++) {
          frag_load_b_w4(b_frags[kk][nn], packed_w, N,
                         k_offset + uint(kk1) + uint(kk * 16),
                         n_base + uint(nn * 16), sc);
        }
      }

      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_a[i] = a_frags[mm][kk][i];
            }
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
  }

  // Remainder
  int rem_k = int(K) - gemm_k_iters * BK;
  for (int kk1 = 0; kk1 < rem_k; kk1 += 16) {
    int8_t a_frag[TM][kElemsPerFrag];
    int8_t b_frag[TN][kElemsPerFrag];
    short psk = short(max(0, rem_k - kk1));
    uint k_abs = uint(gemm_k_iters * BK + kk1);

    for (short mm = 0; mm < TM; mm++) {
      const device int8_t *ptr = sg_A + k_abs + (sc.y + mm * 16) * K + sc.x;
      for (short i = 0; i < 2; i++) {
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          a_frag[mm][i * kElemCols + j] =
              (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : int8_t(0);
        }
      }
    }

    for (short nn = 0; nn < TN; nn++) {
      uint n = n_base + uint(nn * 16);
      for (short i = 0; i < 2; i++) {
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.y + i * kElemRowsJump);
          if (ki < psk) {
            uint k = k_abs + uint(ki);
            uint ni = n + uint(sc.x) + uint(j);
            uint byte_row = k >> 1;
            uint8_t packed = packed_w[byte_row * N + ni];
            uint8_t nibble = (k & 1) == 0 ? (packed >> 4) : (packed & 0xF);
            b_frag[nn][i * kElemCols + j] = int8_t(nibble) - 8;
          } else {
            b_frag[nn][i * kElemCols + j] = 0;
          }
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

  device half *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++) {
    for (short nn = 0; nn < TN; nn++) {
      nax_frag_store_dequant(c_frags[mm * TN + nn], D, int(N), sc,
                             short(mm * 16), short(nn * 16), M, N, m_base,
                             n_base, scale_a, scale_w);
    }
  }
}

kernel void w4a8_matmul_fused_dequant(
    const device int8_t *A [[buffer(0)]],
    const device uint8_t *packed_w [[buffer(1)]], device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]], const device float *scale_a [[buffer(6)]],
    const device float *scale_w [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]], constant uint &tiles_n [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  w4a8_gemm_impl<128, 128, 512, 32, 4, 4>(A, packed_w, C, M, N, K, scale_a,
                                          scale_w, swizzle_log, tiles_m,
                                          tiles_n, tgid, sgid, lid);
}

kernel void w4a8_matmul_fused_dequant_small(
    const device int8_t *A [[buffer(0)]],
    const device uint8_t *packed_w [[buffer(1)]], device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]], const device float *scale_a [[buffer(6)]],
    const device float *scale_w [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]], constant uint &tiles_n [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  w4a8_gemm_impl<32, 128, 512, 32, 1, 4>(A, packed_w, C, M, N, K, scale_a,
                                         scale_w, swizzle_log, tiles_m, tiles_n,
                                         tgid, sgid, lid);
}
