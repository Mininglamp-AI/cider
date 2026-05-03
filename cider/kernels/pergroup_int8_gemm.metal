// ============================================================
// Per-group INT8 TensorOps GEMM — symmetric quantization (bias=0)
// Target: Apple M5 (G17G), Metal 4
//
// Weight layout: B is [N, K] int8 (per-group symmetric quantized)
//   - scales_w: [num_groups, N] float32 (TRANSPOSED for coalesced access)
//   - scales_a: [M] float32 (per-token activation scales)
//
// Computes: C[m,n] = scale_a[m] * Sigma_g { float(dot_int32[m,n,g]) * scale_w[g,n] }
//   where dot_int32[m,n,g] = Sigma_{k in group g} A_int8[m,k] * B_int8[n,k]
//
// Supported group_size: 64, 128, 256
// V2: scale_w transposed [num_groups, N] for coalesced SIMD access
// ============================================================

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

// -- NAXFrag layout constants
constant constexpr short kEPF = 8;
constant constexpr short kEC = 4;
constant constexpr short kERJ = 8;

// -- NAXFrag coordinate mapping
inline short2 nax_coord(ushort lid) {
  short qid = short(lid >> 2);
  short fm = ((qid & 4) | ((short(lid) >> 1) & 3));
  short fn = ((qid & 2) | (short(lid) & 1)) * 4;
  return short2{fn, fm};
}

// -- Fragment load: device -> register
template <typename T>
inline void frag_load(thread T *dst, const device T *src, int ld, short2 sc,
                      short off_m = 0, short off_n = 0) {
  src += (sc.y + off_m) * ld + (sc.x + off_n);
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kEC; j++) {
      dst[i * kEC + j] = src[(i * kERJ) * ld + j];
    }
  }
}

// -- Per-group GEMM implementation (scale_w transposed: [num_groups, N])
template <int BM, int BN, int BK, int SK, int WM, int WN>
void pergroup_gemm_impl(const device int8_t *A, const device int8_t *B,
                        device half *C, uint M, uint N, uint K,
                        const device float *scale_a,
                        const device float *scale_w, // [num_groups, N] transposed
                        const device half *bias,     // [N] half
                        uint swizzle_log, uint tiles_m, uint tiles_n,
                        uint2 tgid, uint sgid, uint lid) {
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

  short2 sc = nax_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

  const device int8_t *sg_A = A + m_base * K;
  const device int8_t *sg_B = B + n_base * K;

  uint num_groups = K / BK;

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

  // Float accumulator (across all groups)
  float acc[TM * TN][kEPF];
  for (int f = 0; f < TM * TN; f++) {
    for (int i = 0; i < kEPF; i++) {
      acc[f][i] = 0.0f;
    }
  }

  // -- Main K loop: one iteration per group
  for (uint g = 0; g < num_groups; g++) {
    // INT32 accumulator for this group
    int32_t c_frags[TM * TN][kEPF];
    for (int f = 0; f < TM * TN; f++) {
      for (int i = 0; i < kEPF; i++) {
        c_frags[f][i] = 0;
      }
    }

    // Inner loop within group
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kEPF];
      int8_t b_frags[TN][TK][kEPF];

      for (short mm = 0; mm < TM; mm++) {
        for (short kk = 0; kk < TK; kk++) {
          frag_load(a_frags[mm][kk], sg_A + kk1, int(K), sc, short(mm * 16),
                    short(kk * 16));
        }
      }

      for (short nn = 0; nn < TN; nn++) {
        for (short kk = 0; kk < TK; kk++) {
          frag_load(b_frags[nn][kk], sg_B + kk1, int(K), sc, short(nn * 16),
                    short(kk * 16));
        }
      }

      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kEPF; i++) {
              ct_a[i] = a_frags[mm][kk][i];
            }
            for (short i = 0; i < kEPF; i++) {
              ct_b[i] = b_frags[nn][kk][i];
              ct_b[kEPF + i] = b_frags[nn + 1][kk][i];
            }
            short c0 = mm * TN + nn, c1 = c0 + 1;
            for (short i = 0; i < kEPF; i++) {
              ct_c[i] = c_frags[c0][i];
              ct_c[kEPF + i] = c_frags[c1][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kEPF; i++) {
              c_frags[c0][i] = ct_c[i];
              c_frags[c1][i] = ct_c[kEPF + i];
            }
          }
        }
      }
    }

    // -- Flush: int32 * scale_w[g, n] -> accumulate
    // scale_w is [num_groups, N]: scale_w[g * N + n_idx] is coalesced for adjacent n
    for (short mm = 0; mm < TM; mm++) {
      for (short nn = 0; nn < TN; nn++) {
        short fidx = mm * TN + nn;
        float sw[kEC];
        for (short j = 0; j < kEC; j++) {
          uint n_idx = n_base + uint(sc.x) + uint(nn * 16) + uint(j);
          sw[j] = (n_idx < N) ? scale_w[g * N + n_idx] : 0.0f;
        }
        for (short i = 0; i < 2; i++) {
          for (short j = 0; j < kEC; j++) {
            acc[fidx][i * kEC + j] += float(c_frags[fidx][i * kEC + j]) * sw[j];
          }
        }
      }
    }

    sg_A += BK;
    sg_B += BK;
  }

  // -- Store: acc * scale_a + bias -> half
  device half *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++) {
    for (short nn = 0; nn < TN; nn++) {
      short fidx = mm * TN + nn;
      for (short i = 0; i < 2; i++) {
        for (short j = 0; j < kEC; j++) {
          uint mi = m_base + uint(sc.y) + uint(mm * 16 + i * kERJ);
          uint ni = n_base + uint(sc.x) + uint(nn * 16 + j);
          if (mi < M && ni < N) {
            float val = acc[fidx][i * kEC + j] * scale_a[mi] + float(bias[ni]);
            D[(sc.y + mm * 16 + i * kERJ) * int(N) + (sc.x + nn * 16 + j)] =
                half(val);
          }
        }
      }
    }
  }
}

// ============================================================
// Kernel entry points
// ============================================================

#define GEMM_ENTRY(SUFFIX, BM_V, BN_V, BK_V, WM_V, WN_V)                       \
  kernel void pergroup_int8_gemm_##SUFFIX(                                     \
      const device int8_t *A [[buffer(0)]],                                    \
      const device int8_t *B [[buffer(1)]], device half *C [[buffer(2)]],      \
      constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]],          \
      constant uint &K [[buffer(5)]],                                          \
      const device float *scale_a [[buffer(6)]],                               \
      const device float *scale_w [[buffer(7)]],                               \
      constant uint &swizzle_log [[buffer(8)]],                                \
      constant uint &tiles_m [[buffer(9)]],                                    \
      constant uint &tiles_n [[buffer(10)]],                                   \
      const device half *bias [[buffer(11)]],                                  \
      uint2 tgid [[threadgroup_position_in_grid]],                             \
      uint sgid [[simdgroup_index_in_threadgroup]],                            \
      uint lid [[thread_index_in_simdgroup]]) {                                \
    pergroup_gemm_impl<BM_V, BN_V, BK_V, 32, WM_V, WN_V>(                      \
        A, B, C, M, N, K, scale_a, scale_w, bias, swizzle_log, tiles_m,        \
        tiles_n, tgid, sgid, lid);                                             \
  }

GEMM_ENTRY(g64, 128, 128, 64, 4, 4)
GEMM_ENTRY(g64_small, 32, 128, 64, 1, 4)
GEMM_ENTRY(g128, 128, 128, 128, 4, 4)
GEMM_ENTRY(g128_small, 32, 128, 128, 1, 4)
GEMM_ENTRY(g256, 128, 128, 256, 4, 4)
GEMM_ENTRY(g256_small, 32, 128, 256, 1, 4)
