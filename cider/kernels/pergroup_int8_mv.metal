// ============================================================
// Per-group INT8 MV kernel V5 — symmetric-only fast path
//
// Match MLX qmv_fast: 2 SG, VPT=8, block=256
// Symmetric quantization: new_bias is always zero, skip correction.
//
// Formula:
//   y[n] = Sigma_g { scale_w[g,n] * dot_g(w[n], x) } + bias[n]
//
// V2: scale_w transposed [num_groups, N] for coalesced access
// ============================================================

#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int NUM_SIMDGROUPS = 2;
constant constexpr int RESULTS_PER_SG = 4;
constant constexpr int VPT = 8;
constant constexpr int BLOCK_K = VPT * SIMD_SIZE; // 256

template <int GROUP_SIZE>
inline void pergroup_mv_v5_impl(
    const device half *x,
    const device int8_t *W,
    device half *y,
    const device float *scale_w,    // [num_groups, N] transposed
    const device float *new_bias [[maybe_unused]],
    constant uint &N, constant uint &K,
    const device half *bias,
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid  [[thread_index_in_simdgroup]])
{
    constexpr int rows_per_tg = NUM_SIMDGROUPS * RESULTS_PER_SG; // 8
    const uint out_row = tgid * rows_per_tg + sgid * RESULTS_PER_SG;
    const uint num_groups = K / GROUP_SIZE;

    float result[RESULTS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pointers
    const device half *xp = x + lid * VPT;

    const device int8_t *wrows[RESULTS_PER_SG];
    for (int r = 0; r < RESULTS_PER_SG; r++) {
        uint n_idx = out_row + r;
        wrows[r] = (n_idx < N) ? (W + n_idx * K + lid * VPT) : W;
    }

    for (uint k = 0; k < K; k += BLOCK_K) {
        // Load x tile
        float xv[VPT];
        for (int i = 0; i < VPT; i++) {
            xv[i] = float(xp[i]);
        }

        // Group index for this thread data
        uint g = (k + lid * VPT) / GROUP_SIZE;

        for (int r = 0; r < RESULTS_PER_SG; r++) {
            uint n_idx = out_row + r;
            if (n_idx >= N) continue;

            // Load 8 int8 weights as 2x uint32
            uint32_t packed0 = *reinterpret_cast<const device uint32_t *>(wrows[r]);
            uint32_t packed1 = *reinterpret_cast<const device uint32_t *>(wrows[r] + 4);

            float b0 = float(int8_t(packed0 & 0xFF));
            float b1 = float(int8_t((packed0 >> 8) & 0xFF));
            float b2 = float(int8_t((packed0 >> 16) & 0xFF));
            float b3 = float(int8_t((packed0 >> 24) & 0xFF));
            float b4 = float(int8_t(packed1 & 0xFF));
            float b5 = float(int8_t((packed1 >> 8) & 0xFF));
            float b6 = float(int8_t((packed1 >> 16) & 0xFF));
            float b7 = float(int8_t((packed1 >> 24) & 0xFF));

            float dot = xv[0]*b0 + xv[1]*b1 + xv[2]*b2 + xv[3]*b3
                      + xv[4]*b4 + xv[5]*b5 + xv[6]*b6 + xv[7]*b7;

            // Transposed: scale_w[g * N + n_idx] (coalesced for adjacent n)
            float sw = scale_w[g * N + (out_row + r)];
            result[r] += sw * dot;
        }

        // Advance pointers
        xp += BLOCK_K;
        for (int r = 0; r < RESULTS_PER_SG; r++) {
            wrows[r] += BLOCK_K;
        }
    }

    // Reduce across simdgroup
    for (int r = 0; r < RESULTS_PER_SG; r++) {
        result[r] = simd_sum(result[r]);
    }

    // Write result
    if (lid == 0) {
        for (int r = 0; r < RESULTS_PER_SG; r++) {
            uint n_idx = out_row + r;
            if (n_idx < N) {
                y[n_idx] = half(result[r] + float(bias[n_idx]));
            }
        }
    }
}

// ============================================================
// Entry points
// ============================================================

kernel void pergroup_int8_mv_g64(
    const device half *x [[buffer(0)]],
    const device int8_t *W [[buffer(1)]],
    device half *y [[buffer(2)]],
    const device float *scale_w [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    const device half *bias [[buffer(6)]],
    const device float *new_bias [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]])
{
    pergroup_mv_v5_impl<64>(x, W, y, scale_w, new_bias, N, K, bias, tgid, sgid, lid);
}

kernel void pergroup_int8_mv_g128(
    const device half *x [[buffer(0)]],
    const device int8_t *W [[buffer(1)]],
    device half *y [[buffer(2)]],
    const device float *scale_w [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    const device half *bias [[buffer(6)]],
    const device float *new_bias [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]])
{
    pergroup_mv_v5_impl<128>(x, W, y, scale_w, new_bias, N, K, bias, tgid, sgid, lid);
}

kernel void pergroup_int8_mv_g256(
    const device half *x [[buffer(0)]],
    const device int8_t *W [[buffer(1)]],
    device half *y [[buffer(2)]],
    const device float *scale_w [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    const device half *bias [[buffer(6)]],
    const device float *new_bias [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]])
{
    pergroup_mv_v5_impl<256>(x, W, y, scale_w, new_bias, N, K, bias, tgid, sgid, lid);
}
