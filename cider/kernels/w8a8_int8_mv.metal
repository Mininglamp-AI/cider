// w8a8_int8_mv.metal — Per-channel symmetric INT8 MV (optimized)
// Matches MLX qmv_fast structure: 2 SG, VPT=8, block=256
// y[n] = scale_w[n] * dot(W_int8[n,:], x[:]) + bias[n]

#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_SIZE      = 32;
constant constexpr int NUM_SIMDGROUPS = 2;
constant constexpr int RESULTS_PER_SG = 4;
constant constexpr int VPT            = 8;
constant constexpr int BLOCK_K        = VPT * SIMD_SIZE;  // 256

kernel void w8a8_int8_mv(
    const device half *x          [[buffer(0)]],
    const device int8_t *W        [[buffer(1)]],
    device half *y                [[buffer(2)]],
    const device float *scale_w   [[buffer(3)]],
    constant uint &N              [[buffer(4)]],
    constant uint &K              [[buffer(5)]],
    const device half *bias       [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid  [[thread_index_in_simdgroup]])
{
    const uint rows_per_tg = NUM_SIMDGROUPS * RESULTS_PER_SG;  // 8
    const uint out_row = tgid * rows_per_tg + sgid * RESULTS_PER_SG;

    float result[RESULTS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pointers: each thread handles VPT consecutive elements per block
    const device half *xp = x + lid * VPT;

    for (uint k = 0; k < K; k += BLOCK_K) {
        // Load x tile into registers
        float xv[VPT];
        for (int i = 0; i < VPT; i++) {
            xv[i] = float(xp[i]);
        }

        // Dot product with weight rows
        for (int r = 0; r < RESULTS_PER_SG; r++) {
            uint n_idx = out_row + r;
            if (n_idx < N) {
                const device int8_t *wp = W + n_idx * K + k + lid * VPT;
                float dot = 0.0f;
                for (int i = 0; i < VPT; i++) {
                    dot += float(wp[i]) * xv[i];
                }
                result[r] += dot;
            }
        }
        xp += BLOCK_K;
    }

    // Reduce across simdgroup
    for (int r = 0; r < RESULTS_PER_SG; r++) {
        result[r] = simd_sum(result[r]);
    }

    // Write back: y = scale * dot + bias
    if (lid == 0) {
        for (int r = 0; r < RESULTS_PER_SG; r++) {
            uint n_idx = out_row + r;
            if (n_idx < N) {
                float val = scale_w[n_idx] * result[r] + float(bias[n_idx]);
                y[n_idx] = half(val);
            }
        }
    }
}
