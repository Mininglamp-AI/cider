// int8_mv_v4.metal — Plan A optimized: vectorized loads + tuning
//
// Key optimizations vs v2:
// 1. Vectorized weight loads: read 4 uint8 at once via uint32
// 2. NUM_SIMDGROUPS=4 (vs 2 in v2) for better occupancy
// 3. VPT=16 with vec4 loads (4 bytes per load instruction)
// 4. Unrolled inner loop
//
// No K-tiling (atomic overhead > L2 gain for MV bandwidth-bound regime).
// Instead, rely on vectorized loads to reduce load instruction count.

#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int NUM_SIMDGROUPS = 4;
constant constexpr int RESULTS_PER_SG = 4;
constant constexpr int GROUP_SIZE = 64;
constant constexpr int VPT = 16;  // Process 16 elements per thread per iteration
constant constexpr int BLOCK_K = VPT * SIMD_SIZE;  // 512

template <typename T>
METAL_FUNC void plan_a_v4_impl(
    const device int8_t *w [[buffer(0)]],       // [N, K] int8 (really uint8)
    const device float *scales [[buffer(1)]],    // [N, num_groups]
    const device float *biases [[buffer(2)]],    // [N, num_groups]
    const device T *x [[buffer(3)]],             // [B, K]
    device T *y [[buffer(4)]],                   // [B, N]
    const constant int &K [[buffer(5)]],
    const constant int &N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  typedef float U;

  const int out_row = tid.y * (NUM_SIMDGROUPS * RESULTS_PER_SG)
                    + simd_gid * RESULTS_PER_SG;
  if (out_row >= N) return;

  const int num_groups = K / GROUP_SIZE;
  const int batch_idx = tid.x;

  // Pointers — each thread handles VPT=16 consecutive elements
  const device uint8_t *wp_base = (const device uint8_t *)(w + out_row * K);
  const device T *xp = x + batch_idx * K + simd_lid * VPT;
  device T *yp = y + batch_idx * N + out_row;

  // scale_step: how many threads fit in one group
  // group_size=64, VPT=16 → 4 threads per group
  // scale index for this thread within a block: simd_lid / (GROUP_SIZE/VPT) = simd_lid / 4
  const int threads_per_group = GROUP_SIZE / VPT;  // 4

  U result[RESULTS_PER_SG] = {0, 0, 0, 0};

  for (int k = 0; k < K; k += BLOCK_K) {
    // Load x values — VPT=16
    U x_vals[VPT];
    U x_sum = 0;
    for (int i = 0; i < VPT; i++) {
      x_vals[i] = static_cast<U>(xp[i]);
      x_sum += x_vals[i];
    }

    // Group index for this thread's chunk
    int group_idx = (k + simd_lid * VPT) / GROUP_SIZE;

    for (int row = 0; row < RESULTS_PER_SG; row++) {
      if (out_row + row >= N) break;
      const device uint8_t *wl = wp_base + (row * K) + k + simd_lid * VPT;
      U s = scales[(out_row + row) * num_groups + group_idx];
      U b = biases[(out_row + row) * num_groups + group_idx];

      // Vectorized dot product using uint32 loads
      U dot = 0;
      const device uint32_t *wl4 = (const device uint32_t *)wl;
      for (int i = 0; i < VPT / 4; i++) {
        uint32_t packed = wl4[i];
        dot += x_vals[i*4 + 0] * U(packed & 0xFF);
        dot += x_vals[i*4 + 1] * U((packed >> 8) & 0xFF);
        dot += x_vals[i*4 + 2] * U((packed >> 16) & 0xFF);
        dot += x_vals[i*4 + 3] * U((packed >> 24) & 0xFF);
      }
      result[row] += s * dot + b * x_sum;
    }

    xp += BLOCK_K;
  }

  for (int row = 0; row < RESULTS_PER_SG; row++) {
    U total = simd_sum(result[row]);
    if (simd_lid == 0 && (out_row + row) < N) {
      yp[row] = static_cast<T>(total);
    }
  }
}

// ── Entry points ─────────────────────────────────────────────

[[kernel]] void int8_mv_v4_f16(
    const device int8_t *w [[buffer(0)]],
    const device float *scales [[buffer(1)]],
    const device float *biases [[buffer(2)]],
    const device half *x [[buffer(3)]],
    device half *y [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  plan_a_v4_impl<half>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

[[kernel]] void int8_mv_v4_bf16(
    const device int8_t *w [[buffer(0)]],
    const device float *scales [[buffer(1)]],
    const device float *biases [[buffer(2)]],
    const device bfloat *x [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  plan_a_v4_impl<bfloat>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}
