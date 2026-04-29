// int8_mv_v3.metal — Plan A with K-tile swizzle for large-K L2 optimization
//
// Key insight: For large K (e.g., 18944), a single threadgroup iterating all K
// has poor L2 cache utilization because each K-block access pattern is far apart.
//
// Solution: Split K into tiles (K_TILE=4096). Each grid.z handles one K-tile.
// Adjacent threadgroups in Y process adjacent N rows for the same K-tile,
// so their weight reads hit the same L2 cache lines.
//
// Partial sums from each K-tile are accumulated via atomic_add_explicit on output.
// For small K (<=K_TILE), grid.z=1 and behavior is identical to v2.

#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int NUM_SIMDGROUPS = 4;  // Increased for better occupancy
constant constexpr int RESULTS_PER_SG = 4;
constant constexpr int GROUP_SIZE = 64;
constant constexpr int VPT = 8;
constant constexpr int BLOCK_K = VPT * SIMD_SIZE;  // 256 elements per K iteration
constant constexpr int K_TILE = 4096;  // K elements per grid.z tile

// ── Plan A + K-tile: weight dequant, FP16 activation, tiled K ──

template <typename T>
METAL_FUNC void plan_a_tiled_impl(
    const device int8_t *w [[buffer(0)]],       // [N, K] int8 (really uint8)
    const device float *scales [[buffer(1)]],    // [N, num_groups]
    const device float *biases [[buffer(2)]],    // [N, num_groups]
    const device T *x [[buffer(3)]],             // [B, K]
    device float *y [[buffer(4)]],               // [B, N] float32 for atomic accumulation
    const constant int &K [[buffer(5)]],
    const constant int &N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  typedef float U;

  const int out_row = tid.y * (NUM_SIMDGROUPS * RESULTS_PER_SG)
                    + simd_gid * RESULTS_PER_SG;
  if (out_row >= N) return;

  // K-tile range for this threadgroup
  const int k_tile_start = tid.z * K_TILE;
  const int k_tile_end = min(k_tile_start + K_TILE, K);
  if (k_tile_start >= K) return;

  const int num_groups = K / GROUP_SIZE;
  const int batch_idx = tid.x;

  // Starting group index for this K-tile
  const int group_start = k_tile_start / GROUP_SIZE;

  // Weight pointer starts at k_tile_start offset
  const device uint8_t *wp = (const device uint8_t *)(w + out_row * K)
                           + k_tile_start + simd_lid * VPT;
  // Scale/bias pointer starts at the group corresponding to k_tile_start
  const int scale_offset = simd_lid / (GROUP_SIZE / VPT);  // = simd_lid / 8
  const device float *sc = scales + out_row * num_groups + group_start + scale_offset;
  const device float *bi = biases + out_row * num_groups + group_start + scale_offset;
  // Input pointer
  const device T *xp = x + batch_idx * K + k_tile_start + simd_lid * VPT;

  U result[RESULTS_PER_SG] = {0, 0, 0, 0};

  for (int k = k_tile_start; k < k_tile_end; k += BLOCK_K) {
    // Check bounds
    if (k + simd_lid * VPT >= K) break;

    // Load x values
    U x_vals[VPT];
    U x_sum = 0;
    for (int i = 0; i < VPT; i++) {
      x_vals[i] = static_cast<U>(xp[i]);
      x_sum += x_vals[i];
    }

    for (int row = 0; row < RESULTS_PER_SG; row++) {
      if (out_row + row >= N) break;
      const device uint8_t *wl = wp + row * K;
      U s = sc[row * num_groups];
      U b = bi[row * num_groups];

      U dot = 0;
      for (int i = 0; i < VPT; i++) {
        dot += x_vals[i] * static_cast<U>(wl[i]);
      }
      result[row] += s * dot + b * x_sum;
    }

    wp += BLOCK_K;
    sc += BLOCK_K / GROUP_SIZE;
    bi += BLOCK_K / GROUP_SIZE;
    xp += BLOCK_K;
  }

  // Reduce within SIMD group
  for (int row = 0; row < RESULTS_PER_SG; row++) {
    U total = simd_sum(result[row]);
    if (simd_lid == 0 && (out_row + row) < N) {
      // Atomic add to output (multiple K-tiles contribute)
      device float *out_ptr = y + batch_idx * N + out_row + row;
      // Use atomic for K-tiled version, direct write for single tile
      atomic_fetch_add_explicit(
          (device atomic<float> *)out_ptr, total, memory_order_relaxed);
    }
  }
}

// ── Non-tiled Plan A (original, for small K) ────────────────

template <typename T>
METAL_FUNC void plan_a_direct_impl(
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
  const int scale_step = GROUP_SIZE / VPT;  // 8

  const device uint8_t *wp = (const device uint8_t *)(w + out_row * K)
                           + simd_lid * VPT;
  const device float *sc = scales + out_row * num_groups + simd_lid / scale_step;
  const device float *bi = biases + out_row * num_groups + simd_lid / scale_step;
  const device T *xp = x + tid.x * K + simd_lid * VPT;
  device T *yp = y + tid.x * N + out_row;

  U result[RESULTS_PER_SG] = {0, 0, 0, 0};

  for (int k = 0; k < K; k += BLOCK_K) {
    U x_vals[VPT];
    U x_sum = 0;
    for (int i = 0; i < VPT; i++) {
      x_vals[i] = static_cast<U>(xp[i]);
      x_sum += x_vals[i];
    }

    for (int row = 0; row < RESULTS_PER_SG; row++) {
      if (out_row + row >= N) break;
      const device uint8_t *wl = wp + row * K;
      U s = sc[row * num_groups];
      U b = bi[row * num_groups];

      U dot = 0;
      for (int i = 0; i < VPT; i++) {
        dot += x_vals[i] * static_cast<U>(wl[i]);
      }
      result[row] += s * dot + b * x_sum;
    }

    wp += BLOCK_K;
    sc += BLOCK_K / GROUP_SIZE;
    bi += BLOCK_K / GROUP_SIZE;
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

// Tiled version (for large K, output is float32 for atomic)
[[kernel]] void int8_mv_tiled_f16(
    const device int8_t *w [[buffer(0)]],
    const device float *scales [[buffer(1)]],
    const device float *biases [[buffer(2)]],
    const device half *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  plan_a_tiled_impl<half>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

[[kernel]] void int8_mv_tiled_bf16(
    const device int8_t *w [[buffer(0)]],
    const device float *scales [[buffer(1)]],
    const device float *biases [[buffer(2)]],
    const device bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  plan_a_tiled_impl<bfloat>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

// Direct version (for small K, original behavior)
[[kernel]] void int8_mv_direct_f16(
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
  plan_a_direct_impl<half>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

[[kernel]] void int8_mv_direct_bf16(
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
  plan_a_direct_impl<bfloat>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}
