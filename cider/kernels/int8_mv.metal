// int8_mv.metal — Per-channel INT8 matrix-vector for decode (M=1)
// Modeled after MLX's qmv_fast_impl.
//
// Weight: [N, K] int8 (per-row/per-channel quantized)
// scale_w: [N] float32 (one scale per output channel)
//
// Activation is quantized per-token on-the-fly:
//   a_int8[k] = round(x[k] / scale_a), scale_a = max(|x|) / 127
//
// y[n] = scale_a * scale_w[n] * sum_k(a_int8[k] * w_int8[n][k])
//
// Thread layout (same as MLX qmv_fast):
//   NUM_SIMDGROUPS = 2, RESULTS_PER_SG = 4
//   → 8 output rows per threadgroup
//   VALUES_PER_THREAD = 16, BLOCK_SIZE = 16 * 32 = 512

#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int NUM_SIMDGROUPS = 2;
constant constexpr int RESULTS_PER_SG = 4;
constant constexpr int VALUES_PER_THREAD = 16;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;  // 512

template <typename T>
METAL_FUNC void int8_mv_impl(
    const device int8_t *w [[buffer(0)]],         // [N, K] int8
    const device float *scale_w [[buffer(1)]],     // [N] float32
    const device T *x [[buffer(2)]],               // [B, K]
    device T *y [[buffer(3)]],                     // [B, N]
    const constant int &K [[buffer(4)]],
    const constant int &N [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  typedef float U;

  const int out_row = tid.y * (NUM_SIMDGROUPS * RESULTS_PER_SG)
                    + simd_gid * RESULTS_PER_SG;

  // Weight pointer: row out_row, column simd_lid * VALUES_PER_THREAD
  const device int8_t *wp = w + out_row * K + simd_lid * VALUES_PER_THREAD;

  // Input pointer
  const device T *xp = x + tid.x * K + simd_lid * VALUES_PER_THREAD;

  // Output pointer
  device T *yp = y + tid.x * N + out_row;

  // On-the-fly activation quantization:
  // First pass: find max |x| across the row via simd reduction
  U local_max = 0;
  for (int k = simd_lid * VALUES_PER_THREAD; k < K; k += BLOCK_SIZE) {
    for (int i = 0; i < VALUES_PER_THREAD && (k + i) < K; i++) {
      U val = abs(static_cast<U>(x[tid.x * K + k + i]));
      local_max = max(local_max, val);
    }
  }
  local_max = simd_max(local_max);
  U scale_a = local_max / 127.0f;
  U inv_scale_a = (scale_a > 0) ? (127.0f / local_max) : 0.0f;

  // Accumulators
  int32_t result[RESULTS_PER_SG] = {0, 0, 0, 0};

  // K-loop: int8 dot product
  for (int k = 0; k < K; k += BLOCK_SIZE) {
    // Quantize activation on-the-fly
    int8_t a_vals[VALUES_PER_THREAD];
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      int kk = k + simd_lid * VALUES_PER_THREAD + i;
      if (kk < K) {
        U val = static_cast<U>(xp[i]) * inv_scale_a;
        a_vals[i] = static_cast<int8_t>(clamp(val + 0.5f, -128.0f, 127.0f));
      } else {
        a_vals[i] = 0;
      }
    }

    // Dot product for each output row
    for (int row = 0; row < RESULTS_PER_SG; row++) {
      const device int8_t *wl = wp + row * K;
      int32_t accum = 0;
      for (int i = 0; i < VALUES_PER_THREAD; i++) {
        accum += static_cast<int32_t>(a_vals[i]) * static_cast<int32_t>(wl[i]);
      }
      result[row] += accum;
    }

    wp += BLOCK_SIZE;
    xp += BLOCK_SIZE;
  }

  // Reduce across SIMD, dequant, and store
  for (int row = 0; row < RESULTS_PER_SG; row++) {
    int32_t total = simd_sum(result[row]);
    if (simd_lid == 0) {
      int n_idx = out_row + row;
      if (n_idx < N) {
        U out_val = scale_a * scale_w[n_idx] * static_cast<U>(total);
        yp[row] = static_cast<T>(out_val);
      }
    }
  }
}

// ── Kernel entry points ──────────────────────────────────────
[[kernel]] void int8_mv_float16(
    const device int8_t *w [[buffer(0)]],
    const device float *scale_w [[buffer(1)]],
    const device half *x [[buffer(2)]],
    device half *y [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    const constant int &N [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int8_mv_impl<half>(w, scale_w, x, y, K, N, tid, simd_gid, simd_lid);
}

[[kernel]] void int8_mv_bfloat16(
    const device int8_t *w [[buffer(0)]],
    const device float *scale_w [[buffer(1)]],
    const device bfloat *x [[buffer(2)]],
    device bfloat *y [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    const constant int &N [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int8_mv_impl<bfloat>(w, scale_w, x, y, K, N, tid, simd_gid, simd_lid);
}
