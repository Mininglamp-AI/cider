// int8_mv_v2.metal — Two decode MV variants for [N,K] int8 weight
//
// Plan A: weight dequant on-the-fly, activation stays FP16
//   y[n] = sum_g [ scale[n][g] * sum_{k in g}(x_fp[k] * w_uint8[n][k]) + bias[n][g] * sum_{k in g}(x_fp[k]) ]
//   Note: w stored as int8 but interpreted as uint8 via reinterpret_cast
//
// Plan B: activation per-token int8 quant, integer dot, per-group accumulate
//   scale_a = max|x| / 127
//   a_int8[k] = round(x[k] / scale_a)
//   For each group g: accum_g = sum_{k in g}(a_int8[k] * w_int8[n][k])
//   y[n] = scale_a * sum_g[ scale_w[n][g] * accum_g + bias_w[n][g] * sum_a_g ]
//
// Weight layout: [N, K] stored as int8_t (reinterpret from uint32 packed)
// Scales: [N, num_groups] float32
// Biases: [N, num_groups] float32
// group_size = 64

#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int NUM_SIMDGROUPS = 2;
constant constexpr int RESULTS_PER_SG = 4;
constant constexpr int GROUP_SIZE = 64;

// ════════════════════════════════════════════════════════════════
// PLAN A: Weight dequant, FP16 activation (no activation quantization)
// ════════════════════════════════════════════════════════════════
// Each thread processes VALUES_PER_THREAD=8 elements per K-loop iteration
// Block = 8 * 32 = 256 elements per iteration

constant constexpr int VPT_A = 8;
constant constexpr int BLOCK_A = VPT_A * SIMD_SIZE;  // 256
constant constexpr int SCALE_STEP_A = GROUP_SIZE / VPT_A;  // 8

template <typename T>
METAL_FUNC void plan_a_impl(
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

  // Weight pointer: reinterpret int8 as uint8
  const device uint8_t *wp = (const device uint8_t *)(w + out_row * K)
                           + simd_lid * VPT_A;
  const device float *sc = scales + out_row * num_groups + simd_lid / SCALE_STEP_A;
  const device float *bi = biases + out_row * num_groups + simd_lid / SCALE_STEP_A;
  const device T *xp = x + tid.x * K + simd_lid * VPT_A;
  device T *yp = y + tid.x * N + out_row;

  U result[RESULTS_PER_SG] = {0, 0, 0, 0};

  for (int k = 0; k < K; k += BLOCK_A) {
    // Load x and compute sum for bias term
    U x_vals[VPT_A];
    U x_sum = 0;
    for (int i = 0; i < VPT_A; i++) {
      x_vals[i] = static_cast<U>(xp[i]);
      x_sum += x_vals[i];
    }

    for (int row = 0; row < RESULTS_PER_SG; row++) {
      if (out_row + row >= N) break;
      const device uint8_t *wl = wp + row * K;
      U s = sc[row * num_groups];
      U b = bi[row * num_groups];

      U dot = 0;
      for (int i = 0; i < VPT_A; i++) {
        dot += x_vals[i] * static_cast<U>(wl[i]);
      }
      result[row] += s * dot + b * x_sum;
    }

    wp += BLOCK_A;
    sc += BLOCK_A / GROUP_SIZE;
    bi += BLOCK_A / GROUP_SIZE;
    xp += BLOCK_A;
  }

  for (int row = 0; row < RESULTS_PER_SG; row++) {
    U total = simd_sum(result[row]);
    if (simd_lid == 0 && (out_row + row) < N) {
      yp[row] = static_cast<T>(total);
    }
  }
}

// ════════════════════════════════════════════════════════════════
// PLAN B: Activation per-token int8 quant, integer dot, per-group accum
// ════════════════════════════════════════════════════════════════
// Per-group: accumulate int32 within each group, then float dequant
// VPT=8, BLOCK=256, process group boundaries at SCALE_STEP intervals

constant constexpr int VPT_B = 8;
constant constexpr int BLOCK_B = VPT_B * SIMD_SIZE;  // 256
// Groups per block iteration = BLOCK_B / GROUP_SIZE = 4
constant constexpr int GROUPS_PER_BLOCK = BLOCK_B / GROUP_SIZE;  // 4

template <typename T>
METAL_FUNC void plan_b_impl(
    const device int8_t *w [[buffer(0)]],       // [N, K] int8
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

  // First pass: find max|x| for per-token quantization
  U local_max = 0;
  const device T *x_base = x + tid.x * K;
  for (int k = simd_lid; k < K; k += SIMD_SIZE) {
    U val = abs(static_cast<U>(x_base[k]));
    local_max = max(local_max, val);
  }
  local_max = simd_max(local_max);
  U scale_a = local_max / 127.0f;
  U inv_scale_a = (local_max > 0) ? (127.0f / local_max) : 0.0f;

  // Weight and input pointers
  const device int8_t *wp = w + out_row * K + simd_lid * VPT_B;
  const device T *xp = x_base + simd_lid * VPT_B;
  const device float *sc = scales + out_row * num_groups;
  const device float *bi = biases + out_row * num_groups;
  device T *yp = y + tid.x * N + out_row;

  U result[RESULTS_PER_SG] = {0, 0, 0, 0};

  // Process K in blocks of BLOCK_B=256 (=4 groups)
  for (int k = 0; k < K; k += BLOCK_B) {
    // Quantize activation chunk
    int8_t a_vals[VPT_B];
    int32_t a_sum = 0;
    for (int i = 0; i < VPT_B; i++) {
      int kk = k + simd_lid * VPT_B + i;
      if (kk < K) {
        float val = static_cast<U>(xp[i]) * inv_scale_a;
        int v = int(val + (val >= 0 ? 0.5f : -0.5f));
        v = clamp(v, -128, 127);
        a_vals[i] = static_cast<int8_t>(v);
      } else {
        a_vals[i] = 0;
      }
      a_sum += int32_t(a_vals[i]);
    }

    for (int row = 0; row < RESULTS_PER_SG; row++) {
      if (out_row + row >= N) break;
      const device int8_t *wl = wp + row * K;

      // Integer dot product
      int32_t dot = 0;
      for (int i = 0; i < VPT_B; i++) {
        dot += int32_t(a_vals[i]) * int32_t(wl[i]);
      }

      // Per-group dequant: need to figure out which group this thread's data belongs to
      // Thread simd_lid processes elements [k + simd_lid*VPT_B, k + simd_lid*VPT_B + VPT_B)
      // Group index = (k + simd_lid * VPT_B) / GROUP_SIZE
      int group_idx = (k + simd_lid * VPT_B) / GROUP_SIZE;
      U s = sc[row * num_groups + group_idx];
      U b = bi[row * num_groups + group_idx];

      // Check if thread's 8 elements span a group boundary
      int start_k = k + simd_lid * VPT_B;
      int end_k = start_k + VPT_B - 1;
      int start_group = start_k / GROUP_SIZE;
      int end_group = end_k / GROUP_SIZE;

      if (start_group == end_group) {
        // All 8 elements in same group — simple case
        result[row] += scale_a * (s * U(dot) + b * U(a_sum));
      } else {
        // Elements span group boundary — split
        int split = (end_group * GROUP_SIZE) - start_k;  // elements in first group
        int32_t dot1 = 0, dot2 = 0, sum1 = 0, sum2 = 0;
        for (int i = 0; i < split; i++) {
          dot1 += int32_t(a_vals[i]) * int32_t(wl[i]);
          sum1 += int32_t(a_vals[i]);
        }
        for (int i = split; i < VPT_B; i++) {
          dot2 += int32_t(a_vals[i]) * int32_t(wl[i]);
          sum2 += int32_t(a_vals[i]);
        }
        U s2 = sc[row * num_groups + end_group];
        U b2 = bi[row * num_groups + end_group];
        result[row] += scale_a * (s * U(dot1) + b * U(sum1) + s2 * U(dot2) + b2 * U(sum2));
      }
    }

    wp += BLOCK_B;
    xp += BLOCK_B;
  }

  for (int row = 0; row < RESULTS_PER_SG; row++) {
    U total = simd_sum(result[row]);
    if (simd_lid == 0 && (out_row + row) < N) {
      yp[row] = static_cast<T>(total);
    }
  }
}

// ── Entry points ─────────────────────────────────────────────

// Plan A: weight dequant, FP16 activation
[[kernel]] void int8_mv_plan_a_f16(
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
  plan_a_impl<half>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

[[kernel]] void int8_mv_plan_a_bf16(
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
  plan_a_impl<bfloat>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

// Plan B: activation int8 quant, integer dot, per-group accum
[[kernel]] void int8_mv_plan_b_f16(
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
  plan_b_impl<half>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

[[kernel]] void int8_mv_plan_b_bf16(
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
  plan_b_impl<bfloat>(w, scales, biases, x, y, K, N, tid, simd_gid, simd_lid);
}
