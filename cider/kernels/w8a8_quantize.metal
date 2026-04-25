// ============================================================
// Per-token quantization: FP16 → INT8 + float32 scale
// Target: Apple M5, Metal 4
//
// Each threadgroup handles one row (one token).
// Threads cooperate to find absmax via simdgroup reduce,
// then quantize in parallel.
//
// Host dispatch:
//   threadgroup = (min(256, ceil(K/32)*32), 1, 1)
//   grid = (M, 1, 1)
// ============================================================

#include <metal_stdlib>
using namespace metal;

kernel void
quantize_per_token(const device half *X [[buffer(0)]], // [M, K] FP16 input
                   device int8_t *A [[buffer(1)]],     // [M, K] INT8 output
                   device float *scale [[buffer(2)]],  // [M] float32 scale
                   constant uint &M [[buffer(3)]],
                   constant uint &K [[buffer(4)]],
                   uint gid [[threadgroup_position_in_grid]], // row index
                   uint lid [[thread_index_in_threadgroup]],
                   uint tg_size [[threads_per_threadgroup]]) {
  if (gid >= M) {
    return;
  }

  const device half *row_in = X + gid * K;
  device int8_t *row_out = A + gid * K;

  // Step 1: Find local absmax
  float local_max = 0.0f;
  for (uint i = lid; i < K; i += tg_size) {
    float v = abs(float(row_in[i]));
    local_max = max(local_max, v);
  }

  // Step 2: Simdgroup reduce max
  float sg_max = simd_max(local_max);

  // Step 3: Threadgroup reduce across simdgroups via shared memory
  threadgroup float sg_maxes[8]; // up to 8 simdgroups (256/32)
  threadgroup float shared_scale;
  threadgroup float shared_inv_scale;
  uint sg_id = lid / 32;
  uint sg_lid = lid % 32;
  if (sg_lid == 0) {
    sg_maxes[sg_id] = sg_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final reduce (first simdgroup only)
  if (sg_id == 0) {
    float row_max = 0.0f;
    uint num_sgs = (tg_size + 31) / 32;
    if (sg_lid < num_sgs) {
      row_max = sg_maxes[sg_lid];
    }
    row_max = simd_max(row_max);

    // Compute and broadcast scale
    float s = row_max / 127.0f;
    if (s == 0.0f) {
      s = 1.0f;
    }

    if (sg_lid == 0) {
      shared_scale = s;
      shared_inv_scale = 1.0f / s;
      // Store scale to output
      scale[gid] = s;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 4: All threads read broadcasted scale
  float inv_s = shared_inv_scale;

  // Step 5: Quantize
  for (uint i = lid; i < K; i += tg_size) {
    float v = float(row_in[i]) * inv_s;
    v = clamp(round(v), -128.0f, 127.0f);
    row_out[i] = int8_t(v);
  }
}
