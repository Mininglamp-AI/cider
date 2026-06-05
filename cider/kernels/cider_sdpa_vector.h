// cider_sdpa_vector.h — Cider v9 optimized SDPA vector kernels
//
// Improvements over stock MLX:
//   1. Contiguous chunk layout (cache-friendly sequential access)
//   2. FlashInfer-style register tiling (TILE=4 unroll)
//   3. Adaptive blocks selection per GQA ratio
//
// Template parameters:
//   T      — data type (float, half, bfloat)
//   D      — head dimension (64, 96, 128, 256)
//   BLOCKS — number of blocks for 2-pass (32, 64, 128)
//
// Three kernels:
//   cider_sdpa_vector<T, D>                — 1-pass (short N)
//   cider_sdpa_vector_2pass_1<T, D, BLOCKS> — 2-pass pass1 (per-block partials)
//   cider_sdpa_vector_2pass_2<T, D, BLOCKS> — 2-pass pass2 (reduce)

#pragma once

#include <metal_simdgroup>

using namespace metal;

// ═══════════════════════════════════════════════════════════════
// 1-pass kernel (N <= threshold, no intermediate buffer)
// ═══════════════════════════════════════════════════════════════
template <typename T, int D>
[[kernel]] void cider_sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = D / BD;

    typedef float U;

    const int inner_k_stride = BN * int(k_seq_stride);
    const int inner_v_stride = BN * int(v_seq_stride);

    thread U q_reg[qk_per_thread];
    thread U k_reg[qk_per_thread];
    thread U o_reg[v_per_thread];

    threadgroup U tg_outputs[BN * BD];
    threadgroup U tg_max_scores[BN];
    threadgroup U tg_sum_exp_scores[BN];

    const int q_batch_head_idx = tid.x;
    const int kv_head_idx = q_batch_head_idx / gqa_factor;
    const int o_offset = q_batch_head_idx;

    const device T* q_ptr = queries + o_offset * D + simd_lid * qk_per_thread;
    const device T* k_ptr = keys + kv_head_idx * int(k_head_stride)
                            + simd_gid * int(k_seq_stride) + simd_lid * qk_per_thread;
    const device T* v_ptr = values + kv_head_idx * int(v_head_stride)
                            + simd_gid * int(v_seq_stride) + simd_lid * v_per_thread;

    for (int i = 0; i < qk_per_thread; i++) {
        q_reg[i] = static_cast<U>(scale) * static_cast<U>(q_ptr[i]);
    }
    for (int i = 0; i < v_per_thread; i++) {
        o_reg[i] = 0;
    }

    U max_score = -1e38f;
    U sum_exp_score = 0;

    for (int i = simd_gid; i < N; i += BN) {
        for (int j = 0; j < qk_per_thread; j++) {
            k_reg[j] = static_cast<U>(k_ptr[j]);
        }
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q_reg[j] * k_reg[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
            o_reg[j] = o_reg[j] * factor + exp_score * static_cast<U>(v_ptr[j]);
        }

        k_ptr += inner_k_stride;
        v_ptr += inner_v_stride;
    }

    if (simd_lid == 0) {
        tg_max_scores[simd_gid] = max_score;
        tg_sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = tg_max_scores[simd_lid];
    U new_max_final = simd_max(max_score);
    U factor = fast::exp(max_score - new_max_final);
    sum_exp_score = simd_sum(tg_sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_reg[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid] * factor);
        o_reg[i] = sum_exp_score == 0 ? o_reg[i] : (o_reg[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device T* o_ptr = out + o_offset * D + simd_gid * v_per_thread;
        for (int i = 0; i < v_per_thread; i++) {
            o_ptr[i] = static_cast<T>(o_reg[i]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════
// 2-pass pass1: per-block partial results
// v9: contiguous chunks + TILE=4 register tiling
// ═══════════════════════════════════════════════════════════════
template <typename T, int D, int BLOCKS>
[[kernel]] void cider_sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = D / BD;
    constexpr int TILE = 4;

    typedef float U;

    thread U q_reg[qk_per_thread];
    thread U o_reg[v_per_thread] = {0};

    const int kv_head_idx = tid.x;
    const int batch_idx = tid.y;
    const int block_idx = tid.z;
    const int gqa_factor = tptg.y;
    const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
    const int num_kv_heads = tpg.x;
    const int num_q_heads = num_kv_heads * gqa_factor;
    const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
    const int o_offset = q_batch_head_idx;

    queries += o_offset * D + simd_lid * qk_per_thread;

    const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

    // v9: contiguous chunk layout (not interleaved)
    const int chunk_size = (N + BLOCKS - 1) / BLOCKS;
    const int kv_start = block_idx * chunk_size;
    const int kv_end = min(kv_start + chunk_size, N);

    const device T* k_ptr = keys + kv_batch_head_idx * int(k_head_stride)
                            + kv_start * int(k_seq_stride) + simd_lid * qk_per_thread;
    const device T* v_ptr = values + kv_batch_head_idx * int(v_head_stride)
                            + kv_start * int(v_seq_stride) + simd_lid * v_per_thread;

    device T* o_ptr = partials + o_offset * BLOCKS * D + block_idx * D
                      + simd_lid * v_per_thread;

    // Read query
    for (int i = 0; i < qk_per_thread; i++) {
        q_reg[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
    }

    U max_score = -1e38f;
    U sum_exp_score = 0;

    const int kss = int(k_seq_stride);
    const int vss = int(v_seq_stride);

    // Main loop with TILE=4 unrolling
    int pos = kv_start;
    const int tiled_end = kv_start + ((kv_end - kv_start) / TILE) * TILE;

    for (; pos < tiled_end; pos += TILE) {
        U scores[TILE];
        for (int t = 0; t < TILE; t++) {
            U score = 0;
            const device T* kt = k_ptr + t * kss;
            for (int j = 0; j < qk_per_thread; j++) {
                score += q_reg[j] * static_cast<U>(kt[j]);
            }
            scores[t] = simd_sum(score);
        }
        for (int t = 0; t < TILE; t++) {
            U new_max = max(max_score, scores[t]);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(scores[t] - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            const device T* vt = v_ptr + t * vss;
            for (int j = 0; j < v_per_thread; j++) {
                o_reg[j] = o_reg[j] * factor + exp_score * static_cast<U>(vt[j]);
            }
        }
        k_ptr += TILE * kss;
        v_ptr += TILE * vss;
    }
    // Remainder
    for (; pos < kv_end; pos++) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q_reg[j] * static_cast<U>(k_ptr[j]);
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
            o_reg[j] = o_reg[j] * factor + exp_score * static_cast<U>(v_ptr[j]);
        }
        k_ptr += kss;
        v_ptr += vss;
    }

    // Write partial results
    if (simd_lid == 0) {
        sums[o_offset * BLOCKS + block_idx] = sum_exp_score;
        maxs[o_offset * BLOCKS + block_idx] = max_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
        o_ptr[i] = static_cast<T>(o_reg[i]);
    }
}


// ═══════════════════════════════════════════════════════════════
// 2-pass pass2: reduce partial results across blocks
// ═══════════════════════════════════════════════════════════════
template <typename T, int D, int BLOCKS>
[[kernel]] void cider_sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D / BD;

    typedef float U;

    thread U o_reg[elem_per_thread] = {0};
    threadgroup U tg_outputs[BN * BD];

    const int head_idx = tid.x;

    const device T* p_ptr = partials + head_idx * BLOCKS * D
                            + simd_gid * D + simd_lid * elem_per_thread;
    const device float* s_ptr = sums + head_idx * BLOCKS;
    const device float* m_ptr = maxs + head_idx * BLOCKS;

    // Reduce max
    U max_score = -1e38f;
    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, m_ptr[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);

    // Reduce sum_exp
    U sum_exp_score = 0;
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(m_ptr[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * s_ptr[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    // Reduce partials
    const device float* m_walk = m_ptr;
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(m_walk[simd_gid] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o_reg[i] += factor * static_cast<U>(p_ptr[i]);
        }
        m_walk += BN;
        p_ptr += BN * D;
    }

    // Transpose + reduce via shared memory
    for (int i = 0; i < elem_per_thread; i++) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_reg[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid]);
        o_reg[i] = sum_exp_score == 0 ? o_reg[i] : (o_reg[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device T* o_ptr = out + head_idx * D + simd_gid * elem_per_thread;
        for (int i = 0; i < elem_per_thread; i++) {
            o_ptr[i] = static_cast<T>(o_reg[i]);
        }
    }
}
