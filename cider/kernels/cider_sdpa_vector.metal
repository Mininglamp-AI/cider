// cider_sdpa_vector.metal — Template instantiations for Cider v9 SDPA kernels
//
// Naming convention:
//   1-pass:  cider_sdpa_vector_{type}_{D}
//   2-pass1: cider_sdpa_vector_2pass_1_{type}_{D}_b{BLOCKS}
//   2-pass2: cider_sdpa_vector_2pass_2_{type}_{D}_b{BLOCKS}

#include <metal_stdlib>
#include "cider_sdpa_vector.h"

using namespace metal;

// ── Helper macros ────────────────────────────────────────────────
#define instantiate_1pass(type, type_name, D) \
    template [[host_name("cider_sdpa_vector_" #type_name "_" #D)]] \
    [[kernel]] decltype(cider_sdpa_vector<type, D>) cider_sdpa_vector<type, D>;

#define instantiate_2pass(type, type_name, D, B) \
    template [[host_name("cider_sdpa_vector_2pass_1_" #type_name "_" #D "_b" #B)]] \
    [[kernel]] decltype(cider_sdpa_vector_2pass_1<type, D, B>) cider_sdpa_vector_2pass_1<type, D, B>; \
    template [[host_name("cider_sdpa_vector_2pass_2_" #type_name "_" #D "_b" #B)]] \
    [[kernel]] decltype(cider_sdpa_vector_2pass_2<type, D, B>) cider_sdpa_vector_2pass_2<type, D, B>;

#define instantiate_all_blocks(type, type_name, D) \
    instantiate_2pass(type, type_name, D, 32) \
    instantiate_2pass(type, type_name, D, 64) \
    instantiate_2pass(type, type_name, D, 128)

#define instantiate_heads(type, type_name) \
    instantiate_1pass(type, type_name, 64) \
    instantiate_1pass(type, type_name, 96) \
    instantiate_1pass(type, type_name, 128) \
    instantiate_1pass(type, type_name, 256) \
    instantiate_all_blocks(type, type_name, 64) \
    instantiate_all_blocks(type, type_name, 96) \
    instantiate_all_blocks(type, type_name, 128) \
    instantiate_all_blocks(type, type_name, 256)

// ── Instantiate for all types ────────────────────────────────────
instantiate_heads(float, float32)
instantiate_heads(half, float16)
instantiate_heads(bfloat, bfloat16)
