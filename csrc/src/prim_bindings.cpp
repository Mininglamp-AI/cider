#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "mlx/ops.h"
#include "pergroup_primitive.h"
#include "sdpa_primitive.h"
#include "w4a8_primitive.h"
#include "w8a8_primitive.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

NB_MODULE(_cider_prim, m) {
  m.doc() = "cider: W8A8 + W4A8 INT8 TensorOps + SDPA primitives for Apple M5+";

  m.def("perchannel_linear", &cider::perchannel_linear, "x"_a, "w"_a, "scale_w"_a,
        "bias"_a, "kernel_dir"_a, nb::kw_only(), "stream"_a = nb::none(),
        "W8A8 quantized linear: y = dequant(quant_a(x) @ w_int8) + bias");

  m.def("w4a8_linear", &cider::w4a8_linear, "x"_a, "packed_w"_a, "scale_w"_a,
        "kernel_dir"_a, nb::kw_only(), "stream"_a = nb::none(),
        "W4A8 quantized linear: y = dequant(quant_a(x) @ unpack4(w))");

  m.def("int8_matmul_int32", &cider::int8_matmul_int32, "a"_a, "b"_a,
        "kernel_dir"_a, nb::kw_only(), "stream"_a = nb::none(),
        "Raw INT8xINT8->INT32 matmul (bit-exact, no dequant)");

  m.def("pergroup_linear", &cider::pergroup_linear, "x"_a, "w"_a, "scale_w"_a,
        "bias"_a, "new_bias"_a, "group_size"_a, "kernel_dir"_a, nb::kw_only(),
        "stream"_a = nb::none(),
        "Per-group INT8 linear with bias: prefill GEMM or decode MV with "
        "per-group scales");

  // ── SDPA ──
  m.def("cider_sdpa_1pass", &cider::cider_sdpa_1pass,
        "queries"_a, "keys"_a, "values"_a,
        "gqa_factor"_a, "scale"_a, "kernel_dir"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        "Cider v9 SDPA 1-pass (short sequences)");

  m.def("cider_sdpa_2pass", &cider::cider_sdpa_2pass,
        "queries"_a, "keys"_a, "values"_a,
        "gqa_factor"_a, "blocks"_a, "scale"_a, "kernel_dir"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        "Cider v9 SDPA 2-pass with contiguous chunks + TILE=4 tiling");
}
