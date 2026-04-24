#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/optional.h>

#include "mlx/ops.h"
#include "w8a8_primitive.h"
#include "w4a8_primitive.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

NB_MODULE(_cider_prim, m) {
  m.doc() = "cider: W8A8 + W4A8 INT8 TensorOps primitives for Apple M5+";

  m.def(
      "w8a8_linear",
      &w8a8_mlx::w8a8_linear,
      "x"_a, "w"_a, "scale_w"_a, "kernel_dir"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      "W8A8 quantized linear: y = dequant(quant_a(x) @ w_int8)");

  m.def(
      "w4a8_linear",
      &w8a8_mlx::w4a8_linear,
      "x"_a, "packed_w"_a, "scale_w"_a, "kernel_dir"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      "W4A8 quantized linear: y = dequant(quant_a(x) @ unpack4(w))");

  m.def(
      "int8_matmul_int32",
      &w8a8_mlx::int8_matmul_int32,
      "a"_a, "b"_a, "kernel_dir"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      "Raw INT8xINT8->INT32 matmul (bit-exact, no dequant)");
}
