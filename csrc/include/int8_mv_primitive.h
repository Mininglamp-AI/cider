// INT8 per-group quantized matrix-vector multiply primitive
// y = sum_g [ scale[n][g] * dot(x_g, w_uint8_g) + bias[n][g] * sum(x_g) ]
// On-the-fly dequant from packed uint32 weights.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <string>

namespace w8a8_mlx {

namespace mx = mlx::core;

class Int8MV : public mx::Primitive {
public:
  explicit Int8MV(mx::Stream stream, const std::string &kernel_dir)
      : mx::Primitive(stream), kernel_dir_(kernel_dir) {}

  void eval_cpu(const std::vector<mx::array> &,
                std::vector<mx::array> &) override {
    throw std::runtime_error("Int8MV: CPU not supported");
  }

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "Int8MV"; }
  bool is_equivalent(const mx::Primitive &) const override { return true; }

private:
  std::string kernel_dir_;
};

// Public API — per-group quantized MV
// w_packed: [N, K/4] uint32 (packed uint8)
// scales:   [N, n_groups] float32
// biases:   [N, n_groups] float32
// x:        [B, K] float16/bfloat16/float32
mx::array int8_mv(
    const mx::array &w_packed,
    const mx::array &scales,
    const mx::array &biases,
    const mx::array &x,
    const std::string &kernel_dir,
    mx::StreamOrDevice s = {});

} // namespace w8a8_mlx
