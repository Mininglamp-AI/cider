// Per-group INT8 GEMM/MV primitive header — symmetric quant (no bias)
#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <string>

namespace cider {

namespace mx = mlx::core;

// Per-group INT8 GEMM (prefill) / MV (decode) — symmetric quantization
// Inputs:
//   x: [M, K] float16/bfloat16 — activation
//   w: [N, K] int8 — per-group symmetric quantized weight
//   scale_w: [N, num_groups] float32 — per-group weight scales
//   group_size: 64, 128, or 256
class PerGroupLinear : public mx::Primitive {
public:
  PerGroupLinear(mx::Stream s, const std::string &kernel_dir, int group_size)
      : mx::Primitive(s), kernel_dir_(kernel_dir), group_size_(group_size) {}

  void eval_cpu(const std::vector<mx::array> &,
                std::vector<mx::array> &) override {
    throw std::runtime_error("PerGroupLinear: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "PerGroupLinear"; }
  bool is_equivalent(const mx::Primitive &other) const override { return true; }

private:
  std::string kernel_dir_;
  int group_size_;
};

// Python-facing function
mx::array pergroup_linear(const mx::array &x, const mx::array &w,
                          const mx::array &scale_w, const mx::array &bias,
                          const mx::array &new_bias,
                          int group_size, const std::string &kernel_dir,
                          mx::StreamOrDevice s = {});

} // namespace cider
