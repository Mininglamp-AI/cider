// W4A8 Linear as mlx Custom Primitive.
// Packed INT4 weights × INT8 activations via TensorOps.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <string>

namespace cider {

namespace mx = mlx::core;

// Inputs: [x(M,K) float16, packed_w(K/2,N) uint8, scale_w(N) float32]
// Output: [y(M,N) float16]
//
// Weight layout: [K/2, N] uint8 — packed INT4 symmetric (zero_point=8)
//   high nibble = even k index, low nibble = odd k index
// scale_w: per-column scale (includes group scale pre-folded)
class W4A8Linear : public mx::Primitive {
 public:
  explicit W4A8Linear(mx::Stream stream, const std::string& kernel_dir)
      : mx::Primitive(stream), kernel_dir_(kernel_dir) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("W4A8Linear: CPU not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "W4A8Linear";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    return true;
  }

 private:
  std::string kernel_dir_;
};

mx::array w4a8_linear(
    const mx::array& x,          // [M, K] float16
    const mx::array& packed_w,   // [K/2, N] uint8 (packed INT4)
    const mx::array& scale_w,    // [N] float32
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

}  // namespace cider
