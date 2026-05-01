// W8A8 Linear as mlx Custom Primitive.
// Dispatches quantize + INT8 matmul (prefill) or FP MV (decode) via mlx's CommandEncoder.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace MTL {
class ComputePipelineState;
}

namespace cider {

namespace mx = mlx::core;

// ── Custom Primitive ─────────────────────────────────────────────
// Inputs: [x(M,K) float16, w(N,K) int8, scale_w(N) float32, bias(N) float16]
// Output: [y(M,N) float16]
//
// M > 1: quantize activation + INT8 GEMM (prefill)
// M == 1: FP activation × dequant weight MV (decode)
class W8A8Linear : public mx::Primitive {
 public:
  explicit W8A8Linear(mx::Stream stream, const std::string& kernel_dir)
      : mx::Primitive(stream), kernel_dir_(kernel_dir) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("W8A8Linear: CPU not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "W8A8Linear";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    return true;
  }

 private:
  std::string kernel_dir_;
};


// ── Raw INT32 Matmul Primitive ───────────────────────────────────
// Inputs: [A(M,K) int8, B(K,N) int8]
// Output: [C(M,N) int32]   (bit-exact, no dequant)
class Int8MatMulInt32 : public mx::Primitive {
 public:
  explicit Int8MatMulInt32(mx::Stream stream, const std::string& kernel_dir)
      : mx::Primitive(stream), kernel_dir_(kernel_dir) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("Int8MatMulInt32: CPU not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "Int8MatMulInt32";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    return true;
  }

 private:
  std::string kernel_dir_;
};

// ── Public API ───────────────────────────────────────────────────
mx::array perchannel_linear(
    const mx::array& x,       // [M, K] float16
    const mx::array& w,       // [N, K] int8
    const mx::array& scale_w, // [N] float32
    const mx::array& bias,    // [N] float16
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});


// ── Raw INT32 matmul (for bit-exact testing) ─────────────────────
mx::array int8_matmul_int32(
    const mx::array& a,       // [M, K] int8
    const mx::array& b,       // [K, N] int8
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

}  // namespace cider
