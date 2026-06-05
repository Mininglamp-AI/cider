// Cider SDPA — Custom Primitive for v9 optimized attention
//
// Dispatches 1-pass or 2-pass SDPA kernels via MLX's CommandEncoder.
// All kernel selection (1pass vs 2pass, blocks value) is decided by the
// Python caller and passed as arguments.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <string>

namespace cider {

namespace mx = mlx::core;

// ── 1-pass SDPA Primitive ────────────────────────────────────────
// Inputs: [Q(B*H, D), K(...), V(...)]
// Output: [O(B*H, D)]
class SDPAVector : public mx::Primitive {
 public:
  SDPAVector(mx::Stream stream,
             const std::string& kernel_dir,
             int gqa_factor,
             int N,
             size_t k_head_stride,
             size_t k_seq_stride,
             size_t v_head_stride,
             size_t v_seq_stride,
             float scale)
      : mx::Primitive(stream),
        kernel_dir_(kernel_dir),
        gqa_factor_(gqa_factor),
        N_(N),
        k_head_stride_(k_head_stride),
        k_seq_stride_(k_seq_stride),
        v_head_stride_(v_head_stride),
        v_seq_stride_(v_seq_stride),
        scale_(scale) {}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    throw std::runtime_error("SDPAVector: CPU not supported");
  }

  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  const char* name() const override { return "CiderSDPAVector"; }
  bool is_equivalent(const mx::Primitive& other) const override { return true; }

 private:
  std::string kernel_dir_;
  int gqa_factor_;
  int N_;
  size_t k_head_stride_;
  size_t k_seq_stride_;
  size_t v_head_stride_;
  size_t v_seq_stride_;
  float scale_;
};

// ── 2-pass SDPA Primitive ────────────────────────────────────────
// Inputs: [Q(B*H, D), K(...), V(...)]
// Output: [O(B*H, D)]
// Internally allocates partials/sums/maxs as temporaries.
class SDPAVector2Pass : public mx::Primitive {
 public:
  SDPAVector2Pass(mx::Stream stream,
                  const std::string& kernel_dir,
                  int gqa_factor,
                  int N,
                  int blocks,
                  int num_kv_heads,
                  int batch_size,
                  size_t k_head_stride,
                  size_t k_seq_stride,
                  size_t v_head_stride,
                  size_t v_seq_stride,
                  float scale)
      : mx::Primitive(stream),
        kernel_dir_(kernel_dir),
        gqa_factor_(gqa_factor),
        N_(N),
        blocks_(blocks),
        num_kv_heads_(num_kv_heads),
        batch_size_(batch_size),
        k_head_stride_(k_head_stride),
        k_seq_stride_(k_seq_stride),
        v_head_stride_(v_head_stride),
        v_seq_stride_(v_seq_stride),
        scale_(scale) {}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    throw std::runtime_error("SDPAVector2Pass: CPU not supported");
  }

  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  const char* name() const override { return "CiderSDPAVector2Pass"; }
  bool is_equivalent(const mx::Primitive& other) const override { return true; }

 private:
  std::string kernel_dir_;
  int gqa_factor_;
  int N_;
  int blocks_;
  int num_kv_heads_;
  int batch_size_;
  size_t k_head_stride_;
  size_t k_seq_stride_;
  size_t v_head_stride_;
  size_t v_seq_stride_;
  float scale_;
};

// ── Public API ───────────────────────────────────────────────────

// 1-pass SDPA (short sequences)
mx::array cider_sdpa_1pass(
    const mx::array& queries,   // [B*H, D]
    const mx::array& keys,      // [B*Hkv, N, D]
    const mx::array& values,    // [B*Hkv, N, D]
    int gqa_factor,
    float scale,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

// 2-pass SDPA (long sequences)
mx::array cider_sdpa_2pass(
    const mx::array& queries,   // [B, H, 1, D]
    const mx::array& keys,      // [B, Hkv, N, D]
    const mx::array& values,    // [B, Hkv, N, D]
    int gqa_factor,
    int blocks,
    float scale,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

}  // namespace cider
