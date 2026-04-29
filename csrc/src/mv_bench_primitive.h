#pragma once
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include <string>

namespace w8a8_mlx {
namespace mx = mlx::core;

class MVPlanA : public mx::Primitive {
public:
  explicit MVPlanA(mx::Stream s, const std::string &kdir)
      : mx::Primitive(s), kernel_dir_(kdir) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  const char* name() const override { return "MVPlanA"; }
  bool is_equivalent(const mx::Primitive& other) const override { return true; }
private:
  std::string kernel_dir_;
};

class MVPlanB : public mx::Primitive {
public:
  explicit MVPlanB(mx::Stream s, const std::string &kdir)
      : mx::Primitive(s), kernel_dir_(kdir) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  const char* name() const override { return "MVPlanB"; }
  bool is_equivalent(const mx::Primitive& other) const override { return true; }
private:
  std::string kernel_dir_;
};

class MVPlanATiled : public mx::Primitive {
public:
  explicit MVPlanATiled(mx::Stream s, const std::string &kdir)
      : mx::Primitive(s), kernel_dir_(kdir) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  const char* name() const override { return "MVPlanATiled"; }
  bool is_equivalent(const mx::Primitive& other) const override { return true; }
private:
  std::string kernel_dir_;
};

class MVPlanADirect : public mx::Primitive {
public:
  explicit MVPlanADirect(mx::Stream s, const std::string &kdir)
      : mx::Primitive(s), kernel_dir_(kdir) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  const char* name() const override { return "MVPlanADirect"; }
  bool is_equivalent(const mx::Primitive& other) const override { return true; }
private:
  std::string kernel_dir_;
};

class MVPlanAV4 : public mx::Primitive {
public:
  explicit MVPlanAV4(mx::Stream s, const std::string &kdir)
      : mx::Primitive(s), kernel_dir_(kdir) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  const char* name() const override { return "MVPlanAV4"; }
  bool is_equivalent(const mx::Primitive& other) const override { return true; }
private:
  std::string kernel_dir_;
};

mx::array mv_plan_a(const mx::array &w, const mx::array &scales, const mx::array &biases,
                    const mx::array &x, const std::string &kernel_dir, mx::StreamOrDevice s = {});
mx::array mv_plan_b(const mx::array &w, const mx::array &scales, const mx::array &biases,
                    const mx::array &x, const std::string &kernel_dir, mx::StreamOrDevice s = {});
mx::array mv_plan_a_tiled(const mx::array &w, const mx::array &scales, const mx::array &biases,
                           const mx::array &x, const std::string &kernel_dir, mx::StreamOrDevice s = {});
mx::array mv_plan_a_direct(const mx::array &w, const mx::array &scales, const mx::array &biases,
                            const mx::array &x, const std::string &kernel_dir, mx::StreamOrDevice s = {});
mx::array mv_plan_a_v4(const mx::array &w, const mx::array &scales, const mx::array &biases,
                            const mx::array &x, const std::string &kernel_dir, mx::StreamOrDevice s = {});
} // namespace w8a8_mlx
