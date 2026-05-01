// Per-group INT8 GEMM primitive implementation
//
// Dispatches:
//   1. quantize_per_token: x[M,K] float16 → a_int8[M,K] int8 + scale_a[M]
//   float32
//   2. pergroup_int8_gemm_gXX: A[M,K] int8 × B[N,K] int8 → C[M,N] float16
//      with per-group scale_w[N, num_groups] and per-token scale_a[M]

#include "pergroup_primitive.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace cider {

using namespace mlx::core;

// ── Pipeline cache for per-group kernels ────────────────────────
class PerGroupPipelineCache {
public:
  static PerGroupPipelineCache &instance() {
    static PerGroupPipelineCache cache;
    return cache;
  }

  void ensure_init(const std::string &kernel_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_ && kernel_dir_ == kernel_dir)
      return;
    auto &dev = metal::device(mlx::core::Device::gpu);
    auto *mtl_device = dev.mtl_device();
    pergroup_lib_ =
        compile_source(mtl_device, kernel_dir + "/pergroup_int8_gemm.metal");
    mv_lib_ =
        compile_source(mtl_device, kernel_dir + "/pergroup_int8_mv.metal");
    quantize_lib_ =
        compile_source(mtl_device, kernel_dir + "/w8a8_quantize.metal");
    kernel_dir_ = kernel_dir;
    pipelines_.clear();
    initialized_ = true;
  }

  MTL::ComputePipelineState *get(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) {
      return it->second;
    }
    auto &dev = metal::device(mlx::core::Device::gpu);
    auto *pso = make_pipeline(dev.mtl_device(), name);
    pipelines_[name] = pso;
    return pso;
  }

private:
  PerGroupPipelineCache() = default;
  bool initialized_ = false;
  std::string kernel_dir_;
  std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines_;
  MTL::Library *pergroup_lib_ = nullptr;
  MTL::Library *mv_lib_ = nullptr;
  MTL::Library *quantize_lib_ = nullptr;
  std::mutex mutex_;

  static std::string read_file(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) {
      throw std::runtime_error("Cannot open: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }

  MTL::Library *compile_source(MTL::Device *mtl_device,
                               const std::string &source_path) {
    std::string source = read_file(source_path);
    @autoreleasepool {
      NSString *src = [NSString stringWithUTF8String:source.c_str()];
      MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
      opts.languageVersion = MTLLanguageVersion4_0;
      NSError *error = nil;
      id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
      id<MTLLibrary> lib_objc = [device_objc newLibraryWithSource:src
                                                          options:opts
                                                            error:&error];
      if (!lib_objc) {
        std::string err =
            error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("Metal compile failed (" + source_path +
                                 "): " + err);
      }
      return (__bridge MTL::Library *)(void *)CFBridgingRetain(lib_objc);
    }
  }

  MTL::ComputePipelineState *make_pipeline(MTL::Device *mtl_device,
                                           const std::string &name) {
    @autoreleasepool {
      NSString *fn_name = [NSString stringWithUTF8String:name.c_str()];
      // Try pergroup lib, then mv lib, then quantize lib
      id<MTLFunction> func = nil;
      id<MTLLibrary> lib_objc = (__bridge id<MTLLibrary>)pergroup_lib_;
      func = [lib_objc newFunctionWithName:fn_name];
      if (!func) {
        lib_objc = (__bridge id<MTLLibrary>)mv_lib_;
        func = [lib_objc newFunctionWithName:fn_name];
      }
      if (!func) {
        lib_objc = (__bridge id<MTLLibrary>)quantize_lib_;
        func = [lib_objc newFunctionWithName:fn_name];
      }
      if (!func) {
        throw std::runtime_error("Kernel not found: " + name);
      }
      NSError *error = nil;
      id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
      id<MTLComputePipelineState> pso =
          [device_objc newComputePipelineStateWithFunction:func error:&error];
      if (!pso) {
        std::string err =
            error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("Pipeline failed for " + name + ": " + err);
      }
      return (__bridge MTL::ComputePipelineState *)(void *)CFBridgingRetain(
          pso);
    }
  }
};

void PerGroupLinear::eval_gpu(const std::vector<array> &inputs,
                              std::vector<array> &outputs) {
  auto &x = inputs[0];       // [M, K] float16
  auto &w = inputs[1];       // [N, K] int8
  auto &scale_w = inputs[2]; // [N, num_groups] float32
  auto &bias = inputs[3];    // [N] float16
  auto &new_bias = inputs[4]; // [N, num_groups] float32 (asymmetric correction)
  auto &out = outputs[0];    // [M, N] float16

  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t N = static_cast<uint32_t>(w.shape(0));
  uint32_t K = static_cast<uint32_t>(w.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  auto &cache = PerGroupPipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  auto &s = stream();
  auto &enc = metal::get_command_encoder(s);

  if (M == 1) {
    // ── Decode path: per-group MV kernel ──
    // No activation quantization needed; directly use FP16 activation
    std::string kname;
    if (group_size_ == 64) {
      kname = "pergroup_int8_mv_g64";
    } else if (group_size_ == 128) {
      kname = "pergroup_int8_mv_g128";
    } else {
      kname = "pergroup_int8_mv_g256";
    }

    auto *pso = cache.get(kname);
    enc.set_compute_pipeline_state(pso);

    constexpr uint32_t TOTAL_ROWS = 8; // NUM_SIMDGROUPS(2) * RESULTS_PER_SG(4)
    uint32_t threadgroups = (N + TOTAL_ROWS - 1) / TOTAL_ROWS;
    uint32_t threads = 64; // 2 simdgroups x 32 threads

    enc.set_input_array(x, 0);       // [1, K] float16 → pass as [K]
    enc.set_input_array(w, 1);       // [N, K] int8
    enc.set_output_array(out, 2);    // [1, N] float16 → write as [N]
    enc.set_input_array(scale_w, 3); // [N, num_groups] float32
    enc.set_bytes(N, 4);
    enc.set_bytes(K, 5);
    enc.set_input_array(bias, 6); // [N] float16
    enc.set_input_array(new_bias, 7); // [N, num_groups] float32 (correction)

    enc.dispatch_threadgroups(MTL::Size::Make(threadgroups, 1, 1),
                              MTL::Size::Make(threads, 1, 1));
  } else {
    // ── Prefill path: per-group GEMM with activation quantization ──

    // Scratch: activation quantization
    size_t a_bytes = static_cast<size_t>(M) * K;
    size_t sa_bytes = static_cast<size_t>(M) * sizeof(float);

    array a_int8({static_cast<int>(M), static_cast<int>(K)}, int8, nullptr, {});
    a_int8.set_data(allocator::malloc(a_bytes));

    array sa({static_cast<int>(M)}, float32, nullptr, {});
    sa.set_data(allocator::malloc(sa_bytes));

    // DISPATCH 1: quantize_per_token
    {
      auto *pso = cache.get("quantize_per_token");
      enc.set_compute_pipeline_state(pso);
      enc.set_input_array(x, 0);
      enc.set_output_array(a_int8, 1);
      enc.set_output_array(sa, 2);
      enc.set_bytes(M, 3);
      enc.set_bytes(K, 4);
      uint32_t tg = std::min(256u, std::max(32u, ((K + 31) / 32) * 32));
      enc.dispatch_threadgroups(MTL::Size::Make(M, 1, 1),
                                MTL::Size::Make(tg, 1, 1));
    }

    enc.barrier();

    // DISPATCH 2: per-group GEMM
    {
      std::string kname;
      bool use_small = (M <= 64);
      uint32_t BM, BN, threads;

      if (group_size_ == 64) {
        kname = use_small ? "pergroup_int8_gemm_g64_small"
                          : "pergroup_int8_gemm_g64";
      } else if (group_size_ == 128) {
        kname = use_small ? "pergroup_int8_gemm_g128_small"
                          : "pergroup_int8_gemm_g128";
      } else {
        kname = use_small ? "pergroup_int8_gemm_g256_small"
                          : "pergroup_int8_gemm_g256";
      }

      BM = use_small ? 32 : 128;
      BN = 128;
      threads = use_small ? 128 : 512;

      auto *pso = cache.get(kname);
      enc.set_compute_pipeline_state(pso);

      uint32_t tiles_n = (N + BN - 1) / BN;
      uint32_t tiles_m = (M + BM - 1) / BM;
      uint32_t swizzle_log;
      if (tiles_m <= 3) {
        swizzle_log = 0;
      } else if (tiles_m <= 6) {
        swizzle_log = 1;
      } else {
        swizzle_log = 2;
      }

      uint32_t tile = 1u << swizzle_log;
      uint32_t grid_x = tiles_n * tile;
      uint32_t grid_y = (tiles_m + tile - 1) / tile;

      enc.set_input_array(a_int8, 0);
      enc.set_input_array(w, 1);
      enc.set_output_array(out, 2);
      enc.set_bytes(M, 3);
      enc.set_bytes(N, 4);
      enc.set_bytes(K, 5);
      enc.set_input_array(sa, 6);
      enc.set_input_array(scale_w, 7);
      enc.set_bytes(swizzle_log, 8);
      enc.set_bytes(tiles_m, 9);
      enc.set_bytes(tiles_n, 10);
      enc.set_input_array(bias, 11);

      enc.dispatch_threadgroups(MTL::Size::Make(grid_x, grid_y, 1),
                                MTL::Size::Make(threads, 1, 1));
    }

    enc.add_temporary(a_int8);
    enc.add_temporary(sa);
  }
}

array pergroup_linear(const array &x, const array &w, const array &scale_w,
                      const array &bias, const array &new_bias, int group_size,
                      const std::string &kernel_dir, StreamOrDevice s) {
  if (x.ndim() != 2) {
    throw std::invalid_argument("pergroup_linear: x must be 2D [M,K]");
  }
  if (w.ndim() != 2) {
    throw std::invalid_argument("pergroup_linear: w must be 2D [N,K]");
  }
  if (scale_w.ndim() != 2) {
    throw std::invalid_argument(
        "pergroup_linear: scale_w must be 2D [N,num_groups]");
  }
  if (group_size != 64 && group_size != 128 && group_size != 256) {
    throw std::invalid_argument(
        "pergroup_linear: group_size must be 64, 128, or 256");
  }

  int M = x.shape(0);
  int N = w.shape(0);
  auto stream = to_stream(s);

  auto result =
      array({M, N}, float16,
            std::make_shared<PerGroupLinear>(stream, kernel_dir, group_size),
            {astype(x, float16, stream), astype(w, int8, stream),
             astype(scale_w, float32, stream), astype(bias, float16, stream),
             astype(new_bias, float32, stream)});

  if (x.dtype() == bfloat16) {
    return astype(result, bfloat16, stream);
  }
  return result;
}

} // namespace cider
