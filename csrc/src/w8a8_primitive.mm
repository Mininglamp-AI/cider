// W8A8 Linear V6 — Independent Scratch Buffers (no barrier)
//
// V5 used a shared ScratchPool with barrier between quantize & matmul.
// This forced strict GPU serialization, preventing Metal from pipelining
// operations across layers.
//
// V6 allocates independent scratch buffers per call (via add_temporary),
// removes the barrier, and lets Metal pipeline all dispatches freely.
// Trade-off: ~4-12MB scratch per call × 112 calls = ~672MB temporaries,
// but Metal's allocator reclaims them after eval.

#include "w8a8_primitive.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace w8a8_mlx {

using namespace mlx::core;

struct TileLarge {
  static constexpr uint32_t BM = 128, BN = 128, THREADS = 512;
};
struct TileSmall {
  static constexpr uint32_t BM = 32, BN = 128, THREADS = 128;
};

static std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open: " + path);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  f.close();
  return ss.str();
}

// ── Pipeline cache (no scratch pool in V6) ──────────────────────
class PipelineCache {
public:
  static PipelineCache &instance() {
    static PipelineCache cache;
    return cache;
  }

  void ensure_init(const std::string &kernel_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_ && kernel_dir_ == kernel_dir) {
      return;
    }
    auto &dev = metal::device(mlx::core::Device::gpu);
    auto *mtl_device = dev.mtl_device();
    matmul_lib_ = compile_source(mtl_device, kernel_dir + "/w8a8_matmul.metal");
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
  PipelineCache() = default;
  bool initialized_ = false;
  std::string kernel_dir_;
  std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines_;
  MTL::Library *matmul_lib_ = nullptr;
  MTL::Library *quantize_lib_ = nullptr;
  std::mutex mutex_;

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
      id<MTLLibrary> lib_objc = (__bridge id<MTLLibrary>)matmul_lib_;
      id<MTLFunction> func = [lib_objc newFunctionWithName:fn_name];
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

void W8A8Linear::eval_gpu(const std::vector<array> &inputs,
                          std::vector<array> &outputs) {
  auto &x = inputs[0];       // [M, K] float16
  auto &w = inputs[1];       // [K, N] int8
  auto &scale_w = inputs[2]; // [N] float32
  auto &out = outputs[0];    // [M, N] float16

  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(w.shape(0));
  uint32_t N = static_cast<uint32_t>(w.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  auto &cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  // V6: allocate independent scratch buffers per call
  size_t a_bytes = static_cast<size_t>(M) * K;
  size_t sa_bytes = static_cast<size_t>(M) * sizeof(float);

  array a_int8({static_cast<int>(M), static_cast<int>(K)}, int8, nullptr, {});
  a_int8.set_data(allocator::malloc(a_bytes));

  array sa({static_cast<int>(M)}, float32, nullptr, {});
  sa.set_data(allocator::malloc(sa_bytes));

  auto &s = stream();
  auto &enc = metal::get_command_encoder(s);

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

  // V6: barrier still needed between quantize and matmul within THIS call
  // (matmul reads the scratch that quantize just wrote)
  enc.barrier();

  // DISPATCH 2: matmul_dequant
  {
    bool use_small = (M <= 64);
    const char *kname = use_small ? "w8a8_matmul_fused_dequant_small"
                                  : "w8a8_matmul_fused_dequant";
    uint32_t BM = use_small ? TileSmall::BM : TileLarge::BM;
    uint32_t BN = use_small ? TileSmall::BN : TileLarge::BN;
    uint32_t threads = use_small ? TileSmall::THREADS : TileLarge::THREADS;

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

    enc.dispatch_threadgroups(MTL::Size::Make(grid_x, grid_y, 1),
                              MTL::Size::Make(threads, 1, 1));
  }

  // V6: register scratch buffers as temporaries so MLX frees them after eval
  // This allows Metal to pipeline across different W8A8Linear calls
  // (each call's quantize/matmul pair is still serial via the barrier above,
  //  but different calls can overlap since they use independent buffers)
  enc.add_temporary(a_int8);
  enc.add_temporary(sa);
}

array w8a8_linear(const array &x, const array &w, const array &scale_w,
                  const std::string &kernel_dir, StreamOrDevice s) {
  if (x.ndim() != 2) {
    throw std::invalid_argument("w8a8_linear: x must be 2D [M,K]");
  }
  if (w.ndim() != 2) {
    throw std::invalid_argument("w8a8_linear: w must be 2D [K,N]");
  }

  int M = x.shape(0);
  int N = w.shape(1);
  auto stream = to_stream(s);

  // Kernel computes in float16 internally
  auto result =
      array({M, N}, float16, std::make_shared<W8A8Linear>(stream, kernel_dir),
            {astype(x, float16, stream), astype(w, int8, stream),
             astype(scale_w, float32, stream)});
  // If input was bfloat16, cast output to bfloat16 to match model dtype
  // This is a lazy cast that MLX can fuse with downstream ops
  if (x.dtype() == bfloat16) {
    return astype(result, bfloat16, stream);
  }
  return result;
}

// ── Int8MatMulInt32 primitive (raw INT32 output, no dequant) ────

void Int8MatMulInt32::eval_gpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &a = inputs[0]; // [M, K] int8
  auto &b = inputs[1]; // [K, N] int8

  uint32_t M = static_cast<uint32_t>(a.shape(0));
  uint32_t K = static_cast<uint32_t>(a.shape(1));
  uint32_t N = static_cast<uint32_t>(b.shape(1));

  auto &out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  bool use_small = (M <= 64);
  auto *pso = use_small ? cache.get("int8_matmul_int32_small")
                        : cache.get("int8_matmul_int32");

  uint32_t BM = use_small ? 32 : 128;
  uint32_t BN = 128;
  uint32_t threads = use_small ? 128 : 512;

  uint32_t tiles_n = (N + BN - 1) / BN;
  uint32_t tiles_m = (M + BM - 1) / BM;
  uint32_t swizzle_log;
  if (tiles_n >= 4) {
    swizzle_log = 2;
  } else if (tiles_n >= 2) {
    swizzle_log = 1;
  } else {
    swizzle_log = 0;
  }
  uint32_t tile = 1u << swizzle_log;
  uint32_t grid_x = tiles_n * tile;
  uint32_t grid_y = (tiles_m + tile - 1) / tile;

  auto &s = stream();
  auto &enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(pso);
  enc.set_input_array(a, 0);
  enc.set_input_array(b, 1);
  enc.set_output_array(out, 2);
  enc.set_bytes(&M, sizeof(M), 3);
  enc.set_bytes(&N, sizeof(N), 4);
  enc.set_bytes(&K, sizeof(K), 5);
  enc.set_bytes(&swizzle_log, sizeof(swizzle_log), 6);
  enc.set_bytes(&tiles_m, sizeof(tiles_m), 7);
  enc.set_bytes(&tiles_n, sizeof(tiles_n), 8);
  enc.dispatch_threadgroups(MTL::Size(grid_x, grid_y, 1),
                            MTL::Size(threads, 1, 1));
}

mx::array int8_matmul_int32(const mx::array &a, const mx::array &b,
                            const std::string &kernel_dir,
                            mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(a.shape(0));
  uint32_t N = static_cast<uint32_t>(b.shape(1));
  return mx::array(
      {static_cast<int>(M), static_cast<int>(N)}, mx::int32,
      std::make_shared<Int8MatMulInt32>(mx::to_stream(s), kernel_dir),
      {astype(a, int8, mx::to_stream(s)), astype(b, int8, mx::to_stream(s))});
}

} // namespace w8a8_mlx
