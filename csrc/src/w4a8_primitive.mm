// W4A8 Linear V1 — Packed INT4 weights × INT8 activations
// Reuses W8A8's scratch buffer pool for activation quantization.
// Kernel: unpack W4→INT8 in threadgroup mem, then TensorOps matmul2d.

#include "w4a8_primitive.h"
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

// Scratch buffer pool (same pattern as W8A8)
struct ScratchPool {
  allocator::Buffer a_int8_buf{nullptr};
  size_t a_int8_size = 0;
  allocator::Buffer scale_a_buf{nullptr};
  size_t scale_a_size = 0;

  allocator::Buffer get_a_int8(size_t needed) {
    if (a_int8_buf.ptr() && a_int8_size >= needed) {
      return a_int8_buf;
    }
    if (a_int8_buf.ptr()) {
      allocator::free(a_int8_buf);
    }
    a_int8_buf = allocator::malloc(needed);
    a_int8_size = needed;
    return a_int8_buf;
  }
  allocator::Buffer get_scale_a(size_t needed) {
    if (scale_a_buf.ptr() && scale_a_size >= needed) {
      return scale_a_buf;
    }
    if (scale_a_buf.ptr()) {
      allocator::free(scale_a_buf);
    }
    scale_a_buf = allocator::malloc(needed);
    scale_a_size = needed;
    return scale_a_buf;
  }
  ~ScratchPool() {
    if (a_int8_buf.ptr()) {
      allocator::free(a_int8_buf);
    }
    if (scale_a_buf.ptr()) {
      allocator::free(scale_a_buf);
    }
  }
};

class W4A8PipelineCache {
public:
  static W4A8PipelineCache &instance() {
    static W4A8PipelineCache cache;
    return cache;
  }

  void ensure_init(const std::string &kernel_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_ && kernel_dir_ == kernel_dir) {
      return;
    }
    auto &dev = metal::device(mlx::core::Device::gpu);
    auto *mtl_device = dev.mtl_device();
    w4a8_lib_ = compile_source(mtl_device, kernel_dir + "/w4a8_matmul.metal");
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

  ScratchPool &scratch() { return scratch_pool_; }

private:
  W4A8PipelineCache() = default;
  bool initialized_ = false;
  std::string kernel_dir_;
  std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines_;
  MTL::Library *w4a8_lib_ = nullptr;
  MTL::Library *quantize_lib_ = nullptr;
  ScratchPool scratch_pool_;
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
      // Try w4a8 lib first, then quantize lib
      id<MTLLibrary> lib_objc = (__bridge id<MTLLibrary>)w4a8_lib_;
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

void W4A8Linear::eval_gpu(const std::vector<array> &inputs,
                          std::vector<array> &outputs) {
  auto &x = inputs[0];        // [M, K] float16
  auto &packed_w = inputs[1]; // [K/2, N] uint8
  auto &scale_w = inputs[2];  // [N] float32
  auto &out = outputs[0];     // [M, N] float16

  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t N = static_cast<uint32_t>(packed_w.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  auto &cache = W4A8PipelineCache::instance();
  cache.ensure_init(kernel_dir_);
  auto &pool = cache.scratch();

  size_t a_bytes = static_cast<size_t>(M) * K;
  size_t sa_bytes = static_cast<size_t>(M) * sizeof(float);

  allocator::Buffer a_int8_buf = pool.get_a_int8(a_bytes);
  allocator::Buffer sa_buf = pool.get_scale_a(sa_bytes);

  auto noop = [](allocator::Buffer) {};
  array a_int8(a_int8_buf, {static_cast<int>(M), static_cast<int>(K)}, int8,
               noop);
  array sa(sa_buf, {static_cast<int>(M)}, float32, noop);

  auto &s = stream();
  // auto& dev removed — use free function get_command_encoder
  auto &enc = metal::get_command_encoder(s);

  // DISPATCH 1: quantize activations (reuse w8a8_quantize.metal)
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

  // DISPATCH 2: W4A8 matmul with fused dequant
  {
    bool use_small = (M <= 64);
    const char *kname = use_small ? "w4a8_matmul_fused_dequant_small"
                                  : "w4a8_matmul_fused_dequant";
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

    enc.set_input_array(a_int8, 0);   // A: INT8 activations
    enc.set_input_array(packed_w, 1); // B: packed W4
    enc.set_output_array(out, 2);     // C: FP16 output
    enc.set_bytes(M, 3);
    enc.set_bytes(N, 4);
    enc.set_bytes(K, 5);
    enc.set_input_array(sa, 6);      // scale_a
    enc.set_input_array(scale_w, 7); // scale_w
    enc.set_bytes(swizzle_log, 8);
    enc.set_bytes(tiles_m, 9);
    enc.set_bytes(tiles_n, 10);

    enc.dispatch_threadgroups(MTL::Size::Make(grid_x, grid_y, 1),
                              MTL::Size::Make(threads, 1, 1));
  }
}

array w4a8_linear(const array &x, const array &packed_w, const array &scale_w,
                  const std::string &kernel_dir, StreamOrDevice s) {
  if (x.ndim() != 2) {
    throw std::invalid_argument("w4a8_linear: x must be 2D [M,K]");
  }
  if (packed_w.ndim() != 2) {
    throw std::invalid_argument("w4a8_linear: packed_w must be 2D [K/2,N]");
  }

  int M = x.shape(0);
  int N = packed_w.shape(1);
  auto stream = to_stream(s);

  return array(
      {M, N}, float16, std::make_shared<W4A8Linear>(stream, kernel_dir),
      {astype(x, float16, stream), astype(packed_w, mlx::core::uint8, stream),
       astype(scale_w, float32, stream)});
}

} // namespace cider
