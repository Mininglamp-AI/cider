// int8_mv_primitive.mm — Per-group quantized matrix-vector multiply
// Decode path: on-the-fly dequant from packed uint32 weights
// y[B,N] = sum_g [ scale[n][g]*dot(x_g, w_uint8_g) + bias[n][g]*sum(x_g) ]

#include "int8_mv_primitive.h"
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

static std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

class MVPipelineCache {
public:
  static MVPipelineCache &instance() {
    static MVPipelineCache cache;
    return cache;
  }

  MTL::ComputePipelineState *get(const std::string &name,
                                  const std::string &kernel_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || kernel_dir_ != kernel_dir) {
      auto &dev = metal::device(mlx::core::Device::gpu);
      auto *mtl_device = dev.mtl_device();
      std::string src = read_file(kernel_dir + "/int8_mv.metal");
      @autoreleasepool {
        NSString *ns_src = [NSString stringWithUTF8String:src.c_str()];
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion4_0;
        NSError *error = nil;
        id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
        id<MTLLibrary> lib_objc = [device_objc newLibraryWithSource:ns_src
                                                            options:opts
                                                              error:&error];
        if (!lib_objc) {
          std::string err = error ? [[error localizedDescription] UTF8String] : "Unknown";
          throw std::runtime_error("Metal compile int8_mv failed: " + err);
        }
        lib_ = (__bridge MTL::Library *)(void *)CFBridgingRetain(lib_objc);
      }
      kernel_dir_ = kernel_dir;
      pipelines_.clear();
      initialized_ = true;
    }

    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) return it->second;

    @autoreleasepool {
      NSString *fn_name = [NSString stringWithUTF8String:name.c_str()];
      id<MTLLibrary> lib_objc = (__bridge id<MTLLibrary>)lib_;
      id<MTLFunction> func = [lib_objc newFunctionWithName:fn_name];
      if (!func) throw std::runtime_error("Kernel not found: " + name);
      NSError *error = nil;
      id<MTLDevice> device_objc = (__bridge id<MTLDevice>)lib_objc.device;
      id<MTLComputePipelineState> pso =
          [device_objc newComputePipelineStateWithFunction:func error:&error];
      if (!pso) {
        std::string err = error ? [[error localizedDescription] UTF8String] : "Unknown";
        throw std::runtime_error("Pipeline failed: " + name + ": " + err);
      }
      auto *result = (__bridge MTL::ComputePipelineState *)(void *)CFBridgingRetain(pso);
      pipelines_[name] = result;
      return result;
    }
  }

private:
  MVPipelineCache() = default;
  bool initialized_ = false;
  std::string kernel_dir_;
  MTL::Library *lib_ = nullptr;
  std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines_;
  std::mutex mutex_;
};

void Int8MV::eval_gpu(const std::vector<array> &inputs,
                      std::vector<array> &outputs) {
  auto &w_packed = inputs[0];  // [N, K/4] uint32
  auto &scales = inputs[1];    // [N, n_groups] float32
  auto &biases = inputs[2];    // [N, n_groups] float32
  auto &x = inputs[3];         // [B, K] float16/bfloat16/float32

  auto &out = outputs[0];      // [B, N]

  uint32_t B = (x.ndim() >= 2) ? static_cast<uint32_t>(x.shape(0)) : 1;
  uint32_t N = static_cast<uint32_t>(w_packed.shape(0));
  int32_t K = static_cast<int32_t>(w_packed.shape(1)) * 4;  // unpack

  out.set_data(allocator::malloc(out.nbytes()));

  std::string kname;
  if (out.dtype() == float16) {
    kname = "int8_mv_float16";
  } else if (out.dtype() == bfloat16) {
    kname = "int8_mv_bfloat16";
  } else {
    kname = "int8_mv_float32";
  }

  auto *pso = MVPipelineCache::instance().get(kname, kernel_dir_);

  constexpr uint32_t num_simdgroups = 2;
  constexpr uint32_t results_per_sg = 4;
  constexpr uint32_t SIMD_SIZE = 32;
  uint32_t rows_per_tg = num_simdgroups * results_per_sg;  // 8
  uint32_t grid_y = (N + rows_per_tg - 1) / rows_per_tg;

  int32_t in_vec_size = K;
  int32_t out_vec_size = static_cast<int32_t>(N);

  auto &s = stream();
  auto &enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(pso);
  enc.set_input_array(w_packed, 0);   // [N, K/4] uint32
  enc.set_input_array(scales, 1);     // [N, n_groups] float32
  enc.set_input_array(biases, 2);     // [N, n_groups] float32
  enc.set_input_array(x, 3);          // [B, K]
  enc.set_output_array(out, 4);       // [B, N]
  enc.set_bytes(in_vec_size, 5);
  enc.set_bytes(out_vec_size, 6);
  enc.dispatch_threadgroups(
      MTL::Size::Make(B, grid_y, 1),
      MTL::Size::Make(SIMD_SIZE * num_simdgroups, 1, 1));
}

array int8_mv(const array &w_packed, const array &scales, const array &biases,
              const array &x, const std::string &kernel_dir, StreamOrDevice s) {
  if (w_packed.ndim() != 2) throw std::invalid_argument("int8_mv: w_packed must be 2D [N,K/4]");
  if (scales.ndim() != 2) throw std::invalid_argument("int8_mv: scales must be 2D [N,n_groups]");
  if (x.ndim() < 1) throw std::invalid_argument("int8_mv: x must be at least 1D");

  int B = (x.ndim() >= 2) ? x.shape(0) : 1;
  int N = w_packed.shape(0);
  auto stream = to_stream(s);

  auto out_dtype = x.dtype();
  if (out_dtype != float16 && out_dtype != bfloat16 && out_dtype != float32) {
    out_dtype = float16;
  }

  return array(
      {B, N}, out_dtype,
      std::make_shared<Int8MV>(stream, kernel_dir),
      {astype(w_packed, mlx::core::uint32, stream),
       astype(scales, float32, stream),
       astype(biases, float32, stream),
       astype(x, out_dtype, stream)});
}

} // namespace w8a8_mlx
