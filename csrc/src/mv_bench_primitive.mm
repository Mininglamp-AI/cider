#include "mv_bench_primitive.h"
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
  std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

// ── Kernel Cache (v2 + v3) ──────────────────────────────────
class MVBenchCache {
public:
  static MVBenchCache &instance() {
    static MVBenchCache c; return c;
  }

  MTL::ComputePipelineState *get(const std::string &name, const std::string &kdir,
                                  const std::string &filename) {
    std::lock_guard<std::mutex> lock(mu_);
    std::string key = filename + ":" + name;
    auto it = pso_.find(key);
    if (it != pso_.end()) return it->second;

    // Compile library for this file if not cached
    auto lib_it = libs_.find(filename);
    MTL::Library *lib;
    if (lib_it == libs_.end()) {
      auto &dev = metal::device(mlx::core::Device::gpu);
      std::string src = read_file(kdir + "/" + filename);
      @autoreleasepool {
        NSString *ns = [NSString stringWithUTF8String:src.c_str()];
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion4_0;
        NSError *err = nil;
        id<MTLDevice> d = (__bridge id<MTLDevice>)dev.mtl_device();
        id<MTLLibrary> mtl_lib = [d newLibraryWithSource:ns options:opts error:&err];
        if (!mtl_lib)
          throw std::runtime_error(std::string("Compile fail: ") +
                                   (err ? [[err localizedDescription] UTF8String] : "?"));
        lib = (__bridge MTL::Library*)(void*)CFBridgingRetain(mtl_lib);
      }
      libs_[filename] = lib;
    } else {
      lib = lib_it->second;
    }

    @autoreleasepool {
      NSString *fn = [NSString stringWithUTF8String:name.c_str()];
      id<MTLLibrary> objc_lib = (__bridge id<MTLLibrary>)lib;
      id<MTLFunction> func = [objc_lib newFunctionWithName:fn];
      if (!func) throw std::runtime_error("Kernel not found: " + name);
      NSError *err = nil;
      id<MTLDevice> d = (__bridge id<MTLDevice>)objc_lib.device;
      id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:func error:&err];
      if (!p) throw std::runtime_error("PSO fail: " + name);
      auto *r = (__bridge MTL::ComputePipelineState*)(void*)CFBridgingRetain(p);
      pso_[key] = r;
      return r;
    }
  }

private:
  MVBenchCache() = default;
  std::unordered_map<std::string, MTL::Library*> libs_;
  std::unordered_map<std::string, MTL::ComputePipelineState*> pso_;
  std::mutex mu_;
};

// ── Plan A (v2, non-tiled) ──────────────────────────────────
void MVPlanA::eval_gpu(const std::vector<array> &inputs, std::vector<array> &outputs) {
  auto &w = inputs[0];
  auto &x = inputs[3];
  auto &out = outputs[0];

  int32_t N_val = static_cast<int32_t>(w.shape(0));
  int32_t K_val = static_cast<int32_t>(w.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  std::string kname = (out.dtype() == bfloat16) ? "int8_mv_plan_a_bf16" : "int8_mv_plan_a_f16";
  auto *pso = MVBenchCache::instance().get(kname, kernel_dir_, "int8_mv_v2.metal");

  constexpr uint32_t SIMD_SIZE = 32;
  constexpr uint32_t NUM_SG = 2;
  constexpr uint32_t ROWS_PER_TG = NUM_SG * 4;
  uint32_t B = (x.ndim() >= 2) ? static_cast<uint32_t>(x.shape(0)) : 1;
  uint32_t grid_y = (static_cast<uint32_t>(N_val) + ROWS_PER_TG - 1) / ROWS_PER_TG;

  auto &enc = metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pso);
  enc.set_input_array(w, 0);
  enc.set_input_array(inputs[1], 1);  // scales
  enc.set_input_array(inputs[2], 2);  // biases
  enc.set_input_array(x, 3);
  enc.set_output_array(out, 4);
  enc.set_bytes(K_val, 5);
  enc.set_bytes(N_val, 6);
  enc.dispatch_threadgroups(MTL::Size::Make(B, grid_y, 1),
                            MTL::Size::Make(SIMD_SIZE * NUM_SG, 1, 1));
}

// ── Plan B (v2, non-tiled) ──────────────────────────────────
void MVPlanB::eval_gpu(const std::vector<array> &inputs, std::vector<array> &outputs) {
  auto &w = inputs[0];
  auto &x = inputs[3];
  auto &out = outputs[0];

  int32_t N_val = static_cast<int32_t>(w.shape(0));
  int32_t K_val = static_cast<int32_t>(w.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  std::string kname = (out.dtype() == bfloat16) ? "int8_mv_plan_b_bf16" : "int8_mv_plan_b_f16";
  auto *pso = MVBenchCache::instance().get(kname, kernel_dir_, "int8_mv_v2.metal");

  constexpr uint32_t SIMD_SIZE = 32;
  constexpr uint32_t NUM_SG = 2;
  constexpr uint32_t ROWS_PER_TG = NUM_SG * 4;
  uint32_t B = (x.ndim() >= 2) ? static_cast<uint32_t>(x.shape(0)) : 1;
  uint32_t grid_y = (static_cast<uint32_t>(N_val) + ROWS_PER_TG - 1) / ROWS_PER_TG;

  auto &enc = metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pso);
  enc.set_input_array(w, 0);
  enc.set_input_array(inputs[1], 1);
  enc.set_input_array(inputs[2], 2);
  enc.set_input_array(x, 3);
  enc.set_output_array(out, 4);
  enc.set_bytes(K_val, 5);
  enc.set_bytes(N_val, 6);
  enc.dispatch_threadgroups(MTL::Size::Make(B, grid_y, 1),
                            MTL::Size::Make(SIMD_SIZE * NUM_SG, 1, 1));
}

// ── Plan A Tiled (v3, K-tile swizzle) ───────────────────────
void MVPlanATiled::eval_gpu(const std::vector<array> &inputs, std::vector<array> &outputs) {
  auto &w = inputs[0];
  auto &x = inputs[3];
  auto &out = outputs[0];  // float32 for atomic

  int32_t N_val = static_cast<int32_t>(w.shape(0));
  int32_t K_val = static_cast<int32_t>(w.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));
  // Zero the output buffer (multiple K-tiles do atomic add)
  memset(out.data<void>(), 0, out.nbytes());

  std::string kname = (x.dtype() == bfloat16) ? "int8_mv_tiled_bf16" : "int8_mv_tiled_f16";
  auto *pso = MVBenchCache::instance().get(kname, kernel_dir_, "int8_mv_v3.metal");

  constexpr uint32_t SIMD_SIZE = 32;
  constexpr uint32_t NUM_SG = 4;
  constexpr uint32_t ROWS_PER_TG = NUM_SG * 4;  // 16
  constexpr uint32_t K_TILE = 4096;
  uint32_t B = (x.ndim() >= 2) ? static_cast<uint32_t>(x.shape(0)) : 1;
  uint32_t grid_y = (static_cast<uint32_t>(N_val) + ROWS_PER_TG - 1) / ROWS_PER_TG;
  uint32_t grid_z = (static_cast<uint32_t>(K_val) + K_TILE - 1) / K_TILE;

  auto &enc = metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pso);
  enc.set_input_array(w, 0);
  enc.set_input_array(inputs[1], 1);
  enc.set_input_array(inputs[2], 2);
  enc.set_input_array(x, 3);
  enc.set_output_array(out, 4);
  enc.set_bytes(K_val, 5);
  enc.set_bytes(N_val, 6);
  enc.dispatch_threadgroups(MTL::Size::Make(B, grid_y, grid_z),
                            MTL::Size::Make(SIMD_SIZE * NUM_SG, 1, 1));
}

// ── Plan A Direct (v3, non-tiled, NUM_SG=4) ────────────────
void MVPlanADirect::eval_gpu(const std::vector<array> &inputs, std::vector<array> &outputs) {
  auto &w = inputs[0];
  auto &x = inputs[3];
  auto &out = outputs[0];

  int32_t N_val = static_cast<int32_t>(w.shape(0));
  int32_t K_val = static_cast<int32_t>(w.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  std::string kname = (out.dtype() == bfloat16) ? "int8_mv_direct_bf16" : "int8_mv_direct_f16";
  auto *pso = MVBenchCache::instance().get(kname, kernel_dir_, "int8_mv_v3.metal");

  constexpr uint32_t SIMD_SIZE = 32;
  constexpr uint32_t NUM_SG = 4;
  constexpr uint32_t ROWS_PER_TG = NUM_SG * 4;  // 16
  uint32_t B = (x.ndim() >= 2) ? static_cast<uint32_t>(x.shape(0)) : 1;
  uint32_t grid_y = (static_cast<uint32_t>(N_val) + ROWS_PER_TG - 1) / ROWS_PER_TG;

  auto &enc = metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pso);
  enc.set_input_array(w, 0);
  enc.set_input_array(inputs[1], 1);
  enc.set_input_array(inputs[2], 2);
  enc.set_input_array(x, 3);
  enc.set_output_array(out, 4);
  enc.set_bytes(K_val, 5);
  enc.set_bytes(N_val, 6);
  enc.dispatch_threadgroups(MTL::Size::Make(B, grid_y, 1),
                            MTL::Size::Make(SIMD_SIZE * NUM_SG, 1, 1));
}

// ── Python-facing factory functions ─────────────────────────

array mv_plan_a(const array &w, const array &scales, const array &biases,
                const array &x, const std::string &kernel_dir, StreamOrDevice s) {
  int B = (x.ndim() >= 2) ? x.shape(0) : 1;
  int N = w.shape(0);
  auto st = to_stream(s);
  auto dtype = x.dtype();
  if (dtype != float16 && dtype != bfloat16) dtype = float16;
  return array({B, N}, dtype, std::make_shared<MVPlanA>(st, kernel_dir),
               {astype(w, int8, st), astype(scales, float32, st),
                astype(biases, float32, st), astype(x, dtype, st)});
}

array mv_plan_b(const array &w, const array &scales, const array &biases,
                const array &x, const std::string &kernel_dir, StreamOrDevice s) {
  int B = (x.ndim() >= 2) ? x.shape(0) : 1;
  int N = w.shape(0);
  auto st = to_stream(s);
  auto dtype = x.dtype();
  if (dtype != float16 && dtype != bfloat16) dtype = float16;
  return array({B, N}, dtype, std::make_shared<MVPlanB>(st, kernel_dir),
               {astype(w, int8, st), astype(scales, float32, st),
                astype(biases, float32, st), astype(x, dtype, st)});
}

array mv_plan_a_tiled(const array &w, const array &scales, const array &biases,
                      const array &x, const std::string &kernel_dir, StreamOrDevice s) {
  int B = (x.ndim() >= 2) ? x.shape(0) : 1;
  int N = w.shape(0);
  auto st = to_stream(s);
  auto in_dtype = x.dtype();
  if (in_dtype != float16 && in_dtype != bfloat16) in_dtype = float16;
  // Output is float32 for atomic accumulation
  return array({B, N}, float32, std::make_shared<MVPlanATiled>(st, kernel_dir),
               {astype(w, int8, st), astype(scales, float32, st),
                astype(biases, float32, st), astype(x, in_dtype, st)});
}

array mv_plan_a_direct(const array &w, const array &scales, const array &biases,
                       const array &x, const std::string &kernel_dir, StreamOrDevice s) {
  int B = (x.ndim() >= 2) ? x.shape(0) : 1;
  int N = w.shape(0);
  auto st = to_stream(s);
  auto dtype = x.dtype();
  if (dtype != float16 && dtype != bfloat16) dtype = float16;
  return array({B, N}, dtype, std::make_shared<MVPlanADirect>(st, kernel_dir),
               {astype(w, int8, st), astype(scales, float32, st),
                astype(biases, float32, st), astype(x, dtype, st)});
}

// ── Plan A V4 (vectorized loads, NUM_SG=4, VPT=16) ──────
void MVPlanAV4::eval_gpu(const std::vector<array> &inputs, std::vector<array> &outputs) {
  auto &w = inputs[0];
  auto &x = inputs[3];
  auto &out = outputs[0];

  int32_t N_val = static_cast<int32_t>(w.shape(0));
  int32_t K_val = static_cast<int32_t>(w.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  std::string kname = (out.dtype() == bfloat16) ? "int8_mv_v4_bf16" : "int8_mv_v4_f16";
  auto *pso = MVBenchCache::instance().get(kname, kernel_dir_, "int8_mv_v4.metal");

  constexpr uint32_t SIMD_SIZE = 32;
  constexpr uint32_t NUM_SG = 4;
  constexpr uint32_t ROWS_PER_TG = NUM_SG * 4;  // 16
  uint32_t B = (x.ndim() >= 2) ? static_cast<uint32_t>(x.shape(0)) : 1;
  uint32_t grid_y = (static_cast<uint32_t>(N_val) + ROWS_PER_TG - 1) / ROWS_PER_TG;

  auto &enc = metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pso);
  enc.set_input_array(w, 0);
  enc.set_input_array(inputs[1], 1);
  enc.set_input_array(inputs[2], 2);
  enc.set_input_array(x, 3);
  enc.set_output_array(out, 4);
  enc.set_bytes(K_val, 5);
  enc.set_bytes(N_val, 6);
  enc.dispatch_threadgroups(MTL::Size::Make(B, grid_y, 1),
                            MTL::Size::Make(SIMD_SIZE * NUM_SG, 1, 1));
}

array mv_plan_a_v4(const array &w, const array &scales, const array &biases,
                   const array &x, const std::string &kernel_dir, StreamOrDevice s) {
  int B = (x.ndim() >= 2) ? x.shape(0) : 1;
  int N = w.shape(0);
  auto st = to_stream(s);
  auto dtype = x.dtype();
  if (dtype != float16 && dtype != bfloat16) dtype = float16;
  return array({B, N}, dtype, std::make_shared<MVPlanAV4>(st, kernel_dir),
               {astype(w, int8, st), astype(scales, float32, st),
                astype(biases, float32, st), astype(x, dtype, st)});
}

} // namespace w8a8_mlx
