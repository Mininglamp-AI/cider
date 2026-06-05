// sdpa_primitive.mm — Cider v9 SDPA Custom Primitive implementation
//
// Dispatches precompiled Metal kernels for optimized attention.
// Two primitives: SDPAVector (1-pass) and SDPAVector2Pass (2-pass).

#include "sdpa_primitive.h"
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

// ── Pipeline cache for SDPA kernels ─────────────────────────────
class SDPAPipelineCache {
public:
  static SDPAPipelineCache &instance() {
    static SDPAPipelineCache cache;
    return cache;
  }

  void ensure_init(const std::string &kernel_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_ && kernel_dir_ == kernel_dir)
      return;

    auto &dev = metal::device(mlx::core::Device::gpu);
    auto *mtl_device = dev.mtl_device();

    // Read and compile the SDPA metal source
    std::string source = read_file(kernel_dir + "/cider_sdpa_vector.metal");

    // Prepend the include path — the header is in the same directory
    // We inline the header by reading it and prepending
    std::string header = read_file(kernel_dir + "/cider_sdpa_vector.h");

    // Replace the #include directive with the actual header content
    std::string combined;
    combined.reserve(header.size() + source.size() + 256);
    // Add metal_stdlib include
    combined += "#include <metal_stdlib>\n";
    combined += "#include <metal_simdgroup>\n";
    combined += "using namespace metal;\n\n";
    combined += "// === Inlined cider_sdpa_vector.h ===\n";
    combined += header;
    combined += "\n// === End inlined header ===\n\n";

    // Append the .metal source but strip its own #include lines
    std::istringstream iss(source);
    std::string line;
    while (std::getline(iss, line)) {
      if (line.find("#include") != std::string::npos)
        continue;
      if (line.find("using namespace metal") != std::string::npos)
        continue;
      combined += line + "\n";
    }

    lib_ = compile_source(mtl_device, combined);
    kernel_dir_ = kernel_dir;
    pipelines_.clear();
    initialized_ = true;
  }

  MTL::ComputePipelineState *get(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pipelines_.find(name);
    if (it != pipelines_.end())
      return it->second;

    auto &dev = metal::device(mlx::core::Device::gpu);
    auto *pso = make_pipeline(dev.mtl_device(), name);
    pipelines_[name] = pso;
    return pso;
  }

private:
  SDPAPipelineCache() = default;
  bool initialized_ = false;
  std::string kernel_dir_;
  std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines_;
  MTL::Library *lib_ = nullptr;
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
                               const std::string &source) {
    @autoreleasepool {
      NSString *src = [NSString stringWithUTF8String:source.c_str()];
      MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
      opts.languageVersion = MTLLanguageVersion3_1;
      NSError *error = nil;
      id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
      id<MTLLibrary> lib_objc = [device_objc newLibraryWithSource:src
                                                          options:opts
                                                            error:&error];
      if (!lib_objc) {
        std::string err =
            error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("SDPA Metal compile failed: " + err);
      }
      return (__bridge MTL::Library *)(void *)CFBridgingRetain(lib_objc);
    }
  }

  MTL::ComputePipelineState *make_pipeline(MTL::Device *mtl_device,
                                           const std::string &name) {
    @autoreleasepool {
      NSString *fn_name = [NSString stringWithUTF8String:name.c_str()];
      id<MTLLibrary> lib_objc = (__bridge id<MTLLibrary>)lib_;
      id<MTLFunction> func = [lib_objc newFunctionWithName:fn_name];
      if (!func) {
        throw std::runtime_error("SDPA kernel not found: " + name);
      }
      NSError *error = nil;
      id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
      id<MTLComputePipelineState> pso =
          [device_objc newComputePipelineStateWithFunction:func error:&error];
      if (!pso) {
        std::string err =
            error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("SDPA pipeline failed for " + name + ": " +
                                 err);
      }
      return (__bridge MTL::ComputePipelineState *)(void *)CFBridgingRetain(
          pso);
    }
  }
};

// ── Helper: dtype string ────────────────────────────────────────
static std::string dtype_str(Dtype dt) {
  switch (dt) {
  case float32:
    return "float32";
  case float16:
    return "float16";
  case bfloat16:
    return "bfloat16";
  default:
    throw std::runtime_error("SDPA: unsupported dtype");
  }
}

// ═══════════════════════════════════════════════════════════════
// SDPAVector (1-pass) eval_gpu
// ═══════════════════════════════════════════════════════════════
void SDPAVector::eval_gpu(const std::vector<array> &inputs,
                          std::vector<array> &outputs) {
  auto &queries = inputs[0]; // [B*H, D]
  auto &keys = inputs[1];
  auto &values = inputs[2];
  auto &out = outputs[0];

  int D = queries.shape(-1);
  int total_heads = queries.shape(0);

  out.set_data(allocator::malloc(out.nbytes()));

  auto &cache = SDPAPipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  // Kernel name: cider_sdpa_vector_{type}_{D}
  std::string kname = "cider_sdpa_vector_" + dtype_str(queries.dtype()) + "_" +
                      std::to_string(D);
  auto *pso = cache.get(kname);

  auto &s = stream();
  auto &enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(pso);

  enc.set_input_array(queries, 0);
  enc.set_input_array(keys, 1);
  enc.set_input_array(values, 2);
  enc.set_output_array(out, 3);
  enc.set_bytes(gqa_factor_, 4);
  enc.set_bytes(N_, 5);
  enc.set_bytes(k_head_stride_, 6);
  enc.set_bytes(k_seq_stride_, 7);
  enc.set_bytes(v_head_stride_, 8);
  enc.set_bytes(v_seq_stride_, 9);
  enc.set_bytes(scale_, 10);

  // 1-pass: grid=(total_heads, 1, 1), group=(1024, 1, 1)
  MTL::Size grid_dims(total_heads, 1, 1);
  MTL::Size group_dims(1024, 1, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

// ═══════════════════════════════════════════════════════════════
// SDPAVector2Pass eval_gpu
// ═══════════════════════════════════════════════════════════════
void SDPAVector2Pass::eval_gpu(const std::vector<array> &inputs,
                               std::vector<array> &outputs) {
  auto &queries = inputs[0]; // [B, H, 1, D]
  auto &keys = inputs[1];    // [B, Hkv, N, D]
  auto &values = inputs[2];  // [B, Hkv, N, D]
  auto &out = outputs[0];    // [B, H, 1, D]

  int D = queries.shape(-1);
  int num_q_heads = num_kv_heads_ * gqa_factor_;
  int total_q_heads = batch_size_ * num_q_heads;

  out.set_data(allocator::malloc(out.nbytes()));

  auto &cache = SDPAPipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  auto &s = stream();
  auto &enc = metal::get_command_encoder(s);

  // ── Allocate intermediates ──
  int partial_total = total_q_heads * blocks_;
  size_t partial_bytes =
      static_cast<size_t>(partial_total) * D * size_of(queries.dtype());
  size_t stats_bytes = static_cast<size_t>(partial_total) * sizeof(float);

  array intermediate({partial_total, D}, queries.dtype(), nullptr, {});
  array sums({partial_total}, float32, nullptr, {});
  array maxs({partial_total}, float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  enc.add_temporary(intermediate);
  enc.add_temporary(sums);
  enc.add_temporary(maxs);

  // ── Pass 1 ──
  std::string kname1 = "cider_sdpa_vector_2pass_1_" +
                       dtype_str(queries.dtype()) + "_" + std::to_string(D) +
                       "_b" + std::to_string(blocks_);
  auto *pso1 = cache.get(kname1);
  enc.set_compute_pipeline_state(pso1);

  enc.set_input_array(queries, 0);
  enc.set_input_array(keys, 1);
  enc.set_input_array(values, 2);
  enc.set_output_array(intermediate, 3);
  enc.set_output_array(sums, 4);
  enc.set_output_array(maxs, 5);
  // buffer 6 is skipped (matching MLX convention)
  enc.set_bytes(N_, 7);
  enc.set_bytes(k_head_stride_, 8);
  enc.set_bytes(k_seq_stride_, 9);
  enc.set_bytes(v_head_stride_, 10);
  enc.set_bytes(v_seq_stride_, 11);
  enc.set_bytes(scale_, 12);

  // grid=(num_kv_heads, batch_size, blocks), group=(32, gqa_factor, 1)
  MTL::Size grid1(num_kv_heads_, batch_size_, blocks_);
  MTL::Size group1(32, gqa_factor_, 1);
  enc.dispatch_threadgroups(grid1, group1);

  // ── Pass 2 ──
  std::string kname2 = "cider_sdpa_vector_2pass_2_" +
                       dtype_str(queries.dtype()) + "_" + std::to_string(D) +
                       "_b" + std::to_string(blocks_);
  auto *pso2 = cache.get(kname2);
  enc.set_compute_pipeline_state(pso2);

  enc.set_input_array(intermediate, 0);
  enc.set_input_array(sums, 1);
  enc.set_input_array(maxs, 2);
  enc.set_output_array(out, 3);

  // grid=(total_q_heads, 1, 1), group=(1024, 1, 1)
  MTL::Size grid2(total_q_heads, 1, 1);
  MTL::Size group2(1024, 1, 1);
  enc.dispatch_threadgroups(grid2, group2);
}

// ═══════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════

array cider_sdpa_1pass(const array &queries, const array &keys,
                       const array &values, int gqa_factor, float scale,
                       const std::string &kernel_dir, StreamOrDevice s) {
  auto stream = to_stream(s);

  int total_heads = queries.shape(0);
  int D = queries.shape(-1);
  int N = keys.shape(-2);

  // Compute strides from key/value shapes
  // keys: contiguous [B*Hkv, N, D]
  size_t k_head_stride = static_cast<size_t>(N) * D;
  size_t k_seq_stride = D;
  size_t v_head_stride = k_head_stride;
  size_t v_seq_stride = D;

  return array({total_heads, D}, queries.dtype(),
               std::make_shared<SDPAVector>(stream, kernel_dir, gqa_factor, N,
                                            k_head_stride, k_seq_stride,
                                            v_head_stride, v_seq_stride, scale),
               {queries, keys, values});
}

array cider_sdpa_2pass(const array &queries, const array &keys,
                       const array &values, int gqa_factor, int blocks,
                       float scale, const std::string &kernel_dir,
                       StreamOrDevice s) {
  auto stream = to_stream(s);

  int B = queries.shape(0);
  int H = queries.shape(1);
  int D = queries.shape(-1);
  int Hkv = keys.shape(1);
  int N = keys.shape(2);

  // Strides for 4D layout [B, Hkv, N, D]
  size_t k_head_stride = static_cast<size_t>(N) * D;
  size_t k_seq_stride = D;
  size_t v_head_stride = k_head_stride;
  size_t v_seq_stride = D;

  auto out_shape = queries.shape();

  return array(out_shape, queries.dtype(),
               std::make_shared<SDPAVector2Pass>(
                   stream, kernel_dir, gqa_factor, N, blocks, Hkv, B,
                   k_head_stride, k_seq_stride, v_head_stride, v_seq_stride,
                   scale),
               {queries, keys, values});
}

} // namespace cider
