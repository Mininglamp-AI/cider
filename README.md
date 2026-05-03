# cider

Cider is developed on top of MLX for macOS. It provides online activation quantization operators absent in MLX, with custom int-matmul kernels built as MLX custom primitives supporting full lazy evaluation. It also includes service-side extensions and non-intrusive compatibility patches for mlx_vlm (validated on mlx_vlm 0.4.3), including fixes for Qwen3-VL multi-image inference issues related to RoPE position handling and chunked prefill. 

## Conditional Compilation (M4 / M5)

Cider uses **conditional compilation**: the INT8 TensorOps C++ extension is only built on Apple M5+.

| Chip | `pip install -e .` behavior | `import cider` behavior |
|------|----------------------------|------------------------|
| **M5+** | Full build (CMake + Metal kernels) | All features available |
| **M4 and below** | Skips C++ build, installs pure-Python package | `is_available()` → False, `convert_model()` is a warning no-op |

**Override via environment variable:**
```bash
CIDER_FORCE_BUILD=1 pip install -e .   # Force build (e.g., CI)
CIDER_FORCE_BUILD=0 pip install -e .   # Force skip
```

## Modes

| Mode | Weights | Activations | Compute Path | Status |
|------|---------|-------------|--------------|--------|
| **W8A8** | INT8 symmetric | INT8 per-token | TensorOps matmul2d | ✅ Implemented |
| **W4A8** | INT4 packed (uint8) | INT8 per-token | Unpack→TensorOps | ✅ Implemented |
| W4A16 | — | — | MLX built-in | Baseline |
| W8A16 | — | — | MLX built-in | Baseline |

**W4A16 and W8A16 are already supported by MLX natively** — this SDK provides the missing W8A8 and W4A8 modes that MLX does not implement.

MLX's quantization is **weight-only**: QuantizedLinear dequantizes weights to FP16 and uses FP16 GEMM. While MLX's Steel NAX templates are generic enough to be instantiated with INT8 types (and would achieve identical raw matmul throughput — [see our transparent benchmark](benchmarks/mlx_native/cider_vs_mlx_int8.md)), MLX does not provide the quantization/dequantization pipeline needed for actual W8A8 inference. Cider fills this gap with fused quantize-matmul-dequant primitives.

This SDK implements online INT8 activation quantization and INT8 TensorOps-based compute for the supported inference paths. 

### W8A8 Quantization Granularity

| Granularity | Description | Speed | Precision |
|-------------|-------------|-------|-----------|
| **Per-channel** | One scale per output channel | Fastest (1.8x prefill) | Slightly lower |
| **Per-group (gs=128)** | One scale per 128 elements | Fast (1.5x prefill) | Moderate precision retention |
| **Per-group (gs=64)** | One scale per 64 elements | Moderate (1.3x prefill) | Higher precision |

## Performance (Apple M5 Pro)

### Individual Operator Latency 

Shape [N=10240, K=2560]
| M |   PC(ms) |  PG(ms)  |  w8a16  |  w4a16 |   PC/w8  | PC/w4  | PG/w8  | PG/w4|
|-----|------|------|-----|-----|-----|------|-----|----|
| 1 |   0.27ms |   0.26ms |  0.26ms |  0.18ms |  0.96x | 0.67x | 0.99x | 0.69x |
|128 |   0.34ms  | 0.39ms |  0.49ms |  0.44ms |  1.43x | 1.28x  | 1.26x |  1.13x |
|1024 |   1.23ms |  1.52ms  | 2.24ms  | 2.04ms |  1.82x  | 1.66x | 1.47x | 1.34x|
|4096 |   4.41ms |  5.65ms |  8.12ms |  7.72ms |  1.84x |  1.75x | 1.44x  | 1.37x |
|8192 |   8.71ms |  11.40ms |  16.23ms | 15.09ms |  1.86x | 1.73x | 1.42x | 1.32x|


Shape [N=2560, K=10240]
| M |   PC(ms)  | PG(ms)   | w8a16  |  w4a16 |   PC/w8  | PC/w4  | PG/w8  | PG/w4 |
|--------|------|--------|-------| ---|--------|------|-------------|------------------|
| 1 |   0.25ms |  0.26ms |  0.26ms  | 0.20ms |  1.03x | 0.78x | 0.98x | 0.75x |
|128 |   0.39ms |  0.41ms |  0.55ms |  0.46ms |  1.43x | 1.19x | 1.35x | 1.12x |
| 1024 |   1.31ms |  1.65ms |  2.35ms  | 2.14ms |  1.80x  | 1.64x | 1.43x | 1.30x |
| 4096 |   5.37ms  | 6.79ms  | 8.54ms |  8.04ms |  1.59x | 1.50x | 1.26x | 1.18x |
| 8192 |  10.97ms | 12.94ms | 17.28ms | 16.23ms |  1.58x | 1.48x | 1.34x | 1.25x | 

### End-to-End VLM 

**Qwen3-VL-2B**

| Prompt Tokens | FP16 Prefill (tok/s) | W8A16 Prefill (tok/s) | **W8A8 PC Prefill (tok/s)** | FP16 Decode (tok/s) | W8A16 Decode (tok/s) | **W8A8 PC Decode (tok/s)** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1334 | 3010 | 2065 | **3242** | 70 | 107 | **104** |
| 2393 | 2868 | 1847 | **2983** | 69 | 97 | **100** |
| 3455 | 2777 | 1741 | **2796** | 66 | 90 | **95** |

**Qwen3-VL-4B**

 Prompt Tokens | FP16 Prefill (tok/s) | W8A16 Prefill (tok/s) | **W8A8 PC Prefill (tok/s)** | FP16 Decode (tok/s) | W8A16 Decode (tok/s) | **W8A8 PC Decode (tok/s)** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1334 | 1884 | 1786 | **2186** | 32 | **56** | 54 |
| 2393 | 1815 | 1700 | **2028** | 31 | **55** | 52 |
| 3455 | 1755 | 1603 | **1881** | 30 | **52** | 49 |


### LLM Quantization: Precision vs. Speed Comparison


<table>
  <thead>
    <tr>
      <th>Models</th>
      <th>Quantization Configuration</th>
      <th>wikitext2 PPL（↓）</th>
      <th>Prefill Time (s)（↓）</th>
      <th>Peak Memory (GB)（↓）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"><b>Qwen3-8B</b></td>
      <td>FP16</td>
      <td>9.726</td>
      <td>179.9</td>
      <td>18.93</td>
    </tr>
    <tr>
      <td>W8A16 (mlx RTN)</td>
      <td>9.707</td>
      <td>221.3</td>
      <td>12.07</td>
    </tr>
    <tr>
      <td>W8A8 (per-channel)</td>
      <td>9.756</td>
      <td><b>123.5</b></td>
      <td><b>11.32</b></td>
    </tr>
    <tr>
      <td>W8A8 (per-group gs=64)</td>
      <td>9.744</td>
      <td>179.1</td>
      <td>11.83</td>
    </tr>
    <tr>
      <td>W8A8 (per-group gs=128)</td>
      <td>9.727</td>
      <td>165.8</td>
      <td>11.61</td>
    </tr>
  </tbody>
  <tr style="border-top: 1px solid #333;">
      <td colspan="5" style="padding: 0; height: 3px;"></td>
  </tr>
  <tbody>
    <tr>
      <td rowspan="5"><b>Llama3-8B</b></td>
      <td>FP16</td>
      <td>6.138</td>
      <td>175.8</td>
      <td>18.32</td>
    </tr>
    <tr>
      <td>W8A16 (mlx RTN)</td>
      <td>6.147</td>
      <td>236.9</td>
      <td>11.46</td>
    </tr>
    <tr>
      <td>W8A8 (per-channel)</td>
      <td>6.271</td>
      <td><b>123.3</b></td>
      <td><b>10.69</b></td>
    </tr>
    <tr>
      <td>W8A8 (per-group, gs=64)</td>
      <td>6.269</td>
      <td>178.7</td>
      <td>11.19</td>
    </tr>
    <tr>
      <td>W8A8 (per-group, gs=128)</td>
      <td>6.270</td>
      <td>155.7</td>
      <td>10.98</td>
    </tr>
  </tbody>
</table>

## Requirements

- **Apple M5+** for INT8 TensorOps (M4 and below: installs as pure-Python, `is_available()` returns False)
- Python 3.12+
- MLX >= 0.31
- nanobind >= 2.12 (only needed on M5+ for C++ build)
- CMake >= 3.27 (only needed on M5+ for C++ build)

## Install

```bash
pip install -e .
```

On M5+, this runs CMake to compile the C++ extension, then installs the Python package.
On M4 and below, only the Python package is installed (no compilation errors).

## Quick Start

### One-line Model Conversion (Recommended)

```python
from cider import convert_model, is_available

model, proc = load("path/to/model")

if is_available():
    convert_model(model)
    # CiderLinear auto-detects:
    #   seq_len > 1  → W8A8 INT8 TensorOps (faster prefill)
    #   seq_len == 1 → INT8 MV kernel (near-native decode speed)
else:
    pass  # Falls back to standard MLX inference on M4
```

**Important**
When quantizing Vision-Language Models (VLMs), the vision transformer (ViT) is generally not replaced. Directly using convert_model will quantize the vision model's linear layers as well, which typically causes accuracy drop. For VLMs, we recommend calling convert_model(model.language_model) to apply existing quantization methods like GPTQ, SmoothQuant, and AWQ to the language model only.

Tested on selected MLX transformer models, including Qwen3, Qwen3-VL and Llama3 families. Other architectures may require adaptation.


### Layer-level API

```python
import numpy as np
import mlx.core as mx
from cider import W8A8Linear, W4A8Linear, is_available

assert is_available(), "Requires Apple M5+"

# Prepare weight
W = np.random.randn(4096, 4096).astype(np.float16)

# W8A8 linear (per-channel)
from cider.ops import quantize_weight_int8
w_int8, scale = quantize_weight_int8(W)
layer = W8A8Linear(
    w_int8=mx.array(w_int8), scale_w=mx.array(scale),
    group_size=0, in_features=4096, out_features=4096
)
x = mx.random.normal((32, 4096)).astype(mx.float16)
y = layer(x)    # lazy — builds MLX graph
mx.eval(y)       # GPU executes

# W4A8 linear (half the weight memory)
layer4 = W4A8Linear.from_weights(W)
y4 = layer4(x)
mx.eval(y4)
```

## Low-Level API

```python
from cider import perchannel_linear, w4a8_linear, quantize_weight_int8, pack_weight_int4

# Quantize weights (numpy, offline)
w_int8, scale = quantize_weight_int8(W_np)
packed_w4, scale4 = pack_weight_int4(W_np)

# Primitive calls (return lazy mx.array)
y = perchannel_linear(x, mx.array(w_int8), mx.array(scale))
y4 = w4a8_linear(x, mx.array(packed_w4), mx.array(scale4))
```

## Project Structure

```
cider/
├── cider/              # Python package
│   ├── __init__.py        # Public API (conditional on is_available)
│   ├── ops.py             # Primitive wrappers + quantize helpers
│   ├── nn.py              # CiderLinear, W4A8Linear (nn.Module)
│   ├── convert.py         # convert_model() high-level API
│   └── kernels/           # Metal shaders (bundled)
│       ├── w8a8_matmul.metal       # W8A8 GEMM (prefill, M>1)
│       ├── w8a8_int8_mv.metal      # W8A8 per-channel MV (decode, M=1)
│       ├── w8a8_quantize.metal     # Per-token activation quantization
│       ├── w4a8_matmul.metal       # W4A8 GEMM (prefill)
│       ├── pergroup_int8_gemm.metal # Per-group GEMM (prefill)
│       └── pergroup_int8_mv.metal   # Per-group MV (decode)
├── csrc/                  # C++ MLX primitives (nanobind, M5+ only)
│   ├── include/
│   │   ├── w8a8_primitive.h
│   │   ├── w4a8_primitive.h
│   │   └── pergroup_primitive.h
│   └── src/
│       ├── w8a8_primitive.mm
│       ├── w4a8_primitive.mm
│       ├── pergroup_primitive.mm
│       └── prim_bindings.cpp
├── benchmarks/
│   ├── bench_e2e_wxa16.py    # End-to-end VLM benchmark (Qwen3-VL-2B)
│   ├── bench_full.py         # Isolated kernel latency (per-channel/per-group vs MLX)
│   ├── test_bitexact.py      # Numerical correctness verification
│   └── mlx_native/           # MLX native INT8 comparison
├── tutorial/
│   ├── how_to_write_efficient_int_gemm_m5_en.md
│   └── how_to_write_efficient_int_gemm_m5_zh.md
├── tools/
│   ├── eval_ppl_all.py               # Unified PPL eval (FP16/W8A16/per-channel/per-group)
│   ├── convert_compressed_tensors_to_mlx.py
│   └── smoothquant.py                # SmoothQuant calibration
├── examples/
│   └── basic_usage.py
├── vlm_service/           # OpenAI-style VLM inference server
│   ├── server.py             # FastAPI server (streaming + non-streaming)
│   ├── core_infer.py         # HMInference engine (singleton)
│   ├── custom_qwen3vl.py     # Custom Qwen3-VL generation loop
│   ├── config.py             # Config loader
│   ├── bench_client.py       # Server benchmark client
│   └── client.py             # API client example
├── config/
│   └── config.yaml           # Server & model configuration
├── experimental/             # ANE+GPU hybrid tensor parallelism (M4)
│   ├── split_linear.py       # SplitLinear + ANEBridge + patch_model()
│   ├── bench.py              # End-to-end benchmark
│   ├── libane_bridge_v6.m    # ANE private API bridge (Obj-C source)
│   └── README.md
├── CMakeLists.txt
├── pyproject.toml
├── setup.py               # Conditional build (M5+: full, M4: pure-Python)
└── README.md
```

## VLM Inference Service

`vlm_service/` provides a ready-to-use **OpenAI-style** VLM inference server with W8A8 acceleration.

### Quick Start

1. **Configure** `config/config.yaml`:

```yaml
model_name_or_path: /path/to/your/model   # MLX VLM model (e.g., Qwen3-VL-2B W8A16)
sampling:
  max_new_tokens: 1024
  temperature: 1.0
  top_p: 1.0
server:
  host: 0.0.0.0
  port: 8341
  ttl: 1800
w8a8:
  mode: 'off'   # 'auto' | 'on' | 'off'
```

- `auto`: Enable W8A8 if hardware supports it, fallback to default otherwise
- `on`: Force W8A8 (error if unsupported). "When 'on' is selected, it means your model needs to perform online activation quantization. In this case, Cider itself does **not** guarantee quantization accuracy, and you need to apply some quantization algorithms yourself, such as SmoothQuant, QuaRot, GPTQ, or even QAT, to ensure that the accuracy does not degrade significantly after activation quantization. This option simply provides a way for you to leverage the hardware's computational advantages when your model applies W8A8, rather than just simulating quantization."
- `off`: Disable W8A8, use standard MLX inference

2. **Start the server**:

```bash
cd vlm_service
python server.py --config ../config/config.yaml
```

3. **Send requests** (OpenAI-style API):

```bash
# Text-only
curl http://localhost:8341/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vlm",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'

# With image (base64)
curl http://localhost:8341/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vlm",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."} },
      {"type": "text", "text": "What is in this image?"}
    ]}],
    "stream": true
  }'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (stream / non-stream) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/v1/queue` | GET | Request queue status |

### How W8A8 Works in the Service

When `w8a8.mode` is `auto` or `on`, the server calls `cider.convert_model()` at startup to replace all Linear layers with `CiderLinear`. During inference:

- **Prefill** (processing input tokens, seq_len > 1): Uses W8A8 INT8 GEMM for faster computation
- **Decode** (generating tokens one by one, seq_len == 1): Uses INT8 MV kernel (near-native speed)

No code changes needed — the switching is automatic based on input sequence length.

## Architecture

### MLX Custom Primitives

Both W8A8Linear and W4A8Linear are implemented as `mlx::core::Primitive` subclasses. This means:

1. **Lazy evaluation**: `y = layer(x)` builds a graph node, not immediate computation
2. **Graph composition**: Multiple primitive calls compose into a single MLX graph
3. **Stream scheduling**: MLX's scheduler handles GPU dispatch order

### Metal Kernel Pipeline

Each primitive dispatches Metal compute kernels:

**Prefill (M > 1):**
1. **quantize_per_token**: FP16 activations → INT8 + per-token scales
2. **matmul_fused_dequant**: INT8 × INT8 → INT32 → FP16 (with fused scale dequantization)

**Decode (M = 1):**
- **int8_mv**: Direct INT8 matrix-vector product with on-the-fly weight dequantization (no activation quantization needed)

For W4A8, the GEMM step includes inline INT4→INT8 unpacking in the fragment load.

### TensorOps matmul2d

The INT8 GEMM uses Apple's `mpp::tensor_ops::matmul2d(16, 32, 16)` — hardware-accelerated INT8×INT8→INT32 matrix multiply available on M5+ via Metal 4's `cooperative_tensor` API. This is the same hardware instruction available to MLX's NAX templates. Cider's kernel adds fused dequantization (INT32 × scales → FP16) in the store phase, avoiding an extra device memory round-trip. See [kernel comparison](benchmarks/mlx_native/cider_vs_mlx_int8.md) for details.

### Tile Configurations

| Config | BM | BN | BK | SK | Threads | Use When |
|--------|----|----|----|----|---------|----------|
| Large  | 128 | 128 | 512 | 32 | 512 | M > 64 |
| Small  | 32  | 128 | 512 | 32 | 128 | M ≤ 64 |

Auto-selected based on M. L2 cache swizzle dispatch included.

## ANE+GPU Heterogeneous Tensor Parallelism (experimental)

We found that during inference on Mac, only two hardware computing units—GPU and CPU—were utilized, while the ANE (Apple Neural Engine) computing unit on Mac remained idle. We identified this as a potential optimization opportunity. Inspired by [maderix/ANE](https://github.com/maderix/ANE), we conducted experimental work on a hybrid ANE+GPU inference mode. Currently, we apply this approach to tensor parallel computing. On the M4 chip, during synchronous-only forward inference (MLX natively uses a technique called lazy evaluation, which reduces synchronization overhead; in end-to-end testing, the hybrid inference currently shows no advantage, mainly because we have not yet implemented this using MLX's lazy evaluation—this remains future work), we observed approximately **3%~16%** performance improvement compared to pure GPU inference under synchronize pipeline. This remains exploratory work, and end-to-end gains are currently limited by the lack of a lazy-evaluation-compatible implementation.


During LLM prefill, the GPU's matrix units are fully occupied — but the **Apple Neural Engine sits completely idle**. ANE Split exploits this by splitting each linear layer's GEMM along output channels:

- **ANE** computes ~65% of output channels (FP32, via reverse-engineered private `_ANEClient` API)
- **GPU** computes the remaining ~35% (FP16, standard MLX matmul)
- Both run **concurrently**, and results are concatenated

This is a form of **heterogeneous tensor parallelism** — not data parallelism, not pipeline parallelism — exploiting two distinct compute units on the same SoC.

### Performance (Apple M4, Qwen3-VL-2B Prefill)

| seq | W8A16 GPU | SplitLinear | Speedup vs W8A16 |
|-----|----------|-----------|-------------|
| 512 | 639.9 ms | **615.9 ms** | **1.039×** |
| 1024 | 1348.6 ms | **1156.9 ms** | **1.17×** |

In the tested benchmark cases, cosine similarity was close to 1.0 and top-1 token agreement was 100%.


### Key Design Choices

- **Prefill only**: Decode falls back to original GPU linear (zero overhead)
- **Shared input preparation**: Q/K/V and Gate/Up projections share a single input transpose+numpy copy via `_InputGroup`
- **Auto-routing**: Down projections (IC > 2×OC) stay GPU-only where ANE is inefficient
- **Short-seq bypass**: Sequences < 192 tokens skip splitting (overhead > benefit)

See [`experimental/README.md`](experimental/README.md) for full documentation, usage, and build instructions and limitations.

> **Note:** ANE Split is tested on M4. M5 introduced ANE architecture changes that may break the private API bridge — not yet validated on M5.

## Quantization

| Component | Scheme | Granularity |
|-----------|--------|-------------|
| W8A8 weights | Symmetric INT8 | Per-channel or per-group (gs=64/128) |
| W4A8 weights | Symmetric INT4 (zp=8) | Per-column |
| Activations | Symmetric INT8 | Per-token |
| Accumulation | INT32 | — |
| Output dequant | `C_fp16 = C_int32 * s_act * s_weight` | Per-element |

## Limitations

- **M=1 individual operator**: Per-channel MV kernel is slower than MLX W4A16 for isolated decode calls. The per-group MV kernel is within 5% of MLX W8A16 decode speed in end-to-end benchmarks.
- **Apple M5+ only** for INT8 TensorOps: M4 and below installs but `is_available()` returns False.
- **W4A8 slower than W8A8**: INT4→INT8 unpack ALU overhead (Metal 4 matmul2d has no native INT4 operand).

## Tools

### Unified PPL Evaluation

```bash
# Run all 5 configurations in one script
python tools/eval_ppl_all.py --num-samples 50

# Evaluates: FP16, W8A16 (MLX native), W8A8 per-channel, per-group(gs=64), per-group(gs=128)
# Outputs comparison table at the end
```

## Roadmap

- [x] One-line model conversion API (`convert_model`, auto prefill/decode)
- [x] Automatic dtype handling (float16 / bfloat16)
- [x] Per-channel and per-group W8A8 quantization
- [x] Dedicated decode MV kernel (matches native MLX speed)
- [x] Conditional compilation (M4 graceful fallback)
- [x] mlx_vlm and mlx_lm integration examples
- [ ] ANE primitives lazy evaluation
- [ ] Integrated Pruning Feature
- [ ] KVCache quantization

## Authors

Multimodal Team, Mininglamp Technology

For bug reports, feature requests, and usage questions, please open an issue in this repository.



## Citation

If you find this work useful, please cite:

```bibtex
@software{wang2026cider,
  author = {Multimodal Team, Mininglamp Technology},
  title = {Cider: Exploiting Unused INT8 TensorOps for Faster LLM Prefill on Apple Silicon},
  year = {2026},
  howpublished = {https://github.com/Mininglamp-AI/cider}
}
```

## License

MIT

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) by Apple — primitive API, NAXFrag kernel architecture
- Metal 4 MetalPerformancePrimitives for INT8 TensorOps
- [maderix/ANE](https://github.com/maderix/ANE) — inspired and informed our ANE+GPU tensor-parallel implementation
