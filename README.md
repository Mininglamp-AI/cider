# cider

Cider is developed on top of MLX for macOS. It provides online activation quantization operators absent in MLX, with custom int-matmul kernels built as MLX custom primitives supporting full lazy evaluation. It also extends mlx_vlm by fixing multiple bugs and adding on-device inference services.


## Modes

| Mode | Weights | Activations | Compute Path | Status |
|------|---------|-------------|--------------|--------|
| **W8A8** | INT8 per-column | INT8 per-token | TensorOps matmul2d | ✅ Stable |
| **W4A8** | INT4 packed (uint8) | INT8 per-token | Unpack→TensorOps | ✅ Stable |
| W4A16 | — | — | MLX built-in | Baseline |
| W8A16 | — | — | MLX built-in | Baseline |

**W4A16 and W8A16 are already supported by MLX natively** — this SDK provides the missing W8A8 and W4A8 modes that MLX does not implement.

MLX's quantization is **weight-only**: QuantizedLinear dequantizes weights to FP16 and uses FP16 GEMM. While MLX's Steel NAX templates are generic enough to be instantiated with INT8 types (and would achieve identical raw matmul throughput — [see our transparent benchmark](benchmarks/mlx_native/cider_vs_mlx_int8.md)), MLX does not provide the quantization/dequantization pipeline needed for actual W8A8 inference. Cider fills this gap with fused quantize-matmul-dequant primitives.

This SDK provides true INT8 activation quantization + INT8 TensorOps compute, which:
- **W8A8**: Up to **2x faster** than MLX W4A16 at batch sizes ≥ 32
- **W4A8**: **Half the weight memory** of W8A8, still faster than MLX W4A16

## Performance (Apple M5 Pro, 4096×4096)

###  Individual Operator Latency Comparison

| M | W8A8 | W4A8 | MLX W4A16 | W8A8 vs W4A16 |
|---|------|------|-----------|---------------|
| 1 | 0.47ms | 0.52ms | 0.21ms | 0.44x |
| 16 | 0.23ms | 0.32ms | 0.34ms | **1.48x** |
| 64 | 0.24ms | 0.40ms | 0.53ms | **2.21x** |
| 128 | 0.29ms | 0.47ms | 0.40ms | **1.38x** |
| 256 | 0.41ms | 0.69ms | 0.71ms | **1.73x** |

### End-to-End VLM Prefill (Qwen3-VL-2B)

Real model forward pass, chunked prefill (chunk=2048), bfloat16 model:

| Tokens | BF16 (baseline) | W8A8 Prefill | Speedup |
|--------|-----------------|--------------|---------|
| 1334   | 159ms           | **134ms**    | **1.19x** |
| 2393   | 298ms           | **254ms**    | **1.17x** |
| 3455   | 432ms           | **374ms**    | **1.15x** |

Decode uses original weights (zero overhead). Mode switching is instant.

### LLM Quantization: Precision vs. Speed Comparison


<table>
  <thead>
    <tr>
      <th>Models</th>
      <th>Quantization Configuration</th>
      <th>wikitext2 PPL（&#8595）</th>
      <th>Prefill Speed (tokens/s)（&#8593）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><b>Qwen3-8B</b></td>
      <td>FP16</td>
      <td>9.729</td>
      <td>1695</td>
    </tr>
    <tr>
      <td>W4A16 (AWQ)</td>
      <td>9.991</td>
      <td>1628</td>
    </tr>
    <tr>
      <td>W8A16 (GPTQ)</td>
      <td>9.707</td>
      <td>1484</td>
    </tr>
    <tr>
      <td>W8A8 (GPTQ)</td>
      <td>9.756</td>
      <td><b>2531</b></td>
    </tr>

    
  </tbody>
  <tr style="border-top: 1px solid #333;">
      <td colspan="4" style="padding: 0; height: 3px;"></td>
    </tr>
  <tbody>
    <tr>
      <td rowspan="4"><b>Llama3-8B</b></td>
      <td>FP16</td>
      <td>6.138</td>
      <td>1727</td>
    </tr>
    <tr>
      <td>W4A16 (GPTQ)</td>
      <td>6.809</td>
      <td>1579</td>
    </tr>
    <tr>
      <td>W8A16 (GPTQ)</td>
      <td>6.147</td>
      <td>1477</td>
    </tr>
    <tr>
      <td>W8A8 (GPTQ)</td>
      <td>6.271</td>
      <td><b>2520</b></td>
    </tr>
  </tbody>
  <tr style="border-top: 1px solid #f4efef;">
      <td colspan="4" style="padding: 0; height: 3px;"></td>
    </tr>
</table>

## Requirements

- Apple M5 (Metal 4 TensorOps)
- Python 3.12+
- MLX >= 0.31
- nanobind >= 2.12
- CMake >= 3.27

## Install

```bash
pip install -e .
```

This runs CMake to compile the C++ extension, then installs the Python package.

## Quick Start

### One-line Model Conversion (Recommended)

```python
from cider import convert_model

model, proc = load("path/to/model")  # Any MLX model

# Convert all Linear layers — one line, done.
convert_model(model)

# That's it. CiderLinear auto-detects:
#   seq_len > 1  → W8A8 INT8 TensorOps (faster prefill)
#   seq_len == 1 → original weights (optimal decode)
# No manual mode switching needed.
```

Works with **any MLX model** — Qwen, Llama, Mistral, etc. Automatically handles float16 and bfloat16.

### Layer-level API

```python
import numpy as np
import mlx.core as mx
from cider import W8A8Linear, W4A8Linear, is_available

assert is_available(), "Requires Apple M5+"

# Prepare weight
W = np.random.randn(4096, 4096).astype(np.float16)

# W8A8 linear
layer = W8A8Linear.from_weights(W)
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
from cider import w8a8_linear, w4a8_linear, quantize_weight_int8, pack_weight_int4

# Quantize weights (numpy, offline)
w_int8, scale = quantize_weight_int8(W_np)
packed_w4, scale4 = pack_weight_int4(W_np)

# Primitive calls (return lazy mx.array)
y = w8a8_linear(x, mx.array(w_int8), mx.array(scale))
y4 = w4a8_linear(x, mx.array(packed_w4), mx.array(scale4))
```

## Project Structure

```
cider/
├── cider/              # Python package
│   ├── __init__.py        # Public API
│   ├── ops.py             # Primitive wrappers + quantize helpers
│   ├── nn.py              # W8A8Linear, W4A8Linear (nn.Module)
│   ├── convert.py         # convert_model() high-level API
│   └── kernels/           # Metal shaders (bundled)
│       ├── w8a8_matmul.metal
│       ├── w4a8_matmul.metal
│       └── w8a8_quantize.metal
├── csrc/                  # C++ MLX primitives (nanobind)
│   ├── include/
│   │   ├── w8a8_primitive.h
│   │   └── w4a8_primitive.h
│   └── src/
│       ├── w8a8_primitive.mm
│       ├── w4a8_primitive.mm
│       └── prim_bindings.cpp
├── tests/
├── benchmarks/
|   └── bench_kernels
├── tutorial
|   ├── how_to_write_efficient_int_gemm_m5_en.md
|   ├── how_to_write_efficient_int_gemm_m5_zh.md
├── examples/
│   └── basic_usage.py
├── vlm_service/           # OpenAI-compatible VLM inference server
│   ├── server.py             # FastAPI server (streaming + non-streaming)
│   ├── core_infer.py         # HMInference engine (singleton)
│   ├── custom_qwen3vl.py     # Custom Qwen3-VL generation loop
│   └── config.py             # Config loader
├── config/
│   └── config.yaml           # Server & model configuration
├── experimental/             # ANE+GPU hybrid tensor parallelism (M4)
│   ├── split_linear.py       # SplitLinear + ANEBridge + patch_model()
│   ├── bench.py              # End-to-end benchmark
│   ├── libane_bridge_v6.m    # ANE private API bridge (Obj-C source)
│   └── README.md
├── tools                     # convert LLMCompressor model to mlx  and test ppl
|   ├──convert_compressed_tensors_to_mlx.py
|   ├──eval_w8a8.py
|   └──eval_ppl_baseline.py
├── CMakeLists.txt
├── pyproject.toml
├── setup.py
└── README.md
```

## VLM Inference Service

`vlm_service/` provides a ready-to-use **OpenAI-compatible** VLM inference server with W8A8 acceleration.

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

3. **Send requests** (OpenAI-compatible API):

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

- **Prefill** (processing input tokens, seq_len > 1): Uses W8A8 INT8 TensorOps for ~10-19% speedup
- **Decode** (generating tokens one by one, seq_len == 1): Falls back to original weights with zero overhead

No code changes needed — the switching is automatic based on input sequence length.

## Architecture

### MLX Custom Primitives

Both W8A8Linear and W4A8Linear are implemented as `mlx::core::Primitive` subclasses. This means:

1. **Lazy evaluation**: `y = layer(x)` builds a graph node, not immediate computation
2. **Graph composition**: Multiple primitive calls compose into a single MLX graph
3. **Stream scheduling**: MLX's scheduler handles GPU dispatch order

### Metal Kernel Pipeline

Each primitive dispatches two Metal compute kernels in a single command encoder:

1. **quantize_per_token**: FP16 activations → INT8 + per-token scales
2. **matmul_fused_dequant**: INT8 × INT8 → INT32 → FP16 (with fused scale dequantization)

For W4A8, step 2 includes inline INT4→INT8 unpacking in the fragment load.

### TensorOps matmul2d

The INT8 GEMM uses Apple's `mpp::tensor_ops::matmul2d(16, 32, 16)` — hardware-accelerated INT8×INT8→INT32 matrix multiply available on M5+ via Metal 4's `cooperative_tensor` API. This is the same hardware instruction available to MLX's NAX templates. Cider's kernel adds fused dequantization (INT32 × scales → FP16) in the store phase, avoiding an extra device memory round-trip. See [kernel comparison](benchmarks/mlx_native/cider_vs_mlx_int8.md) for details.

### Tile Configurations

| Config | BM | BN | BK | SK | Threads | Use When |
|--------|----|----|----|----|---------|----------|
| Large  | 128 | 128 | 512 | 32 | 512 | M > 64 |
| Small  | 32  | 128 | 512 | 32 | 128 | M ≤ 64 |

Auto-selected based on M. L2 cache swizzle dispatch included.

## ANE+GPU Heterogeneous Tensor Parallelism (experimental）

We found that during inference on Mac, only two hardware computing units—GPU and CPU—were utilized, while the ANE (Apple Neural Engine) computing unit on Mac remained idle. We identified this as a potential optimization opportunity. Inspired by [maderix/ANE](https://github.com/maderix/ANE), we conducted experimental work on a hybrid ANE+GPU inference mode. Currently, we apply this approach to tensor parallel computing. On the M4 chip, during synchronous-only forward inference (MLX natively uses a technique called lazy evaluation, which reduces synchronization overhead; in end-to-end testing, the hybrid inference currently shows no advantage, mainly because we have not yet implemented this using MLX's lazy evaluation—this remains future work), we observed approximately **3%~16%** performance improvement compared to pure GPU inference. We believe that GPU+ANE hybrid inference should have even greater potential for improvement.

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

Accuracy: cos ≈ 1.0, top-1 match = 100%.

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
| W8A8 weights | Symmetric INT8 | Per-column |
| W4A8 weights | Symmetric INT4 (zp=8) | Per-column |
| Activations | Symmetric INT8 | Per-token |
| Accumulation | INT32 | — |
| Output dequant | `C_fp16 = C_int32 * s_act * s_weight` | Per-element |

## Limitations

- **M=1 (decode)**: Slower than MLX W4A16 due to activation quantization overhead. For decode, use MLX's native W4A16.
- **Apple M5+ only**: Metal 4 TensorOps required. M1-M4 not supported.
- **Per-column quantization only**: No group quantization yet.
- **W4A8 slower than W8A8**: INT4→INT8 unpack ALU overhead (Metal 4 matmul2d has no native INT4 operand).

## Roadmap

- [x] One-line model conversion API (`convert_model`, auto prefill/decode)
- [x] Automatic dtype handling (float16 / bfloat16)
- [x] Hybrid prefill/decode (auto-detection by sequence length)
- [x] mlx_vlm and mlx_lm integration examples
- [ ] ANE primitives lazy evaluation
- [ ] Integrated Pruning Feature
- [ ] KVCache quantization

## Authors

Multimodal Team, Mininglamp Technology

Please refer to wangshuo.e@mininglamp.com if you find any issue.


## Citation

If you find this work useful, please cite:

```bibtex
@software{wang2026cider,
  author = {Multimodal Team, Mininglamp Technology}，
  title = {Cider: Hardware-Accelerated INT8 TensorOps for Apple Silicon},
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
