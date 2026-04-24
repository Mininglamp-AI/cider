# cider

Hardware-accelerated **INT8 TensorOps** quantized matmul for Apple M5+ GPUs, implemented as **MLX custom primitives** with full lazy evaluation support.

First known open-source implementation of INT8×INT8→INT32 GEMM on Apple GPU via Metal 4 `cooperative_tensor` / `matmul2d` API.

## Modes

| Mode | Weights | Activations | Compute Path | Status |
|------|---------|-------------|--------------|--------|
| **W8A8** | INT8 per-column | INT8 per-token | TensorOps matmul2d | ✅ Stable |
| **W4A8** | INT4 packed (uint8) | INT8 per-token | Unpack→TensorOps | ✅ Stable |
| W4A16 | — | — | MLX built-in | Baseline |
| W8A16 | — | — | MLX built-in | Baseline |

**W4A16 and W8A16 are already supported by MLX natively** — this SDK provides the missing W8A8 and W4A8 modes that MLX does not implement.

## Why This Exists

MLX's quantization is **weight-only**: QuantizedLinear dequantizes weights to FP16 and uses FP16 GEMM. Even with Metal 4 NAX path, MLX never uses INT8×INT8 TensorOps.

This SDK provides true INT8 activation quantization + INT8 TensorOps compute, which:
- **W8A8**: Up to **2x faster** than MLX W4A16 at batch sizes ≥ 32
- **W4A8**: **Half the weight memory** of W8A8, still faster than MLX W4A16

## Performance (Apple M5 Pro, 4096×4096)

| M | W8A8 | W4A8 | MLX W4A16 | W8A8 vs W4A16 |
|---|------|------|-----------|---------------|
| 1 | 0.47ms | 0.52ms | 0.21ms | 0.44x |
| 16 | 0.23ms | 0.32ms | 0.34ms | **1.48x** |
| 64 | 0.24ms | 0.40ms | 0.53ms | **2.21x** |
| 128 | 0.29ms | 0.47ms | 0.40ms | **1.38x** |
| 256 | 0.41ms | 0.69ms | 0.71ms | **1.73x** |

## Requirements

- macOS 26+ (Tahoe)
- Apple M5+ (Metal 4 TensorOps)
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
├── CMakeLists.txt
├── pyproject.toml
├── setup.py
└── README.md
```

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

The INT8 GEMM uses Apple's `mpp::tensor_ops::matmul2d(16, 32, 16)` — hardware-accelerated INT8×INT8→INT32 matrix multiply available on M5+ via Metal 4's `cooperative_tensor` API.

### Tile Configurations

| Config | BM | BN | BK | SK | Threads | Use When |
|--------|----|----|----|----|---------|----------|
| Large  | 128 | 128 | 512 | 32 | 512 | M > 64 |
| Small  | 32  | 128 | 512 | 32 | 128 | M ≤ 64 |

Auto-selected based on M. L2 cache swizzle dispatch included.

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

- [ ] Group quantization for W8A8/W4A8
- [ ] Upstream to MLX as native op
- [ ] PyTorch tensor binding via pybind11
- [ ] Fused QKV + gate/up projection layers
- [ ] Decode path optimization (hybrid W4A16 decode + W8A8 prefill)

## Authors

Multimodal Team, Mininglamp Technology

Please refer to wangshuo.e@mininglamp.com if you find any issue.


## Citation

If you find this work useful, please cite:

```bibtex
@software{wang2026cider,
  author = {Multimodal Team, Mininglamp Technology}
  title = {Cider: Hardware-Accelerated INT8 TensorOps for Apple Silicon},
  year = {2026},
  howpublished = {GitHub}
}
```

## License

MIT

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) by Apple — primitive API, NAXFrag kernel architecture
- Metal 4 MetalPerformancePrimitives for INT8 TensorOps
