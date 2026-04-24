# ANE Split — ANE+GPU Tensor Parallelism for Prefill Acceleration

Split each linear layer's GEMM along output channels: **ANE computes ~65%** while **GPU computes ~35%**, running concurrently. This exploits the idle Apple Neural Engine during LLM inference to speed up prefill without any accuracy loss.

> **Platform:** Apple M4 (tested). M5 ANE API changes may cause failures — not yet validated.

## How It Works

```
Input x ──┬── ANE (65% output channels, FP32, private API) ──┐
           │                                                    ├── concat → output
           └── GPU (35% output channels, FP16, MLX matmul)  ──┘
```

1. **SplitLinear** wraps each `nn.Linear` / `nn.QuantizedLinear`
2. In **prefill mode** (seq ≥ 192): split path with ANE+GPU concurrency
3. In **decode mode**: falls back to original `nn.Linear` on GPU (zero overhead)
4. Same-input projections (Q/K/V, Gate/Up) share input preparation via `_InputGroup`

### Automatic Layer Routing

| Layer Type | Routing | Reason |
|------------|---------|--------|
| Q, K, V, O projections | ANE+GPU split | Expand: IC → OC, ANE efficient |
| Gate, Up projections | ANE+GPU split | Expand: IC → OC, ANE efficient |
| Down projection | GPU only | Narrow: IC > 2×OC, ANE inefficient |
| Short sequences (< 192) | GPU only | Split overhead > benefit |

## Performance (Apple M4, Qwen3-VL-2B)

| seq | FP16 GPU | W8A16 GPU | SplitLinear | Speedup vs FP16 | Speedup vs W8A16 |
|-----|----------|-----------|-------------|------------------|-------------------|
| 256 | 321.8 ms | 318.5 ms | **299.7 ms** | **1.07×** | **1.06×** |
| 512 | 649.1 ms | 641.2 ms | **552.2 ms** | **1.18×** | **1.16×** |
| 1024 | 1324.0 ms | 1348.6 ms | **1156.9 ms** | **1.14×** | **1.17×** |

- Accuracy: cos ≈ 1.0, top-1 match = 100%
- 168 layers split (28 layers × 6 projections), 28 GPU-only (down_proj)

## Files

```
ane_split/
├── split_linear.py          # SplitLinear + ANEBridge + patch_model()
├── bench.py                 # End-to-end benchmark (W8A16 vs SplitLinear)
├── libane_bridge_v6.m       # ANE private API bridge (Objective-C source)
├── libane_bridge_v6.dylib   # Pre-built ANE bridge (arm64, macOS)
├── ane_transpose_bench.c    # vDSP transpose microbenchmark
└── README.md
```

## Quick Start

```python
from split_linear import patch_model, SplitLinear

# Load any MLX VLM model
from mlx_vlm.utils import load as vlm_load
model, processor = vlm_load("path/to/model")

# Patch all linear layers (one-liner)
bridge = patch_model(model, seq=512)

# Enable split for prefill, disable for decode
SplitLinear.set_prefill(True)   # prefill: ANE+GPU parallel
# ... run prefill ...
SplitLinear.set_prefill(False)  # decode: original GPU path
```

## Benchmark

```bash
# Default: seq=512
python3 bench.py

# Custom seq length
python3 bench.py 1024

# Custom model path
MODEL_PATH=/path/to/model python3 bench.py 512
```

## Building the ANE Bridge

The pre-built `libane_bridge_v6.dylib` is included. To rebuild from source:

```bash
clang -shared -O2 -framework Foundation -framework CoreML \
    -framework Accelerate -o libane_bridge_v6.dylib libane_bridge_v6.m
```

> Requires macOS with ANE private frameworks. Uses undocumented `_ANEClient` API.

## Limitations

- **M4 only** — M5 ANE internal changes may break the private API bridge
- **Fixed sequence length** — ANE models are compiled for a specific seq; re-patch needed for different lengths
- **FP32 on ANE** — ANE operates in FP32 (no INT8/FP16 GEMM); benefit comes from parallelism, not precision
- **Memory overhead** — ANE models consume additional system memory (~200MB for 2B model)
- **No decode benefit** — Decode is single-token, falls back to GPU (no split overhead)

## License

MIT
