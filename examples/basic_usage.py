#!/usr/bin/env python3
"""Basic usage example for Cider.

Demonstrates W8A8 and W4A8 quantized linear layers.
All operations return lazy mx.array — evaluated on mx.eval().
"""
import numpy as np
import mlx.core as mx
from cider import W8A8Linear, W4A8Linear, is_available

# Check hardware
if not is_available():
    print("INT8 TensorOps not available (requires Apple M5+)")
    exit(1)
print("INT8 TensorOps: available")

# ── Prepare weights ──────────────────────────────────────────────
K, N = 4096, 4096
W_fp16 = np.random.randn(K, N).astype(np.float16)

# ── W8A8 ─────────────────────────────────────────────────────────
w8a8 = W8A8Linear.from_weights(W_fp16)
print(f"W8A8: [{w8a8.input_dims}, {w8a8.output_dims}]")

x = mx.random.normal((32, K)).astype(mx.float16)
y = w8a8(x)          # lazy — not computed yet
mx.eval(y)            # now the GPU runs
print(f"W8A8 output: {y.shape}, dtype={y.dtype}")

# ── W4A8 ─────────────────────────────────────────────────────────
w4a8 = W4A8Linear.from_weights(W_fp16)
print(f"W4A8: [{w4a8.input_dims}, {w4a8.output_dims}]")
print(f"W4A8 weight storage: {w4a8.packed_weight.nbytes / 1024:.0f} KB "
      f"(vs W8A8: {w8a8.weight.nbytes / 1024:.0f} KB)")

y4 = w4a8(x)
mx.eval(y4)
print(f"W4A8 output: {y4.shape}, dtype={y4.dtype}")

# ── Lazy composition ─────────────────────────────────────────────
# Multiple ops compose into a single MLX graph, evaluated together
z = w8a8(x) + w4a8(x)  # builds graph, no GPU work yet
mx.eval(z)               # single evaluation
print(f"Composed output: {z.shape}")
