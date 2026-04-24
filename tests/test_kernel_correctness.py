#!/usr/bin/env python3
"""Kernel-level correctness tests for cider.

Tests W8A8 and W4A8 primitives against numpy reference implementations.
Uses both cosine similarity and normalized mean absolute error (NMAE).

Usage:
    cd /path/to/cider
    python tests/test_kernel_correctness.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
from cider import w8a8_linear, w4a8_linear, quantize_weight_int8, pack_weight_int4


# ── Numpy reference: W8A8 ───────────────────────────────────────

def ref_w8a8(x_fp16, w_int8, scale_w):
    x = x_fp16.astype(np.float32)
    row_max = np.max(np.abs(x), axis=1, keepdims=True)
    s_act = np.where(row_max == 0, 1.0, row_max / 255.0)
    x_int8 = np.clip(np.round(x / s_act), -128, 127).astype(np.float64)
    c = x_int8 @ w_int8.astype(np.float64)
    return (c * s_act * scale_w[np.newaxis, :]).astype(np.float32)


def ref_w4a8(x_fp16, packed_w, scale_w, K, zero_point=8):
    x = x_fp16.astype(np.float32)
    N = packed_w.shape[1]
    w_even = ((packed_w >> 4) & 0xF).astype(np.int8) - zero_point
    w_odd = (packed_w & 0xF).astype(np.int8) - zero_point
    w_int8 = np.empty((K, N), dtype=np.float64)
    w_int8[0::2, :] = w_even.astype(np.float64)
    w_int8[1::2, :] = w_odd.astype(np.float64)
    row_max = np.max(np.abs(x), axis=1, keepdims=True)
    s_act = np.where(row_max == 0, 1.0, row_max / 255.0)
    x_int8 = np.clip(np.round(x / s_act), -128, 127).astype(np.float64)
    c = x_int8 @ w_int8
    return (c * s_act * scale_w[np.newaxis, :]).astype(np.float32)


# ── Metrics ─────────────────────────────────────────────────────

def compute_metrics(y_gpu, y_ref):
    """Cosine similarity + normalized mean absolute error."""
    g = y_gpu.flatten().astype(np.float64)
    r = y_ref.flatten().astype(np.float64)
    cos = np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-12)
    nmae = np.mean(np.abs(g - r)) / (np.mean(np.abs(r)) + 1e-12)
    return cos, nmae


# ── Test shapes ─────────────────────────────────────────────────

SHAPES = [
    (1,    128,  128),
    (1,    4096, 4096),
    (4,    2048, 2048),
    (16,   4096, 4096),
    (32,   4096, 4096),
    (64,   4096, 4096),
    (128,  4096, 4096),
    (16,   4096, 8192),
    (16,   8192, 4096),
    (64,   2048, 4096),
]

COS_THRESHOLD = 0.9999    # cosine > 0.9999
NMAE_THRESHOLD = 0.005    # NMAE < 0.5%


def test_w8a8():
    print("=" * 70)
    print("W8A8 Kernel Correctness Tests")
    print("=" * 70)
    passed, failed = 0, 0

    for M, K, N in SHAPES:
        np.random.seed(42)
        x_fp16 = np.random.randn(M, K).astype(np.float16)
        w_fp32 = np.random.randn(K, N).astype(np.float32)
        w_int8, scale_w = quantize_weight_int8(w_fp32)

        y_gpu = w8a8_linear(mx.array(x_fp16), mx.array(w_int8), mx.array(scale_w))
        mx.eval(y_gpu)
        y_gpu_np = np.array(y_gpu).astype(np.float32)
        y_ref = ref_w8a8(x_fp16, w_int8, scale_w)

        cos, nmae = compute_metrics(y_gpu_np, y_ref)
        ok = cos >= COS_THRESHOLD and nmae <= NMAE_THRESHOLD
        passed += ok; failed += (not ok)
        tag = "PASS" if ok else "FAIL"
        print(f"  M={M:>4d} K={K:>4d} N={N:>4d}  cos={cos:.6f}  nmae={nmae:.6f}  [{tag}]")

    print(f"\nW8A8: {passed}/{passed+failed} passed")
    return failed == 0


def test_w4a8():
    print("\n" + "=" * 70)
    print("W4A8 Kernel Correctness Tests")
    print("=" * 70)
    passed, failed = 0, 0

    for M, K, N in SHAPES:
        np.random.seed(42)
        x_fp16 = np.random.randn(M, K).astype(np.float16)
        w_fp32 = np.random.randn(K, N).astype(np.float32)
        packed_w, scale_w = pack_weight_int4(w_fp32)

        y_gpu = w4a8_linear(mx.array(x_fp16), mx.array(packed_w), mx.array(scale_w))
        mx.eval(y_gpu)
        y_gpu_np = np.array(y_gpu).astype(np.float32)
        y_ref = ref_w4a8(x_fp16, packed_w, scale_w, K)

        cos, nmae = compute_metrics(y_gpu_np, y_ref)
        ok = cos >= COS_THRESHOLD and nmae <= NMAE_THRESHOLD
        passed += ok; failed += (not ok)
        tag = "PASS" if ok else "FAIL"
        print(f"  M={M:>4d} K={K:>4d} N={N:>4d}  cos={cos:.6f}  nmae={nmae:.6f}  [{tag}]")

    print(f"\nW4A8: {passed}/{passed+failed} passed")
    return failed == 0


if __name__ == "__main__":
    ok1 = test_w8a8()
    ok2 = test_w4a8()
    print("\n" + "=" * 70)
    if ok1 and ok2:
        print("ALL KERNEL CORRECTNESS TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
