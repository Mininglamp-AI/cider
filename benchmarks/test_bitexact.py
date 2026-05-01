#!/usr/bin/env python3
"""Bit-exact correctness test for INT8×INT8→INT32 TensorOps kernel.

Pure integer matmul must produce EXACT results — no floating-point
tolerance, no cosine similarity. Every element must match bit-for-bit.

Usage:
    cd /path/to/cider
    python tests/test_bitexact.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
from cider import int8_matmul_int32

# ── Test shapes ─────────────────────────────────────────────────
SHAPES = [
    # (M, K, N)
    (1,    2,    2),       # minimal
    (2,    4,    2),       # tiny
    (1,    128,  128),     # single row
    (1,    4096, 4096),    # decode-size
    (4,    2048, 2048),    # small batch
    (16,   4096, 4096),    # medium
    (32,   4096, 4096),    # tile boundary (small→large)
    (64,   4096, 4096),    # large tile
    (128,  4096, 4096),    # full large tile
    (16,   4096, 8192),    # wide N
    (16,   8192, 4096),    # wide K
    (64,   2048, 4096),    # mixed
    (7,    513,  1025),    # non-aligned (prime-ish)
    (33,   4097, 4095),    # non-aligned near 4096
]


def test_bitexact():
    print("=" * 70)
    print("INT8×INT8→INT32 Bit-Exact Tests")
    print("=" * 70)
    passed, failed = 0, 0

    for M, K, N in SHAPES:
        np.random.seed(42)
        a_np = np.random.randint(-128, 128, (M, K), dtype=np.int8)
        b_np = np.random.randint(-128, 128, (N, K), dtype=np.int8)

        # GPU kernel
        c_gpu = int8_matmul_int32(mx.array(a_np), mx.array(b_np))
        mx.eval(c_gpu)
        c_gpu_np = np.array(c_gpu)

        # Numpy reference (int64 to avoid overflow in accumulation)
        c_ref = (a_np.astype(np.int64) @ b_np.astype(np.int64).T).astype(np.int32)

        # Bit-exact check
        match = np.array_equal(c_gpu_np, c_ref)
        if match:
            passed += 1
            tag = "PASS"
        else:
            failed += 1
            diff = (c_gpu_np != c_ref)
            n_diff = diff.sum()
            # Show first mismatch
            idx = np.argwhere(diff)[0]
            tag = f"FAIL ({n_diff}/{M*N} mismatches, first@{tuple(idx)}: gpu={c_gpu_np[tuple(idx)]} ref={c_ref[tuple(idx)]})"

        print(f"  M={M:>4d} K={K:>4d} N={N:>4d}  [{tag}]")

    print(f"\nBit-exact: {passed}/{passed+failed} passed")
    return failed == 0


if __name__ == "__main__":
    ok = test_bitexact()
    print("\n" + "=" * 70)
    if ok:
        print("ALL BIT-EXACT TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
