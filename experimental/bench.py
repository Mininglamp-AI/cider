#!/usr/bin/env python3
"""
bench_w8a16_splitlinear.py — W8A16 vs SplitLinear prefill benchmark.

Usage:
    python3 bench_w8a16_splitlinear.py [seq_len]   # default 512

Requires:
    - mlx_vlm with Qwen3-VL-2B-Instruct-8bit model
    - split_linear.py (V4) + libane_bridge_v6.dylib in same directory
"""
import sys, time, os
import numpy as np
import mlx.core as mx
from mlx_vlm.utils import load as vlm_load
from mlx_vlm.models.cache import KVCache

# ─── Config ───
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 512
W8A16_MODEL = '~/Downloads/sft_baseline_v2_w8a16'
N_WARMUP = 3
N_BENCH = 8

def bench_forward(lang, ids, pos, n_layers, reps):
    ts = []
    for _ in range(reps):
        cache = [KVCache() for _ in range(n_layers)]
        t0 = time.perf_counter()
        mx.eval(lang(ids, cache=cache, position_ids=pos).logits)
        ts.append((time.perf_counter() - t0) * 1000)
    return ts

def main():
    print(f"\n{'='*60}")
    print(f"  W8A16 vs SplitLinear Prefill Benchmark (seq={SEQ})")
    print(f"{'='*60}")
    print(f"  Model: {W8A16_MODEL}")
    print(f"  Warmup: {N_WARMUP}, Bench: {N_BENCH}\n")

    # Load model
    print("[1/4] Loading W8A16 model...")
    model, _ = vlm_load(W8A16_MODEL)
    lang = model.language_model
    N = lang.args.num_hidden_layers
    print(f"  {N} layers loaded\n")

    # Prepare inputs
    ids = mx.ones((1, SEQ), dtype=mx.int32)
    pos = mx.broadcast_to(
        mx.arange(SEQ).reshape(1, SEQ)[None, :, :],
        (3, 1, SEQ)
    )
    mx.eval(ids, pos)

    # ─── W8A16 Baseline ───
    print("[2/4] W8A16 GPU baseline")
    bench_forward(lang, ids, pos, N, N_WARMUP)  # warmup
    ts_w8 = bench_forward(lang, ids, pos, N, N_BENCH)
    for i, t in enumerate(ts_w8):
        print(f"  Run {i+1}: {t:.1f}ms")
    med_w8 = float(np.median(ts_w8))
    print(f"  Median: {med_w8:.1f}ms\n")

    # ─── Reference logits ───
    print("[3/4] Computing reference logits for accuracy check...")
    c_ref = [KVCache() for _ in range(N)]
    ref = np.array(lang(ids, cache=c_ref, position_ids=pos).logits.astype(mx.float32))

    # ─── SplitLinear ───
    print("[4/4] Patch with SplitLinear + benchmark")
    from split_linear import patch_model, SplitLinear
    bridge = patch_model(model, SEQ)
    SplitLinear.set_prefill(True)

    # Warmup
    bench_forward(lang, ids, pos, N, N_WARMUP)

    # Accuracy
    c_hyb = [KVCache() for _ in range(N)]
    hyb = np.array(lang(ids, cache=c_hyb, position_ids=pos).logits.astype(mx.float32))
    cos = float(np.dot(ref.flatten(), hyb.flatten()) /
                (np.linalg.norm(ref.flatten()) * np.linalg.norm(hyb.flatten()) + 1e-12))
    top1 = float((ref.argmax(-1) == hyb.argmax(-1)).mean() * 100)
    print(f"  Accuracy: cos={cos:.6f}, top1={top1:.1f}%")

    # Benchmark
    ts_sp = bench_forward(lang, ids, pos, N, N_BENCH)
    for i, t in enumerate(ts_sp):
        print(f"  Run {i+1}: {t:.1f}ms")
    med_sp = float(np.median(ts_sp))

    # ─── Summary ───
    speedup = med_w8 / med_sp
    delta = med_sp - med_w8
    print(f"\n{'='*60}")
    print(f"  W8A16 GPU:     {med_w8:.1f}ms")
    print(f"  SplitLinear:   {med_sp:.1f}ms  ({speedup:.3f}x)")
    print(f"  Delta:         {delta:+.1f}ms")
    print(f"  Accuracy:      cos={cos:.6f}  top1={top1:.1f}%")
    print(f"{'='*60}")

    if speedup >= 1.0:
        print(f"  ✅ SplitLinear faster by {(speedup-1)*100:.1f}%")
    else:
        print(f"  ⚠️  SplitLinear slower by {(1-speedup)*100:.1f}%")

if __name__ == '__main__':
    main()
