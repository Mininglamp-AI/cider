#!/usr/bin/env python3
"""
W8A16 Benchmark: MLX QuantizedLinear vs SplitLinear (ANE+GPU)
Compares native MLX 8-bit inference against SplitLinear on the same model.

Usage:
    python3 bench.py [seq_len]
    MODEL_PATH=/path/to/model python3 bench.py 512
"""
import sys, os, time, gc
import numpy as np

# Must use py314 env
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_linear import patch_model, SplitLinear

from mlx_vlm.utils import load as vlm_load
from mlx_vlm.models.cache import KVCache

W8A16_MODEL = os.environ.get('MODEL_PATH', '/Users/ws/Downloads/weights/mlx/Qwen3-VL-2B-Instruct-8bit')
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N_WARMUP = 3
N_BENCH = 10


def bench_prefill(lang, seq, n_warmup, n_bench, label=""):
    """Benchmark prefill latency."""
    N = lang.args.num_hidden_layers
    ids = mx.ones((1, seq), dtype=mx.int32)
    pos = mx.broadcast_to(mx.arange(seq).reshape(1, seq)[None, :, :], (3, 1, seq))
    mx.eval(ids, pos)

    # Warmup
    for _ in range(n_warmup):
        c = [KVCache() for _ in range(N)]
        mx.eval(lang(ids, cache=c, position_ids=pos).logits)

    # Benchmark
    ts = []
    for i in range(n_bench):
        c = [KVCache() for _ in range(N)]
        t0 = time.perf_counter()
        mx.eval(lang(ids, cache=c, position_ids=pos).logits)
        t = (time.perf_counter() - t0) * 1000
        ts.append(t)
        print(f"  [{label}] Run {i+1}: {t:.1f}ms")

    med = float(np.median(ts))
    return med


def get_reference_logits(lang, seq):
    """Get reference logits for accuracy comparison."""
    N = lang.args.num_hidden_layers
    ids = mx.ones((1, seq), dtype=mx.int32)
    pos = mx.broadcast_to(mx.arange(seq).reshape(1, seq)[None, :, :], (3, 1, seq))
    mx.eval(ids, pos)
    c = [KVCache() for _ in range(N)]
    out = lang(ids, cache=c, position_ids=pos)
    logits = np.array(out.logits)
    mx.eval()
    return logits


def main():
    print(f"\n{'='*65}")
    print(f"  W8A16 Benchmark: MLX QuantizedLinear vs SplitLinear")
    print(f"  Model: Qwen3-VL-2B-Instruct-8bit")
    print(f"  seq={SEQ}, warmup={N_WARMUP}, bench={N_BENCH}")
    print(f"{'='*65}\n")

    # ── Phase 1: MLX W8A16 Baseline ──
    print("[1/4] Loading W8A16 model...")
    model, proc = vlm_load(W8A16_MODEL)
    lang = model.language_model

    # Confirm it's quantized
    la = lang.model.layers[0]
    print(f"  q_proj: {type(la.self_attn.q_proj).__name__} "
          f"bits={la.self_attn.q_proj.bits} group_size={la.self_attn.q_proj.group_size}")

    print(f"\n[2/4] MLX W8A16 baseline (seq={SEQ})")
    ref_logits = get_reference_logits(lang, SEQ)
    bl = bench_prefill(lang, SEQ, N_WARMUP, N_BENCH, label="W8A16")

    # ── Phase 2: Patch with SplitLinear ──
    print(f"\n[3/4] Patching with SplitLinear...")
    bridge = patch_model(model, SEQ)
    SplitLinear.set_prefill(True)

    # Accuracy check
    hyb_logits = get_reference_logits(lang, SEQ)
    cos = float(np.dot(ref_logits.flatten(), hyb_logits.flatten()) /
                (np.linalg.norm(ref_logits) * np.linalg.norm(hyb_logits) + 1e-12))
    top1_ref = ref_logits.argmax(-1)
    top1_hyb = hyb_logits.argmax(-1)
    top1_match = float((top1_ref == top1_hyb).mean() * 100)
    print(f"  Accuracy: cos={cos:.6f}, top1={top1_match:.1f}%")

    print(f"\n[4/4] SplitLinear benchmark (seq={SEQ})")
    sl = bench_prefill(lang, SEQ, N_WARMUP, N_BENCH, label="Split")

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  RESULTS (seq={SEQ})")
    print(f"{'='*65}")
    print(f"  MLX W8A16 (QuantizedLinear):  {bl:.1f}ms")
    print(f"  SplitLinear (ANE+GPU FP16):   {sl:.1f}ms  ({bl/sl:.3f}x)")
    print(f"  Accuracy: cos={cos:.6f}, top1={top1_match:.1f}%")
    print(f"  ANE models loaded: {bridge.model_count}")
    print(f"{'='*65}")

    # Multi-seq sweep
    print(f"\n\n{'='*65}")
    print(f"  Multi-seq sweep")
    print(f"{'='*65}")

    # Need to reload for fair comparison at different seq lengths
    # (SplitLinear is compiled for fixed seq, but W8A16 baseline doesn't care)
    for test_seq in [128, 256, 512]:
        if test_seq == SEQ:
            print(f"\n  seq={test_seq}: W8A16={bl:.1f}ms  Split={sl:.1f}ms  "
                  f"ratio={bl/sl:.3f}x  cos={cos:.6f}")
            continue

        # For other seq lengths, only run W8A16 baseline
        # (SplitLinear is compiled for SEQ, so we'd need re-patch)
        print(f"\n  seq={test_seq}: (SplitLinear compiled for seq={SEQ}, "
              f"skipping cross-seq comparison)")


if __name__ == '__main__':
    main()
