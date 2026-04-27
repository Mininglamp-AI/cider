#!/usr/bin/env python3
"""Benchmark: W8A16 baseline vs W8A8 on LLM-only (ViT stays float).
Uses bench_e2e_wxa16.py's real trajectory data pipeline.
"""
import sys, os, time, gc, io, base64, re, json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from PIL import Image
from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "vlm_service"))

from session_data.replay_prompt import build_prompt_at_step
from vlm_service.custom_qwen3vl import custom_stream_generate
from mlx_vlm.utils import load as vlm_load

W8A16_MODEL = '/path/to/sft_baseline_v2_w8a16'
SESSION_DIR = os.path.join(ROOT_DIR, "session_data")
STEPS       = [0, 1, 2]
MAX_TOKENS  = 200
PREFILL_STEP = 8192
N_SPEED_RUNS = 3


def load_step(session_dir, step):
    data = build_prompt_at_step(session_dir, step)
    pil_images = [Image.open(io.BytesIO(base64.b64decode(b))) for b in data['images']]
    return data, pil_images


def build_messages(data):
    prompt_text = data['prompt']
    n_images = prompt_text.count('<image>')
    parts = prompt_text.split('<image>')
    content = []
    for i, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if i < n_images:
            content.append({"type": "image"})
    return [
        {"role": "system", "content": data['system_prompt']},
        {"role": "user", "content": content},
    ]


def run_generate(model, processor, messages, pil_images):
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
    text = ""; prefill_time = 0; decode_tps = 0; prompt_tokens = 0; gen_tokens = 0
    for resp in custom_stream_generate(
        model, processor, prompt=prompt, image=pil_images,
        max_tokens=MAX_TOKENS, temperature=0.0,
        prefill_step_size=PREFILL_STEP, verbose=False,
    ):
        text += resp.text
        prefill_time = resp.prompt_tokens / resp.prompt_tps if resp.prompt_tps > 0 else 0
        decode_tps = resp.generation_tps
        prompt_tokens = resp.prompt_tokens
        gen_tokens = resp.generation_tokens
    total_time = prefill_time + (gen_tokens / decode_tps if decode_tps > 0 else 0)
    return text, {
        'prefill_ms': prefill_time * 1000,
        'prefill_tps': prompt_tokens / prefill_time if prefill_time > 0 else 0,
        'decode_tps': decode_tps,
        'prompt_tokens': prompt_tokens,
        'gen_tokens': gen_tokens,
        'total_ms': total_time * 1000,
    }


def extract_action(text):
    m = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def main():
    print("=" * 70)
    print("  W8A16 Baseline  vs  W8A8 LLM-only (ViT stays float)")
    print("=" * 70)

    # Load trajectory data
    print("\nLoading trajectory data...")
    step_data = {}
    for s in STEPS:
        data, images = load_step(SESSION_DIR, s)
        msgs = build_messages(data)
        step_data[s] = (data, images, msgs)
        print(f"  Step {s}: {len(images)} images")

    results = {}

    # ── Config 1: W8A16 Baseline (no cider) ──
    print(f"\n{'='*70}")
    print("  W8A16 Baseline (no cider)")
    print(f"{'='*70}")
    model, proc = vlm_load(W8A16_MODEL)
    
    # Count ViT vs LLM layers for reference
    n_vit_linear = sum(1 for _ in model.vision_tower.parameters())
    print(f"  ViT parameters groups: {n_vit_linear}")
    
    cfg_results = {}
    for s in STEPS:
        data, images, msgs = step_data[s]
        print(f"\n  Step {s} ({len(images)} imgs):")
        # warmup
        text, timing = run_generate(model, proc, msgs, images)
        print(f"    Warmup: {timing['prompt_tokens']} tok, prefill {timing['prefill_ms']:.0f}ms")
        # speed runs
        timings = []
        for r in range(N_SPEED_RUNS):
            _, t = run_generate(model, proc, msgs, images)
            timings.append(t)
            print(f"    Run {r+1}: prefill {t['prefill_ms']:.0f}ms ({t['prefill_tps']:.0f} tok/s) | decode {t['decode_tps']:.1f} tok/s | total {t['total_ms']:.0f}ms")
        cfg_results[s] = {
            'text': text, 'action': extract_action(text), 'timings': timings,
            'prompt_tokens': timings[0]['prompt_tokens'],
        }
    results['baseline'] = cfg_results
    del model, proc; gc.collect(); mx.clear_cache()

    # ── Config 2: W8A8 LLM-only ──
    print(f"\n{'='*70}")
    print("  W8A8 LLM-only (ViT stays float)")
    print(f"{'='*70}")
    model, proc = vlm_load(W8A16_MODEL)
    
    # Only convert language_model, leave vision_tower untouched
    from cider import convert_model
    stats = convert_model(model.language_model)
    print(f"  [cider] LLM-only: {stats}")
    
    # Verify ViT is untouched
    from cider.nn import CiderLinear
    vit_cider = sum(1 for m in model.vision_tower.modules() if isinstance(m, CiderLinear))
    print(f"  ViT CiderLinear count (should be 0): {vit_cider}")
    
    cfg_results = {}
    for s in STEPS:
        data, images, msgs = step_data[s]
        print(f"\n  Step {s} ({len(images)} imgs):")
        text, timing = run_generate(model, proc, msgs, images)
        print(f"    Warmup: {timing['prompt_tokens']} tok, prefill {timing['prefill_ms']:.0f}ms")
        timings = []
        for r in range(N_SPEED_RUNS):
            _, t = run_generate(model, proc, msgs, images)
            timings.append(t)
            print(f"    Run {r+1}: prefill {t['prefill_ms']:.0f}ms ({t['prefill_tps']:.0f} tok/s) | decode {t['decode_tps']:.1f} tok/s | total {t['total_ms']:.0f}ms")
        cfg_results[s] = {
            'text': text, 'action': extract_action(text), 'timings': timings,
            'prompt_tokens': timings[0]['prompt_tokens'],
        }
    results['llm_w8a8'] = cfg_results
    del model, proc; gc.collect(); mx.clear_cache()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: W8A16 Baseline vs W8A8 LLM-only")
    print(f"{'='*70}")
    print(f"{'Step':>5} {'Tok':>6} {'Baseline':>12} {'LLM W8A8':>12} {'Speedup':>10} {'DecBase':>10} {'DecW8A8':>10}")
    print("-" * 70)
    
    for s in STEPS:
        b_ts = results['baseline'][s]['timings']
        w_ts = results['llm_w8a8'][s]['timings']
        b_prefill = float(np.median([t['prefill_ms'] for t in b_ts]))
        w_prefill = float(np.median([t['prefill_ms'] for t in w_ts]))
        b_total = float(np.median([t['total_ms'] for t in b_ts]))
        w_total = float(np.median([t['total_ms'] for t in w_ts]))
        b_decode = float(np.median([t['decode_tps'] for t in b_ts]))
        w_decode = float(np.median([t['decode_tps'] for t in w_ts]))
        tok = results['baseline'][s]['prompt_tokens']
        sp_prefill = b_prefill / w_prefill if w_prefill > 0 else 0
        sp_total = b_total / w_total if w_total > 0 else 0
        print(f"{s:>5} {tok:>6} {b_prefill:>8.0f}ms   {w_prefill:>8.0f}ms   {sp_prefill:>7.2f}x   {b_decode:>7.1f}t/s {w_decode:>7.1f}t/s")
    
    # Accuracy check
    print(f"\n  ACTION COMPARISON:")
    for s in STEPS:
        ba = results['baseline'][s]['action']
        wa = results['llm_w8a8'][s]['action']
        match = "✅ IDENTICAL" if ba == wa else "⚠️ DIFFERENT"
        print(f"    Step {s}: {match}")
        if ba != wa:
            print(f"      Baseline: {ba}")
            print(f"      LLM W8A8: {wa}")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
