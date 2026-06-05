#!/usr/bin/env python3
"""
End-to-end benchmark on REAL trajectory data:
  1. GPU FP16 baseline
  2. GPU W8A16 (native quantized_matmul)
  3. GPU W8A8_pg (INT8 TensorOps via cider, pergroup)
  4. GPU W8A8_pc (INT8 TensorOps via cider, perchannel)
  5-8. Above 4 configs + Cider SDPA patch (decode acceleration)

Uses replay_prompt.build_prompt_at_step() to build real prompts with real screenshots.
Compares both accuracy (action output) and speed (prefill + decode).
"""
import sys, os, time, gc, re, json, io, base64
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
home_dir = os.path.expanduser("~")
FP16_MODEL  = os.path.join(home_dir, 'Downloads/checkpoint-50349')
W8A16_MODEL = os.path.join(home_dir, 'Downloads/checkpoint-50349-mlx_w8a16')
SESSION_DIR = os.path.join(ROOT_DIR, "session_data")
STEPS       = [0, 1, 2]  # step 0: 1img, step 1: 2imgs, step 2: 3imgs
MAX_TOKENS  = 200
PREFILL_STEP = 8192
N_SPEED_RUNS = 2  # per-step speed runs


def load_step(session_dir, step):
    """Load prompt + images for a given step."""
    data = build_prompt_at_step(session_dir, step)
    pil_images = [Image.open(io.BytesIO(base64.b64decode(b))) for b in data['images']]
    return data, pil_images


def build_messages(data):
    """Build chat messages from replay_prompt data, with <image> placeholders."""
    prompt_text = data['prompt']
    n_images = prompt_text.count('<image>')
    parts = prompt_text.split('<image>')
    content = []
    for i, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if i < n_images:
            content.append({"type": "image"})

    messages = [
        {"role": "system", "content": data['system_prompt']},
        {"role": "user", "content": content},
    ]
    return messages


def run_generate(model, processor, messages, pil_images):
    """Run one generation, return (text, timing_dict)."""
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

    text = ""
    prefill_time = 0
    decode_tps = 0
    prompt_tokens = 0
    gen_tokens = 0

    try:
        for resp in custom_stream_generate(
            model, processor,
            prompt=prompt,
            image=pil_images,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            prefill_step_size=PREFILL_STEP,
            verbose=False,
        ):
            text += resp.text
            prefill_time = resp.prompt_tokens / resp.prompt_tps if resp.prompt_tps > 0 else 0
            decode_tps = resp.generation_tps
            prompt_tokens = resp.prompt_tokens
            gen_tokens = resp.generation_tokens
    except UnicodeDecodeError:
        # mlx_vlm detokenizer sometimes fails on partial utf-8 sequences
        pass

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
    """Extract action from model output."""
    m = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_coords(action_str):
    """Extract (x, y) from action string."""
    if not action_str:
        return None
    nums = re.findall(r'\d+', action_str)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return None


def main():
    print("=" * 80)
    print("  E2E Benchmark: FP16 / W8A16 / W8A8 × MLX SDPA / Cider SDPA")
    print("=" * 80)
    print(f"  Session: {SESSION_DIR}")
    print(f"  Steps: {STEPS}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Speed runs per step: {N_SPEED_RUNS}")

    # Load all steps
    print("\nLoading trajectory data...")
    step_data = {}
    for s in STEPS:
        data, images = load_step(SESSION_DIR, s)
        msgs = build_messages(data)
        step_data[s] = (data, images, msgs)
        print(f"  Step {s}: {len(images)} images")

    # Import cider for SDPA patching
    import cider

    # configs: (key, model_path, use_cider_quant, use_cider_sdpa, label)
    configs = [
        # Without Cider SDPA (MLX default attention)
        ('fp16',          FP16_MODEL,  False, False, "FP16"),
        ('fp16+sdpa',     FP16_MODEL,  False, True,  "FP16+CiderSDPA"),
        ('w8a16',         W8A16_MODEL, False, False, "W8A16"),
        ('w8a16+sdpa',    W8A16_MODEL, False, True,  "W8A16+CiderSDPA"),
        ('w8a8_pg',       W8A16_MODEL, True,  False, "W8A8_pg"),
        ('w8a8_pg+sdpa',  W8A16_MODEL, True,  True,  "W8A8_pg+CiderSDPA"),
        ('w8a8_pc',       FP16_MODEL,  True,  False, "W8A8_pc"),
        ('w8a8_pc+sdpa',  FP16_MODEL,  True,  True,  "W8A8_pc+CiderSDPA"),
    ]

    all_results = {}

    for cfg_key, model_path, use_cider_quant, use_cider_sdpa, label in configs:
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")

        # Load model
        print(f"  Loading {model_path}...")
        model, proc = vlm_load(model_path)

        # Apply cider quantization if needed
        if use_cider_quant:
            from cider import convert_model
            stats = convert_model(model)
            print(f"  [cider quant] {stats}")

        # Apply cider SDPA if needed
        if use_cider_sdpa:
            cider.patch_sdpa(verbose=True)
        else:
            cider.unpatch_sdpa(verbose=False)

        cfg_results = {}

        for s in STEPS:
            data, images, msgs = step_data[s]
            print(f"\n  Step {s} ({len(images)} imgs):")

            # Warmup
            text, timing = run_generate(model, proc, msgs, images)
            print(f"    Warmup: {timing['prompt_tokens']} prompt tok, "
                  f"prefill {timing['prefill_ms']:.0f}ms "
                  f"({timing['prefill_tps']:.0f} tok/s), "
                  f"decode {timing['decode_tps']:.1f} tok/s")

            # Accuracy run (use warmup result)
            action = extract_action(text)
            coords = extract_coords(action)

            # Speed runs
            timings = []
            for r in range(N_SPEED_RUNS):
                _, t = run_generate(model, proc, msgs, images)
                timings.append(t)
                print(f"    Run {r+1}: prefill {t['prefill_ms']:.0f}ms "
                      f"({t['prefill_tps']:.0f} tok/s) | "
                      f"decode {t['decode_tps']:.1f} tok/s | "
                      f"total {t['total_ms']:.0f}ms")

            cfg_results[s] = {
                'text': text,
                'action': action,
                'coords': coords,
                'timings': timings,
                'prompt_tokens': timings[0]['prompt_tokens'],
            }

            print(f"    Action: {action}")

        all_results[cfg_key] = cfg_results

        # Free model before loading next
        del model, proc
        gc.collect()
        mx.clear_cache()

    # Ensure SDPA is unpatched after benchmark
    cider.unpatch_sdpa(verbose=False)

    # ── Speed Summary ──
    print(f"\n{'='*80}")
    print(f"  SPEED SUMMARY (median of {N_SPEED_RUNS} runs)")
    print(f"{'='*80}")

    cfg_keys = [c[0] for c in configs]
    cfg_labels = {c[0]: c[4] for c in configs}

    # Table: per-step results
    print(f"\n  {'Config':<22s}", end="")
    for s in STEPS:
        print(f" | Step{s} prefill  decode  total", end="")
    print()
    print(f"  {'─'*22}", end="")
    for _ in STEPS:
        print(f" | {'─'*30}", end="")
    print()

    for k in cfg_keys:
        line = f"  {cfg_labels[k]:<22s}"
        for s in STEPS:
            ts = all_results[k][s]['timings']
            pf = float(np.median([t['prefill_tps'] for t in ts]))
            dc = float(np.median([t['decode_tps'] for t in ts]))
            tt = float(np.median([t['total_ms'] for t in ts]))
            line += f" | {pf:>7.0f}t/s {dc:>5.1f}t/s {tt:>5.0f}ms"
        print(line)

    # ── Decode speed comparison (SDPA impact) ──
    print(f"\n{'='*80}")
    print(f"  DECODE SPEED: MLX SDPA vs Cider SDPA")
    print(f"{'='*80}")

    # Group by base config
    base_configs = ['fp16', 'w8a16', 'w8a8_pg', 'w8a8_pc']
    for base in base_configs:
        sdpa_key = base + '+sdpa'
        if base not in all_results or sdpa_key not in all_results:
            continue

        print(f"\n  {cfg_labels[base]}:")
        for s in STEPS:
            ts_base = all_results[base][s]['timings']
            ts_sdpa = all_results[sdpa_key][s]['timings']
            dc_base = float(np.median([t['decode_tps'] for t in ts_base]))
            dc_sdpa = float(np.median([t['decode_tps'] for t in ts_sdpa]))
            speedup = dc_sdpa / dc_base if dc_base > 0 else 0
            pf_base = float(np.median([t['prefill_tps'] for t in ts_base]))
            pf_sdpa = float(np.median([t['prefill_tps'] for t in ts_sdpa]))
            print(f"    Step {s}: decode {dc_base:.1f} → {dc_sdpa:.1f} tok/s "
                  f"({speedup:.2f}x) | prefill {pf_base:.0f} → {pf_sdpa:.0f} tok/s")

    # ── Accuracy comparison ──
    print(f"\n{'='*80}")
    print(f"  ACCURACY: SDPA patch should not change outputs")
    print(f"{'='*80}")

    for base in base_configs:
        sdpa_key = base + '+sdpa'
        if base not in all_results or sdpa_key not in all_results:
            continue
        mismatches = 0
        for s in STEPS:
            a1 = all_results[base][s]['action']
            a2 = all_results[sdpa_key][s]['action']
            if a1 != a2:
                mismatches += 1
                c1 = all_results[base][s]['coords']
                c2 = all_results[sdpa_key][s]['coords']
                if c1 and c2:
                    diff = (abs(c1[0]-c2[0]), abs(c1[1]-c2[1]))
                    close = all(d <= 5 for d in diff)
                    tag = f"{'≈' if close else '≠'} diff=({diff[0]},{diff[1]})"
                else:
                    tag = "≠"
                print(f"  {cfg_labels[base]} Step {s}: {tag}")
                print(f"    base: {a1}")
                print(f"    sdpa: {a2}")
        if mismatches == 0:
            print(f"  {cfg_labels[base]}: all {len(STEPS)} steps IDENTICAL ✅")

    # ── Overall Summary ──
    print(f"\n{'='*80}")
    print(f"  OVERALL DECODE tok/s (median across all steps)")
    print(f"{'='*80}")
    for k in cfg_keys:
        all_decode = []
        for s in STEPS:
            ts = all_results[k][s]['timings']
            all_decode.append(float(np.median([t['decode_tps'] for t in ts])))
        med = np.median(all_decode)
        print(f"  {cfg_labels[k]:<22s}: {med:.1f} tok/s")

    # Save
    out_path = '/tmp/e2e_wxa16_sdpa_results.json'
    save_data = {}
    for k in cfg_keys:
        save_data[k] = {}
        for s in STEPS:
            r = all_results[k][s]
            save_data[k][str(s)] = {
                'action': r['action'],
                'coords': r['coords'],
                'prompt_tokens': r['prompt_tokens'],
                'timings_median': {
                    'prefill_ms': float(np.median([t['prefill_ms'] for t in r['timings']])),
                    'prefill_tps': float(np.median([t['prefill_tps'] for t in r['timings']])),
                    'decode_tps': float(np.median([t['decode_tps'] for t in r['timings']])),
                    'total_ms': float(np.median([t['total_ms'] for t in r['timings']])),
                },
            }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
