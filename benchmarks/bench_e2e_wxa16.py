#!/usr/bin/env python3
"""
End-to-end benchmark on REAL trajectory data:
  1. GPU FP16 baseline
  2. GPU W8A16 (native quantized_matmul)
  3. GPU W8A8 (INT8 TensorOps via cider fused_hybrid, prefill only)

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

FP16_MODEL  = '/Users/ws/Downloads/sft_baseline_v2'
W8A16_MODEL = '/Users/ws/Downloads/sft_baseline_v2_w8a16'
SESSION_DIR = os.path.join(ROOT_DIR, "session_data")
STEPS       = [0, 1, 2]  # step 0: 1img, step 1: 2imgs, step 2: 3imgs
MAX_TOKENS  = 200
PREFILL_STEP = 8192
N_SPEED_RUNS = 2  # per-step speed runs (fewer since we have multiple steps)


# ── W8A8 fused_hybrid bootstrap ─────────────────────────────────
def _init_w8a8():
    """Bootstrap cider fused_hybrid: fix ext path, return (convert_model_fused, set_mode)."""
    import cider.fused_hybrid as fh

    # Fix _ensure_ext: _w8a8_prim lives at ROOT/cider/_cider_prim, not ext/lib/_w8a8_prim
    if fh._w8a8_prim is None:
        try:
            fh._ensure_ext()
        except (ImportError, ModuleNotFoundError):
            # Fallback: use _cider_prim directly
            import _cider_prim
            fh._w8a8_prim = _cider_prim
            fh._KERNEL_DIR = os.path.join(ROOT_DIR, "cider", "kernels")

    return fh.convert_model_fused, fh.set_mode


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


def run_generate(model, processor, messages, pil_images, mode='default'):
    """Run one generation, return (text, timing_dict).

    mode: 'default' | 'split' | 'w8a8'
      - default: no patching, pure GPU
      - split: SplitLinear ANE+GPU hybrid (prefill only)
      - w8a8: fused_hybrid W8A8 INT8 TensorOps (prefill only)
    """
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

    # Pre-prefill mode switch
    if mode == 'split':
        from experimental.split_linear import SplitLinear
        SplitLinear.set_prefill(True)
    elif mode == 'w8a8':
        from cider.fused_hybrid import set_mode
        set_mode("prefill")

    text = ""
    prefill_time = 0
    decode_tps = 0
    prompt_tokens = 0
    gen_tokens = 0
    first_token = True

    for resp in custom_stream_generate(
        model, processor,
        prompt=prompt,
        image=pil_images,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        prefill_step_size=PREFILL_STEP,
        verbose=False,
    ):
        # Switch to decode mode after first token (end of prefill)
        if first_token:
            if mode == 'split':
                from experimental.split_linear import SplitLinear
                SplitLinear.set_prefill(False)
            elif mode == 'w8a8':
                from cider.fused_hybrid import set_mode
                set_mode("decode")
            first_token = False

        text += resp.text
        prefill_time = resp.prompt_tokens / resp.prompt_tps if resp.prompt_tps > 0 else 0
        decode_tps = resp.generation_tps
        prompt_tokens = resp.prompt_tokens
        gen_tokens = resp.generation_tokens

    # Ensure decode mode restored
    if mode == 'split':
        from experimental.split_linear import SplitLinear
        SplitLinear.set_prefill(False)
    elif mode == 'w8a8':
        from cider.fused_hybrid import set_mode
        set_mode("decode")

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
    print("=" * 70)
    print("  E2E Benchmark on Real Trajectory Data")
    print("  FP16 GPU  vs  W8A16 GPU  vs  W8A8 GPU (prefill only)")
    print("=" * 70)
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

    # configs: (key, model_path, mode, label)
    #   mode: 'default' | 'split' | 'w8a8'
    configs = [
        ('fp16',   FP16_MODEL,  'default', "GPU FP16"),
        ('w8a16',  W8A16_MODEL, 'default', "GPU W8A16"),
        ('w8a8',   W8A16_MODEL, 'w8a8',    "GPU W8A8"),
    ]

    all_results = {}

    for cfg_key, model_path, mode, label in configs:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        # Load model
        print(f"  Loading {model_path}...")
        model, proc = vlm_load(model_path)

        # Mode-specific model conversion
        if mode == 'split':
            from experimental.split_linear import patch_model
            bridge = patch_model(model, PREFILL_STEP, verbose=True)
        elif mode == 'w8a8':
            convert_model_fused, set_mode = _init_w8a8()
            stats = convert_model_fused(model, verbose=True)
            set_mode("decode")  # start in decode mode, run_generate switches

        cfg_results = {}

        for s in STEPS:
            data, images, msgs = step_data[s]
            print(f"\n  Step {s} ({len(images)} imgs):")

            # Warmup
            text, timing = run_generate(model, proc, msgs, images, mode=mode)
            print(f"    Warmup: {timing['prompt_tokens']} prompt tok, "
                  f"prefill {timing['prefill_ms']:.0f}ms, "
                  f"decode {timing['decode_tps']:.1f} tok/s")

            # Accuracy run (use warmup result)
            action = extract_action(text)
            coords = extract_coords(action)

            # Speed runs
            timings = []
            for r in range(N_SPEED_RUNS):
                _, t = run_generate(model, proc, msgs, images, mode=mode)
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
            if coords:
                print(f"    Coords: {coords}")

        all_results[cfg_key] = cfg_results

        # Free model before loading next
        del model, proc
        gc.collect()
        mx.clear_cache()

    # ── Accuracy Summary ──
    print(f"\n{'='*70}")
    print(f"  ACCURACY COMPARISON")
    print(f"{'='*70}")

    cfg_keys = [c[0] for c in configs]
    cfg_labels = {c[0]: c[3] for c in configs}

    for s in STEPS:
        print(f"\n  Step {s} (prompt={all_results[cfg_keys[0]][s]['prompt_tokens']} tok):")
        for k in cfg_keys:
            r = all_results[k][s]
            print(f"    {cfg_labels[k]:15s}: {r['action'] or '(no action)'}")

        # Pairwise coord comparison
        for i in range(len(cfg_keys)):
            for j in range(i+1, len(cfg_keys)):
                k1, k2 = cfg_keys[i], cfg_keys[j]
                c1 = all_results[k1][s]['coords']
                c2 = all_results[k2][s]['coords']
                a1 = all_results[k1][s]['action']
                a2 = all_results[k2][s]['action']
                if a1 == a2:
                    tag = "✅ IDENTICAL"
                elif c1 and c2:
                    diff = (abs(c1[0]-c2[0]), abs(c1[1]-c2[1]))
                    close = all(d <= 5 for d in diff)
                    tag = f"{'✅' if close else '⚠️'} diff=({diff[0]},{diff[1]})"
                else:
                    tag = "⚠️ DIFFERENT"
                print(f"      {cfg_labels[k1]} vs {cfg_labels[k2]}: {tag}")

    # ── Speed Summary ──
    print(f"\n{'='*70}")
    print(f"  SPEED SUMMARY (median of {N_SPEED_RUNS} runs)")
    print(f"{'='*70}")

    header = f"  {'Step':>4s} {'PrTok':>5s}"
    for k in cfg_keys:
        header += f" | {cfg_labels[k]:>15s}"
    print(header)
    print(f"  {'─' * (12 + 18 * len(cfg_keys))}")

    total_by_cfg = {k: [] for k in cfg_keys}

    for s in STEPS:
        line = f"  {s:>4d} {all_results[cfg_keys[0]][s]['prompt_tokens']:>5d}"
        for k in cfg_keys:
            ts = all_results[k][s]['timings']
            med_total = float(np.median([t['total_ms'] for t in ts]))
            total_by_cfg[k].append(med_total)
            med_prefill = float(np.median([t['prefill_tps'] for t in ts]))
            med_decode = float(np.median([t['decode_tps'] for t in ts]))
            line += f" | {med_total:7.0f}ms {med_decode:5.1f}d"
        print(line)

    # Aggregate
    print(f"\n  {'Aggregate':>10s}", end="")
    for k in cfg_keys:
        avg = np.mean(total_by_cfg[k])
        print(f" | {cfg_labels[k]:>15s}: {avg:.0f}ms avg", end="")
    print()

    # Speedup vs FP16
    if 'fp16' in total_by_cfg:
        fp16_avg = np.mean(total_by_cfg['fp16'])
        print(f"\n  Speedup vs FP16:")
        for k in cfg_keys:
            if k == 'fp16':
                continue
            avg = np.mean(total_by_cfg[k])
            print(f"    {cfg_labels[k]:15s}: {fp16_avg/avg:.2f}x overall")

        print(f"\n  Per-step speedup vs FP16:")
        for s in STEPS:
            fp16_t = total_by_cfg['fp16'][STEPS.index(s)]
            parts = [f"Step {s}:"]
            for k in cfg_keys:
                if k == 'fp16':
                    continue
                t = total_by_cfg[k][STEPS.index(s)]
                parts.append(f"{cfg_labels[k]}={fp16_t/t:.2f}x")
            print(f"    {' | '.join(parts)}")

    # Speedup vs W8A16 (most relevant comparison for W8A8)
    if 'w8a16' in total_by_cfg and 'w8a8' in total_by_cfg:
        print(f"\n  Speedup W8A8 vs W8A16:")
        for s in STEPS:
            w8a16_t = total_by_cfg['w8a16'][STEPS.index(s)]
            w8a8_t = total_by_cfg['w8a8'][STEPS.index(s)]
            print(f"    Step {s}: {w8a16_t/w8a8_t:.2f}x")
        w16_avg = np.mean(total_by_cfg['w8a16'])
        w8_avg = np.mean(total_by_cfg['w8a8'])
        print(f"    Overall: {w16_avg/w8_avg:.2f}x")

    # Decode tok/s
    print(f"\n  Decode tok/s (median across all steps):")
    for k in cfg_keys:
        all_decode = []
        for s in STEPS:
            ts = all_results[k][s]['timings']
            all_decode.append(float(np.median([t['decode_tps'] for t in ts])))
        med = np.median(all_decode)
        print(f"    {cfg_labels[k]:15s}: {med:.1f} tok/s")

    print(f"\n{'='*70}")

    # Save
    out_path = '/tmp/e2e_wxa16_results.json'
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
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
