#!/usr/bin/env python3
"""Unified PPL evaluation: FP16, W8A16, Cider per-channel, per-group gs=64, per-group gs=128."""
import argparse, math, sys, time, gc, os
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tqdm
from mlx_lm.utils import get_total_parameters, load
from cider.nn import CiderLinear

home_dir = os.path.expanduser("~")
WIKITEXT_LOCAL = os.path.join(home_dir, "Downloads/wikitext/wikitext-2-raw-v1")

FP16_MODEL = os.path.join(home_dir, "Downloads/Meta-Llama-3-8B")
W8A16_MODEL = os.path.join(home_dir, "Downloads/Llama3-8B-GPTQ-W8A16-mlx")

def load_wikitext_local(tokenizer, split="test"):
    from datasets import load_dataset
    split_map = {
        "test": f"{WIKITEXT_LOCAL}/test-00000-of-00001.parquet",
    }
    data = load_dataset("parquet", data_files={split: split_map[split]}, split=split)
    encodings = tokenizer.encode("\n\n".join(data["text"]), return_tensors="pt")
    return encodings.numpy()

def convert_model_with_gs(model, target_group_size):
    counter = [0]
    _TARGET = (nn.Linear, nn.QuantizedLinear)
    def _walk(module):
        for name, child in module.children().items():
            if isinstance(child, _TARGET):
                setattr(module, name, CiderLinear.from_float(child, target_group_size=target_group_size))
                counter[0] += 1
                if counter[0] % 28 == 0: gc.collect()
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, _TARGET):
                        child[i] = CiderLinear.from_float(item, target_group_size=target_group_size)
                        counter[0] += 1
                        if counter[0] % 28 == 0: gc.collect()
                    elif isinstance(item, nn.Module):
                        _walk(item)
            elif isinstance(child, nn.Module):
                _walk(child)
    _walk(model)
    return counter[0]

def eval_ppl(model, data, n_samples, seq_length):
    all_losses = []
    total_chunks = data.shape[1] // seq_length
    n = n_samples if n_samples > 0 else total_chunks
    for i in tqdm.tqdm(range(n), desc="Evaluating PPL"):
        batch = mx.array(data[:, i * seq_length: (i + 1) * seq_length])
        logits = model(batch[:, :-1]).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        mx.eval(losses)
        all_losses.append(losses.flatten())
    all_losses = mx.concatenate(all_losses)
    mean_loss = all_losses.mean().item()
    ppl = math.exp(mean_loss)
    std_dev = mx.sqrt(mx.var(all_losses, ddof=1)).item()
    se_ppl = ppl * (std_dev / math.sqrt(all_losses.size))
    return ppl, se_ppl

def run_config(config_name, model_path, cider_gs=None, data=None, n_samples=50, seq_length=2048):
    """Run a single config and return results dict."""
    print(f"\n{'='*60}")
    print(f"  {config_name}")
    print(f"{'='*60}")
    
    print(f"  Loading {model_path}...")
    model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
    
    if cider_gs is not None:
        gs_str = "per-channel" if cider_gs == 0 else f"per-group(gs={cider_gs})"
        print(f"  Converting to CiderLinear ({gs_str})...")
        t0 = time.perf_counter()
        n = convert_model_with_gs(model, cider_gs)
        print(f"  {n} layers converted in {time.perf_counter()-t0:.1f}s")
    
    # Warmup
    dummy = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _ = model(dummy); mx.eval(_)
    
    if data is None:
        data = load_wikitext_local(tokenizer, "test")
    
    mx.reset_peak_memory()
    start = time.time()
    ppl, se = eval_ppl(model, data, n_samples, seq_length)
    elapsed = time.time() - start
    peak_mem = mx.get_peak_memory() / 1e9
    
    result = {"name": config_name, "ppl": ppl, "se": se, "time": elapsed, "mem": peak_mem}
    print(f"  PPL = {ppl:.3f} ± {se:.3f} | {elapsed:.1f}s | {peak_mem:.2f} GB")
    
    # Free memory
    del model
    gc.collect()
    mx.metal.clear_cache()
    
    return result, data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()
    
    np.random.seed(123)
    mx.random.seed(123)
    
    configs = [
        ("Baseline FP16", FP16_MODEL, None),
        ("Baseline W8A16 (MLX native)", W8A16_MODEL, None),
        ("Cider W8A8 per-channel", W8A16_MODEL, 0),
        ("Cider W8A8 per-group(gs=64)", W8A16_MODEL, 64),
        ("Cider W8A8 per-group(gs=128)", W8A16_MODEL, 128),
    ]
    
    results = []
    data = None
    
    for name, model_path, cider_gs in configs:
        np.random.seed(123)
        mx.random.seed(123)
        r, data = run_config(name, model_path, cider_gs, data, args.num_samples, args.sequence_length)
        results.append(r)
    
    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  PPL COMPARISON SUMMARY (Qwen3-8B, wikitext-2, seq={args.sequence_length}, n={args.num_samples})")
    print(f"{'='*70}")
    print(f"  {'Config':<30} {'PPL':>10} {'±SE':>8} {'Time':>8} {'Mem':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    
    baseline_ppl = results[0]["ppl"] if results else None
    for r in results:
        delta = f"(+{(r['ppl']-baseline_ppl)/baseline_ppl*100:.2f}%)" if baseline_ppl and r['ppl'] != baseline_ppl else ""
        print(f"  {r['name']:<30} {r['ppl']:>8.3f}  ±{r['se']:.3f} {r['time']:>6.1f}s {r['mem']:>6.2f}GB {delta}")

if __name__ == "__main__":
    main()
