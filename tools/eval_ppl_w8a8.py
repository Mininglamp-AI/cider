#!/usr/bin/env python3
"""
Evaluate PPL with Cider W8A8 INT8 TensorOps on prefill.

Uses the same evaluation logic as main_eval.py, but patches the model
with CiderLinear (W8A8 for seq_len>1, original for seq_len==1).
Since PPL evaluation is all prefill (seq_len=2048), every forward pass
goes through the W8A8 path.

Usage:
    python eval_ppl_w8a8.py --model /path/to/mlx-w8a16-model
"""
import argparse
import math
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tqdm

from mlx_lm.utils import get_total_parameters, load

# Add cider to path
from cider.convert import convert_model

WIKITEXT_LOCAL = "~/Downloads/wikitext/wikitext-2-raw-v1"


def load_wikitext_local(tokenizer, split="test"):
    from datasets import load_dataset
    split_map = {
        "test":  f"{WIKITEXT_LOCAL}/test-00000-of-00001.parquet",
        "train": f"{WIKITEXT_LOCAL}/train-00000-of-00001.parquet",
        "validation": f"{WIKITEXT_LOCAL}/validation-00000-of-00001.parquet",
    }
    data = load_dataset("parquet", data_files={split: split_map[split]}, split=split)
    encodings = tokenizer.encode("\n\n".join(data["text"]), return_tensors="pt")
    return encodings.numpy()


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

    print()
    all_losses = mx.concatenate(all_losses)
    mean_loss = all_losses.mean().item()
    ppl = math.exp(mean_loss)
    std_dev = mx.sqrt(mx.var(all_losses, ddof=1)).item()
    num_tokens = all_losses.size
    se_ppl = ppl * (std_dev / math.sqrt(num_tokens))
    return ppl, se_ppl


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPL with Cider W8A8")
    parser.add_argument("--model", type=str,
                        default="~/Downloads/Qwen3-8B-GPTQ-W8A16-mlx")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print(f"Loading model from {args.model}...")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    total_params = get_total_parameters(model)
    print(f"Model loaded: {total_params/1e6:.1f}M parameters")

    # Convert to CiderLinear (W8A8 on prefill, original on decode)
    print("\nConverting to CiderLinear (W8A8)...")
    stats = convert_model(model)
    print(f"  {stats['n_converted']} layers converted in {stats['elapsed_s']:.1f}s")

    # Warm up one forward pass to ensure INT8 weights are materialized
    print("Warming up W8A8 weights...")
    dummy = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _ = model(dummy)
    mx.eval(_)
    print("  W8A8 ready.\n")

    # Load data
    print("Loading dataset...")
    print(f"  Sequence length: {args.sequence_length}")
    data = load_wikitext_local(tokenizer, "test")
    print(f"  Loaded {len(data)} samples")

    # Evaluate
    print(f"\nEvaluating PPL with W8A8 prefill...")
    start_time = time.time()
    ppl, se = eval_ppl(model, data, args.num_samples, args.sequence_length)
    eval_time = time.time() - start_time
    tokens_evaluated = data.shape[0] * (data.shape[1] - 1)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (W8A8)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Mode: Cider W8A8 (prefill INT8 TensorOps)")
    print(f"Perplexity: {ppl:.3f} ± {se:.3f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"Tokens per second: {tokens_evaluated / eval_time:.0f}")

    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Total tokens: {data.size}")


if __name__ == "__main__":
    main()
