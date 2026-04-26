# Copyright © 2025 Apple Inc.
"""
Evaluate perplexity (PPL) of MLX models.
"""

import argparse
import math
import time
import tqdm
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from datasets import load_dataset
from mlx_lm.utils import get_total_parameters, load

from mlx_lm import convert

awq_model = "~/Downloads/Qwen3-8B-AWQ-W4A16/Qwen3-8B-AWQ-W4A16"

WIKITEXT_LOCAL = "~/Downloads/wikitext/wikitext-2-raw-v1"
def load_wikitext_local(tokenizer, split="test"):
    """Load wikitext-2-raw-v1 from local parquet files."""
    split_map = {
        "test":  f"{WIKITEXT_LOCAL}/test-00000-of-00001.parquet",
        "train": f"{WIKITEXT_LOCAL}/train-00000-of-00001.parquet",
        "validation": f"{WIKITEXT_LOCAL}/validation-00000-of-00001.parquet",
    }
    data = load_dataset("parquet", data_files={split: split_map[split]}, split=split)
    encodings = tokenizer.encode(
        "\n\n".join(data["text"]), return_tensors="pt"
    )

    return encodings.numpy()

def eval_ppl(model, data, n_samples, seq_length):
    """
    Evaluate perplexity on a dataset with standard error calculation.

    Args:
        model: The model to evaluate
        data: Tokenized data tensor
        batch_size: Batch size for evaluation

    Returns:
        tuple: (perplexity, standard_error)
    """
    all_losses = []

    total_chunks = data.shape[1] // seq_length
    n = n_samples if n_samples > 0 else total_chunks
    for i in tqdm.tqdm(range(n), desc="Evaluating PPL"):
        batch = mx.array(data[:, i * seq_length: (i + 1) * seq_length])
        # Forward pass: get logits for all tokens except last
        logits = model(batch[:, :-1]).astype(mx.float32)

        # Calculate cross-entropy loss with next tokens
        losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        mx.eval(losses)
        # Store individual token losses
        all_losses.append(losses.flatten())


    print()  # New line after progress

    # Concatenate all losses into a single array
    all_losses = mx.concatenate(all_losses)

    # Calculate mean loss and perplexity
    mean_loss = all_losses.mean().item()
    ppl = math.exp(mean_loss)
    # Calculate standard error
    std_dev = mx.sqrt(mx.var(all_losses, ddof=1)).item()
    num_tokens = all_losses.size
    standard_error = std_dev / math.sqrt(num_tokens)
    # Delta approximation for standard error of perplexity
    standard_error_ppl = ppl * standard_error

    return ppl, standard_error_ppl


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of MLX models")
    parser.add_argument(
        "--model",
        type=str,
        default="~/Downloads/Meta-Llama-3-8B",
        help="Path to model or Hugging Face model ID",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer/model loading from Hugging Face.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Sequence length for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to use (-1 for all available)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="~/Downloads/wikitext/wikitext-2-raw-v1",
        help="A Hugging Face dataset which is compatible with an mlx-lm dataset format.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for data sampling"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # Load model
    print(f"Loading model from {args.model}...")
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    model, tokenizer = load(args.model, tokenizer_config=tokenizer_config)

    # Count parameters
    total_params = get_total_parameters(model)
    print(f"Model loaded: {total_params/1e6:.1f}M parameters")

    # Load evaluation data
    print(f"\nLoading dataset...")
    print(f"  Sequence length: {args.sequence_length}")

    data = load_wikitext_local(
        tokenizer,
        "test"
    )

    print(f"  Loaded {len(data)} samples")

    # Evaluate perplexity
    print(f"\nEvaluating perplexity with batch size {args.batch_size}...")
    start_time = time.time()

    ppl, se = eval_ppl(model, data, args.num_samples, args.sequence_length)

    eval_time = time.time() - start_time
    tokens_evaluated = data.shape[0] * (data.shape[1] - 1)  # B * (L - 1)
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Perplexity: {ppl:.3f} ± {se:.3f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"Tokens per second: {tokens_evaluated / eval_time:.0f}")

    # Additional statistics
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Total tokens: {data.size}")


if __name__ == "__main__":
    main()
