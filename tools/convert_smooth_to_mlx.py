#!/usr/bin/env python3
"""Convert LLMCompressor SmoothQuant W8A8 (raw INT8) → MLX fp16 format.

SmoothQuant format:
  - layer.weight: INT8 [out, in] (raw, not packed)
  - layer.weight_scale: bfloat16 [out, 1] (per-channel)
  - Smooth factors absorbed into layernorm weights

Output (MLX-native fp16):
  - layer.weight: fp16 [out, in] (dequantized: int8 * scale)
  - Other tensors: fp16 passthrough

After conversion, mlx_lm.load() works directly.
Then cider convert_model() does online INT8 re-quantization for prefill.

Usage:
  python convert_smooth_to_mlx.py /path/to/Smooth-W8A8 /path/to/output [--verify]
"""
import argparse, json, os, shutil, glob
import numpy as np
from pathlib import Path


def convert(src_dir, dst_dir):
    from safetensors.torch import load_file, save_file
    import torch

    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with open(src_dir / "config.json") as f:
        config = json.load(f)

    qconfig = config.get("quantization_config", {})
    ignore = set(qconfig.get("ignore", []))
    print(f"Ignore: {ignore}")

    # Copy non-safetensor files
    for f in src_dir.iterdir():
        if f.suffix == ".safetensors" or f.name in {"model.safetensors.index.json", "config.json"}:
            continue
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)

    # Load all shards
    st_files = sorted(glob.glob(str(src_dir / "model*.safetensors")))
    if not st_files:
        st_files = sorted(glob.glob(str(src_dir / "*.safetensors")))
    all_tensors = {}
    for st_file in st_files:
        fname = os.path.basename(st_file)
        print(f"  Loading {fname}...")
        for key, tensor in load_file(st_file).items():
            all_tensors[key] = (tensor, fname)

    # Find quantized layers (have weight_scale)
    scale_bases = {k.removesuffix(".weight_scale") for k in all_tensors if k.endswith(".weight_scale")}
    print(f"  {len(scale_bases)} quantized layers (raw INT8 + per-channel scale)")

    shard_out = {}
    weight_map = {}
    n_converted = 0
    n_passthrough = 0

    for key, (tensor, fname) in sorted(all_tensors.items()):
        if key.endswith(".weight_scale"):
            continue

        base = key.removesuffix(".weight")
        if key.endswith(".weight") and base in scale_bases:
            w_int8 = tensor  # [out, in] int8
            scale = all_tensors[f"{base}.weight_scale"][0]  # [out, 1] bf16

            assert w_int8.dtype == torch.int8, f"{key}: expected int8, got {w_int8.dtype}"

            # Dequantize: fp16 = int8 * scale
            w_fp = (w_int8.to(torch.float32) * scale.to(torch.float32)).to(torch.float16)

            shard_out.setdefault(fname, {})[key] = w_fp
            weight_map[key] = fname
            n_converted += 1
            if n_converted % 50 == 0:
                print(f"    Dequantized {n_converted} layers...")
        else:
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            shard_out.setdefault(fname, {})[key] = tensor
            weight_map[key] = fname
            n_passthrough += 1

    # Save
    for fname, tensors in sorted(shard_out.items()):
        save_file(tensors, str(dst_dir / fname))
        print(f"  Saved {fname}: {len(tensors)} tensors")

    with open(dst_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2)

    # Clean config (remove quantization_config)
    new_config = dict(config)
    new_config.pop("quantization_config", None)
    with open(dst_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    print(f"\n  {n_converted} layers dequantized to fp16, {n_passthrough} passthrough")
    return {"converted": n_converted, "passthrough": n_passthrough}


def verify(src_dir, dst_dir, n_check=3):
    """Verify dequant(int8 * scale) == output fp16 weight."""
    from safetensors.torch import load_file
    import torch

    src_all, dst_all = {}, {}
    for f in sorted(glob.glob(str(Path(src_dir) / "model*.safetensors"))):
        src_all.update(load_file(f))
    for f in sorted(glob.glob(str(Path(dst_dir) / "model*.safetensors"))):
        dst_all.update(load_file(f))

    checked = 0
    for k in sorted(src_all):
        if not k.endswith(".weight_scale") or checked >= n_check:
            continue
        base = k.removesuffix(".weight_scale")
        w_key = f"{base}.weight"

        w_int8 = src_all[w_key]
        scale = src_all[k]
        ref = (w_int8.to(torch.float32) * scale.to(torch.float32))

        out = dst_all[w_key].to(torch.float32)
        max_diff = (ref - out).abs().max().item()
        print(f"  {base}: max_diff={max_diff:.2e} {'OK' if max_diff < 0.01 else 'FAIL'}")
        checked += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    stats = convert(args.src, args.dst)
    if args.verify:
        print("\n--- Verify ---")
        verify(args.src, args.dst)
    print(f"\n Done. Use with: mlx_lm.load('{args.dst}')")
