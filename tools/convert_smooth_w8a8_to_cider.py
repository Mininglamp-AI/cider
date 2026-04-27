#!/usr/bin/env python3
"""Convert llmcompressor SmoothQuant W8A8 → cider-ready MLX format.

Output format:
  - weight: fp16 [out, in] (dequantized, for mlx_lm to load + decode path)
  - cider_w8: int8 [in, out] (transposed, for cider W8A8 prefill)
  - cider_s8: float32 [out] (per-column scale for cider kernel)
  - Other tensors: fp16 passthrough

Usage:
  python convert_smooth_w8a8_to_cider.py SRC DST [--verify]
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

    # Copy aux files
    for f in src_dir.iterdir():
        if f.suffix == ".safetensors" or f.name in {"model.safetensors.index.json", "config.json"}:
            continue
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)

    # Load all shards
    st_files = sorted(glob.glob(str(src_dir / "model*.safetensors")))
    all_tensors = {}
    for st_file in st_files:
        fname = os.path.basename(st_file)
        print(f"  Loading {fname}...")
        for key, tensor in load_file(st_file).items():
            all_tensors[key] = (tensor, fname)

    scale_bases = {k.removesuffix(".weight_scale") for k in all_tensors if k.endswith(".weight_scale")}
    print(f"  {len(scale_bases)} quantized layers")

    shard_out = {}
    weight_map = {}
    n_converted = 0

    for key, (tensor, fname) in sorted(all_tensors.items()):
        if key.endswith(".weight_scale"):
            continue  # handled with weight

        base = key.removesuffix(".weight")
        if key.endswith(".weight") and base in scale_bases:
            w_int8 = tensor  # [out, in] int8
            scale = all_tensors[f"{base}.weight_scale"][0]  # [out, 1] bf16

            # 1. Dequant → fp16 weight for mlx_lm load + decode
            w_fp = (w_int8.to(torch.float32) * scale.to(torch.float32)).to(torch.float16)
            shard_out.setdefault(fname, {})[key] = w_fp

            # 2. Cider INT8: transpose [out,in]→[in,out], scale [out,1]→[out]
            w_t = w_int8.t().contiguous()
            s_flat = scale.squeeze(-1).to(torch.float32)
            shard_out[fname][f"{base}.cider_w8"] = w_t
            shard_out[fname][f"{base}.cider_s8"] = s_flat

            weight_map[key] = fname
            weight_map[f"{base}.cider_w8"] = fname
            weight_map[f"{base}.cider_s8"] = fname
            n_converted += 1
        else:
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            shard_out.setdefault(fname, {})[key] = tensor
            weight_map[key] = fname

    for fname, tensors in sorted(shard_out.items()):
        save_file(tensors, str(dst_dir / fname))
        print(f"  Saved {fname}: {len(tensors)} tensors")

    with open(dst_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2)

    new_config = dict(config)
    new_config.pop("quantization_config", None)
    new_config["cider_w8a8"] = True
    with open(dst_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    print(f"\n✅ {n_converted} layers → cider W8A8 + fp16 decode")
    return {"converted": n_converted}


def verify(dst_dir, n_check=3):
    """Spot-check: dequant(cider_w8 * cider_s8) ≈ weight."""
    from safetensors.torch import load_file
    import torch
    dst_dir = Path(dst_dir)
    all_t = {}
    for f in sorted(glob.glob(str(dst_dir / "model*.safetensors"))):
        all_t.update(load_file(f))

    checked = 0
    for key in sorted(all_t):
        if not key.endswith(".cider_w8") or checked >= n_check:
            continue
        base = key.removesuffix(".cider_w8")
        w8 = all_t[key]           # [in, out] int8
        s8 = all_t[f"{base}.cider_s8"]  # [out] float32
        w_fp = all_t[f"{base}.weight"]   # [out, in] fp16

        # Reconstruct: (w8.T * s8) should ≈ w_fp
        recon = (w8.t().to(torch.float32) * s8.unsqueeze(1).to(torch.float32))
        ref = w_fp.to(torch.float32)
        max_diff = (recon - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(recon.flatten().unsqueeze(0),
                                                      ref.flatten().unsqueeze(0)).item()
        status = "✅" if cos > 0.9999 else "❌"
        print(f"  {status} {base}: max_diff={max_diff:.2e}, cos={cos:.6f}")
        checked += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    convert(args.src, args.dst)
    if args.verify:
        print("\n--- Verify ---")
        verify(args.dst)
