#!/usr/bin/env python3
"""
Convert LLMCompressor compressed-tensors quantized model → MLX-native format.

Supports any bit-width (W4A16, W8A16, etc.) produced by LLMCompressor's
pack-quantized format, including per-channel (strategy=channel) and per-group
(strategy=group) quantization.  Handles cross-shard layers automatically.

Usage:
  python convert_compressed_tensors_to_mlx.py /path/to/src /path/to/dst [--verify]
"""
import argparse
import json
import os
import shutil
import glob
import numpy as np
from pathlib import Path

_MLX_MAX_GROUP_SIZE = {4: 128, 8: 128}
_MLX_DEFAULT_GROUP_SIZE = 128


def _effective_group_size(num_bits, config_group_size, in_features):
    if config_group_size is not None and config_group_size > 0:
        return config_group_size
    max_gs = _MLX_MAX_GROUP_SIZE.get(num_bits, _MLX_DEFAULT_GROUP_SIZE)
    return min(max_gs, in_features)


def convert(src_dir: str, dst_dir: str) -> dict:
    from safetensors.torch import load_file, save_file
    import torch

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with open(src_dir / "config.json") as f:
        config = json.load(f)

    qconfig = config.get("quantization_config", {})
    group_cfg = qconfig.get("config_groups", {}).get("group_0", {}).get("weights", {})
    config_group_size = group_cfg.get("group_size", None)
    num_bits = group_cfg.get("num_bits", 4)
    symmetric = group_cfg.get("symmetric", True)
    strategy = group_cfg.get("strategy", "group")
    ignore = set(qconfig.get("ignore", []))

    pack_factor = 32 // num_bits
    offset = 1 << (num_bits - 1)
    per_channel = (strategy == "channel") or (config_group_size is None)

    print(f"Config: {num_bits}-bit, strategy={strategy}, "
          f"config_group_size={config_group_size}, "
          f"symmetric={symmetric}, offset={offset}, ignore={ignore}")

    # Copy auxiliary files
    skip_ext = {".safetensors"}
    skip_names = {"model.safetensors.index.json", "config.json"}
    for f in src_dir.iterdir():
        if f.suffix in skip_ext or f.name in skip_names:
            continue
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)

    # ── Load ALL shards, track which shard each key came from ────
    st_files = sorted(glob.glob(str(src_dir / "model*.safetensors")))
    if not st_files:
        st_files = sorted(glob.glob(str(src_dir / "*.safetensors")))

    # key → (tensor, source_filename)
    all_tensors = {}
    for st_file in st_files:
        fname = os.path.basename(st_file)
        print(f"  Loading {fname}...")
        for key, tensor in load_file(st_file).items():
            all_tensors[key] = (tensor, fname)

    # ── Classify: quantized vs plain ─────────────────────────────
    quant_layers = {}  # base → {packed, scale, shape}
    plain = {}         # key → (tensor, fname)

    for key, (tensor, fname) in all_tensors.items():
        matched = False
        for suffix, part in [(".weight_packed", "packed"),
                             (".weight_scale", "scale"),
                             (".weight_shape", "shape")]:
            if key.endswith(suffix):
                base = key.removesuffix(suffix)
                quant_layers.setdefault(base, {})[part] = (tensor, fname)
                matched = True
                break
        if not matched:
            plain[key] = (tensor, fname)

    # ── Convert ──────────────────────────────────────────────────
    # Output: per-shard tensors
    import torch
    shard_out = {}    # fname → {key: tensor}
    weight_map = {}
    total_converted = 0
    actual_group_size = None

    for base, parts in sorted(quant_layers.items()):
        if "packed" not in parts:
            print(f"  ⚠ Skipping {base}: no weight_packed")
            continue
        if "scale" not in parts:
            print(f"  ⚠ Skipping {base}: no weight_scale")
            continue

        packed, packed_fname = parts["packed"]
        scale, _ = parts["scale"]
        shape_tensor = parts.get("shape", (None, None))[0]

        if shape_tensor is not None:
            in_features = int(shape_tensor[1].item())
            out_features = int(shape_tensor[0].item())
        else:
            in_features = packed.shape[1] * pack_factor
            out_features = packed.shape[0]

        gs = _effective_group_size(num_bits, config_group_size, in_features)
        if actual_group_size is None:
            actual_group_size = gs
            print(f"  Using group_size={gs} "
                  f"({'broadcast from per-channel' if per_channel else 'from config'})")

        # packed → uint32 as-is
        mlx_weight = torch.from_numpy(packed.numpy().view(np.uint32).copy())

        # scale → fp16, broadcast if per-channel
        scale_fp16 = scale.to(torch.float16)
        n_groups = in_features // gs
        if scale_fp16.shape[1] < n_groups:
            scale_fp16 = scale_fp16.expand(out_features, n_groups).contiguous()

        mlx_scales = scale_fp16
        mlx_biases = -float(offset) * mlx_scales

        # Write to the shard that owns weight_packed
        shard_out.setdefault(packed_fname, {})[f"{base}.weight"] = mlx_weight
        shard_out[packed_fname][f"{base}.scales"] = mlx_scales
        shard_out[packed_fname][f"{base}.biases"] = mlx_biases
        total_converted += 1

    # Plain tensors
    total_passthrough = 0
    for key, (tensor, fname) in plain.items():
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)
        shard_out.setdefault(fname, {})[key] = tensor
        total_passthrough += 1

    # ── Save shards ──────────────────────────────────────────────
    for fname, tensors in sorted(shard_out.items()):
        out_path = dst_dir / fname
        save_file(tensors, str(out_path))
        print(f"  Saved {fname}: {len(tensors)} tensors")
        for key in tensors:
            weight_map[key] = fname

    # Index
    with open(dst_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2)

    # Config
    new_config = dict(config)
    new_config.pop("quantization_config", None)
    new_config["quantization"] = {
        "group_size": actual_group_size or _MLX_DEFAULT_GROUP_SIZE,
        "bits": num_bits,
    }
    with open(dst_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    stats = {"bits": num_bits, "group_size": actual_group_size,
             "strategy": strategy, "converted": total_converted,
             "passthrough": total_passthrough}
    print(f"\n✅ Done — {total_converted} quantized layers converted.")
    print(f"   Output: {dst_dir}")
    return stats


def verify(src_dir: str, dst_dir: str, n_check: int = 3):
    from safetensors.torch import load_file
    import torch

    try:
        import mlx.core as mx
    except ImportError:
        print("⚠ mlx not available — skipping verification")
        return

    src_dir, dst_dir = Path(src_dir), Path(dst_dir)

    with open(src_dir / "config.json") as f:
        src_cfg = json.load(f)
    qconfig = src_cfg.get("quantization_config", {})
    group_cfg = qconfig.get("config_groups", {}).get("group_0", {}).get("weights", {})
    config_group_size = group_cfg.get("group_size", None)
    num_bits = group_cfg.get("num_bits", 4)

    with open(dst_dir / "config.json") as f:
        dst_cfg = json.load(f).get("quantization", {})
    mlx_group_size = dst_cfg.get("group_size", 128)
    bits = dst_cfg.get("bits", num_bits)
    offset = 1 << (bits - 1)
    pack_factor = 32 // bits

    # Load all shards from both dirs
    src_all = {}
    for f in sorted(glob.glob(str(src_dir / "model*.safetensors"))):
        src_all.update(load_file(f))
    dst_all = {}
    for f in sorted(glob.glob(str(dst_dir / "model*.safetensors"))):
        dst_all.update(load_file(f))

    checked = 0
    for key in sorted(src_all):
        if not key.endswith(".weight_packed") or checked >= n_check:
            continue
        base = key.removesuffix(".weight_packed")

        packed = src_all[f"{base}.weight_packed"].numpy().view(np.uint32)
        scale = src_all[f"{base}.weight_scale"].to(torch.float32).numpy()
        shape_t = src_all.get(f"{base}.weight_shape")
        if shape_t is not None:
            out_feat, in_feat = int(shape_t[0].item()), int(shape_t[1].item())
        else:
            in_feat = packed.shape[1] * pack_factor
            out_feat = packed.shape[0]

        ref = np.zeros((out_feat, in_feat), dtype=np.float32)
        for i in range(pack_factor):
            vals = (packed >> (bits * i)) & ((1 << bits) - 1)
            signed = vals.astype(np.int32) - offset
            cols = np.arange(packed.shape[1]) * pack_factor + i
            cols = cols[cols < in_feat]
            if scale.shape[1] == 1:
                ref[:, cols] = signed[:, :len(cols)] * scale[:, 0:1]
            else:
                groups = cols // config_group_size
                ref[:, cols] = signed[:, :len(cols)] * scale[:, groups]

        w = mx.array(dst_all[f"{base}.weight"].numpy().view(np.uint32))
        s = mx.array(dst_all[f"{base}.scales"].to(torch.float32).numpy())
        b = mx.array(dst_all[f"{base}.biases"].to(torch.float32).numpy())
        mlx_deq = np.array(mx.dequantize(w, s, b, group_size=mlx_group_size, bits=bits))

        max_diff = np.max(np.abs(ref - mlx_deq))
        cos = (np.dot(ref.flatten(), mlx_deq.flatten())
               / (np.linalg.norm(ref.flatten()) * np.linalg.norm(mlx_deq.flatten()) + 1e-12))
        status = "✅" if cos > 0.9999 else "❌"
        print(f"  {status} {base}: max_diff={max_diff:.2e}, cos={cos:.8f}")
        checked += 1

    print(f"\n  Verified {checked} layers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LLMCompressor compressed-tensors → MLX format"
    )
    parser.add_argument("src", help="Source compressed-tensors model directory")
    parser.add_argument("dst", help="Destination MLX model directory")
    parser.add_argument("--verify", action="store_true",
                        help="Spot-check dequantized values after conversion")
    args = parser.parse_args()

    convert(args.src, args.dst)
    if args.verify:
        print("\n--- Verification ---")
        verify(args.src, args.dst)
