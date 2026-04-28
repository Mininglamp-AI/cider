#!/usr/bin/env python3
"""Convert LLMCompressor quantized model → MLX fp16 format.

Supported input formats (auto-detected):
  1. pack-quantized: .weight_packed (int32) + .weight_scale + .weight_shape
  2. raw int8: .weight (int8) + .weight_scale
  3. Already fp16/bf16 (passthrough with key remap only)

Key remapping (auto-detected):
  If keys start with "model.language_model.*" / "model.visual.*" (HF Qwen3VL):
    model.language_model.X  →  language_model.model.X
    model.visual.X          →  vision_tower.X
    lm_head.weight          →  dropped (tied weight)
  If keys already match MLX format: no remapping needed.

Usage:
  python convert_smooth_to_mlx.py /path/to/source /path/to/output [--verify]
"""
import argparse, json, os, shutil, glob
import numpy as np
from pathlib import Path


# ─── Key Remapping ──────────────────────────────────────────────────────────────

REMAP_RULES = [
    # (src_prefix, dst_prefix)
    ("model.language_model.", "language_model.model."),
    ("model.visual.", "vision_tower."),
]

DROP_KEYS = {"lm_head.weight"}


def detect_key_format(keys):
    """Detect whether keys need remapping."""
    for k in keys:
        if k.startswith("model.language_model.") or k.startswith("model.visual."):
            return "hf_qwen3vl"
        if k.startswith("language_model.") or k.startswith("vision_tower."):
            return "mlx_native"
    return "unknown"


def remap_key(key, fmt):
    """Remap key based on detected format. Returns None if key should be dropped."""
    if key in DROP_KEYS:
        return None

    if fmt == "hf_qwen3vl":
        for src_prefix, dst_prefix in REMAP_RULES:
            if key.startswith(src_prefix):
                return dst_prefix + key[len(src_prefix):]
        # Unknown prefix in HF format → keep as-is (shouldn't happen)
        return key
    else:
        # Already MLX native or unknown → no remap
        return key


# ─── Unpacking ──────────────────────────────────────────────────────────────────

def unpack_int8_from_int32(packed, original_shape):
    """Unpack 4 x int8 values packed into int32 (little-endian byte order)."""
    import torch
    out_features = packed.shape[0]
    target_out, target_in = int(original_shape[0]), int(original_shape[1])
    assert out_features == target_out

    raw_bytes = packed.contiguous().numpy().view(np.int8)
    raw_bytes = raw_bytes.reshape(out_features, -1)[:, :target_in]
    return torch.from_numpy(raw_bytes.copy()).to(torch.int8)


# ─── Shard Discovery ───────────────────────────────────────────────────────────

def get_shard_files(src_dir):
    """Get shard files from index.json. Falls back to model-*-of-* glob."""
    src_dir = Path(src_dir)
    idx_path = src_dir / "model.safetensors.index.json"

    if idx_path.exists():
        with open(idx_path) as f:
            wm = json.load(f).get("weight_map", {})
        shard_names = sorted(set(wm.values()))
        shard_files = [str(src_dir / n) for n in shard_names if (src_dir / n).exists()]
        if shard_files:
            return shard_files

    # Fallback: only sharded files (avoid stale single model.safetensors)
    shards = sorted(glob.glob(str(src_dir / "model-*-of-*.safetensors")))
    if shards:
        return shards

    # Last resort: single file
    single = src_dir / "model.safetensors"
    if single.exists():
        return [str(single)]

    raise FileNotFoundError(f"No safetensors files found in {src_dir}")


# ─── Main Conversion ───────────────────────────────────────────────────────────

def convert(src_dir, dst_dir):
    from safetensors.torch import load_file, save_file
    import torch

    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with open(src_dir / "config.json") as f:
        config = json.load(f)

    qconfig = config.get("quantization_config", {})
    fmt = qconfig.get("format", "none")
    print(f"  Source quantization format: {fmt}")

    # Copy non-safetensor files (tokenizer, etc.)
    for item in src_dir.iterdir():
        if item.suffix == ".safetensors" or item.name in {"model.safetensors.index.json", "config.json"}:
            continue
        if item.is_file():
            shutil.copy2(item, dst_dir / item.name)

    # Load tensors
    st_files = get_shard_files(src_dir)
    all_tensors = {}
    for st_file in st_files:
        print(f"  Loading {os.path.basename(st_file)}...")
        for key, tensor in load_file(st_file).items():
            all_tensors[key] = tensor

    print(f"  Loaded {len(all_tensors)} tensors from {len(st_files)} shard(s)")

    # Detect key format
    key_fmt = detect_key_format(all_tensors.keys())
    print(f"  Key format: {key_fmt}")

    # ─── Identify quantized layers ───
    # Case 1: pack-quantized (weight_packed + weight_scale + weight_shape)
    packed_bases = set()
    for k in all_tensors:
        if k.endswith(".weight_packed"):
            packed_bases.add(k.removesuffix(".weight_packed"))

    # Case 2: raw int8 (weight is int8 + weight_scale exists)
    raw_int8_bases = set()
    for k in all_tensors:
        if k.endswith(".weight_scale"):
            base = k.removesuffix(".weight_scale")
            if base not in packed_bases:
                w_key = f"{base}.weight"
                if w_key in all_tensors and all_tensors[w_key].dtype == torch.int8:
                    raw_int8_bases.add(base)

    print(f"  Pack-quantized layers: {len(packed_bases)}")
    print(f"  Raw INT8 layers: {len(raw_int8_bases)}")

    # Mark all auxiliary keys
    aux_keys = set()
    for base in packed_bases:
        aux_keys.add(f"{base}.weight_packed")
        aux_keys.add(f"{base}.weight_scale")
        aux_keys.add(f"{base}.weight_shape")
        aux_keys.add(f"{base}.weight")  # may not exist
    for base in raw_int8_bases:
        aux_keys.add(f"{base}.weight")
        aux_keys.add(f"{base}.weight_scale")

    # ─── Build output ───
    output_tensors = {}
    n_dequant = 0
    n_passthrough = 0
    n_dropped = 0

    # 1) Dequant pack-quantized
    for base in sorted(packed_bases):
        packed = all_tensors[f"{base}.weight_packed"]
        scale = all_tensors[f"{base}.weight_scale"]
        shape_t = all_tensors[f"{base}.weight_shape"]
        original_shape = shape_t.tolist()

        w_int8 = unpack_int8_from_int32(packed, original_shape)
        w_fp = (w_int8.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)

        out_key = remap_key(f"{base}.weight", key_fmt)
        if out_key is not None:
            output_tensors[out_key] = w_fp
            n_dequant += 1
        else:
            n_dropped += 1

    # 2) Dequant raw int8
    for base in sorted(raw_int8_bases):
        w = all_tensors[f"{base}.weight"]
        scale = all_tensors[f"{base}.weight_scale"]
        w_fp = (w.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)

        out_key = remap_key(f"{base}.weight", key_fmt)
        if out_key is not None:
            output_tensors[out_key] = w_fp
            n_dequant += 1
        else:
            n_dropped += 1

    # 3) Passthrough everything else
    for key in sorted(all_tensors.keys()):
        if key in aux_keys:
            continue

        out_key = remap_key(key, key_fmt)
        if out_key is None:
            n_dropped += 1
            continue

        tensor = all_tensors[key]
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        elif tensor.dtype == torch.float32:
            tensor = tensor.to(torch.bfloat16)

        output_tensors[out_key] = tensor
        n_passthrough += 1

    if n_dequant % 50 != 0:
        print(f"    Dequantized {n_dequant} layers total")

    # ─── Save ───
    out_file = "model.safetensors"
    save_file(output_tensors, str(dst_dir / out_file))
    print(f"  Saved {out_file}: {len(output_tensors)} tensors")

    weight_map = {k: out_file for k in sorted(output_tensors.keys())}
    with open(dst_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2)

    # Write clean config
    new_config = dict(config)
    new_config.pop("quantization_config", None)
    with open(dst_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    print(f"\n  ✓ Summary: {n_dequant} dequantized, {n_passthrough} passthrough, {n_dropped} dropped")
    print(f"  Output sample:")
    for k in sorted(output_tensors.keys())[:5]:
        print(f"    {k}: {output_tensors[k].dtype} {list(output_tensors[k].shape)}")

    return {"dequantized": n_dequant, "passthrough": n_passthrough, "dropped": n_dropped}


# ─── Verification ──────────────────────────────────────────────────────────────

def verify(src_dir, dst_dir, n_check=5):
    """Spot-check dequant accuracy."""
    from safetensors.torch import load_file
    import torch

    src_all = {}
    for f in get_shard_files(src_dir):
        src_all.update(load_file(f))
    dst_all = {}
    for f in sorted(glob.glob(str(Path(dst_dir) / "*.safetensors"))):
        dst_all.update(load_file(f))

    key_fmt = detect_key_format(src_all.keys())
    checked = 0

    # Check pack-quantized
    for k in sorted(src_all):
        if checked >= n_check:
            break
        if not k.endswith(".weight_packed"):
            continue
        base = k.removesuffix(".weight_packed")
        packed = src_all[k]
        scale = src_all[f"{base}.weight_scale"]
        shape_t = src_all[f"{base}.weight_shape"]
        w_int8 = unpack_int8_from_int32(packed, shape_t.tolist())
        ref = (w_int8.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)

        out_key = remap_key(f"{base}.weight", key_fmt)
        if out_key and out_key in dst_all:
            out = dst_all[out_key]
            diff = (ref.float() - out.float()).abs().max().item()
            status = "✓" if diff == 0 else ("~" if diff < 0.01 else "✗")
            print(f"  {status} {out_key}: shape={list(out.shape)} max_diff={diff:.2e}")
        checked += 1

    # Check raw int8
    for k in sorted(src_all):
        if checked >= n_check:
            break
        if not k.endswith(".weight_scale"):
            continue
        base = k.removesuffix(".weight_scale")
        if f"{base}.weight_packed" in src_all:
            continue
        w_key = f"{base}.weight"
        if w_key not in src_all or src_all[w_key].dtype != torch.int8:
            continue
        ref = (src_all[w_key].to(torch.float32) * src_all[k].to(torch.float32)).to(torch.bfloat16)
        out_key = remap_key(w_key, key_fmt)
        if out_key and out_key in dst_all:
            out = dst_all[out_key]
            diff = (ref.float() - out.float()).abs().max().item()
            status = "✓" if diff == 0 else ("~" if diff < 0.01 else "✗")
            print(f"  {status} {out_key}: shape={list(out.shape)} max_diff={diff:.2e}")
        checked += 1

    if checked == 0:
        print("  (no quantized layers to verify)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert LLMCompressor quantized → MLX fp16")
    p.add_argument("src", help="Source directory")
    p.add_argument("dst", help="Output directory")
    p.add_argument("--verify", action="store_true", help="Verify dequant accuracy")
    args = p.parse_args()

    stats = convert(args.src, args.dst)
    if args.verify:
        print("\n--- Verify ---")
        verify(args.src, args.dst)
    print(f"\nDone. Load with: mlx_lm.load('{args.dst}')")
