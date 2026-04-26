#!/usr/bin/env python3
"""Benchmark: FP16 vs W8A8 vs W8A8-Fused. Step 2 only (3455 tokens)."""
import sys, os, time, io, base64, types
from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)

FP16_MODEL  = '~/Downloads/sft_baseline_v2'
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.utils import load as vlm_load, prepare_inputs
from mlx_vlm.models import cache as cache_mod
from session_data.replay_prompt import build_prompt_at_step
from PIL import Image
from cider import ops
from mlx_vlm.models.qwen3_vl.language import (
    scaled_dot_product_attention,
    apply_multimodal_rotary_pos_emb,
)


def quantize_w8(fp_weight):
    wt = fp_weight.T
    s = mx.max(mx.abs(wt), axis=0) / 127.0
    s = mx.maximum(s, mx.array(1e-10))
    w8 = mx.clip(mx.round(wt / s), -128, 127).astype(mx.int8)
    mx.eval(w8, s)
    return w8, s


class W8A8Linear(nn.Module):
    def __init__(self, fp_linear):
        super().__init__()
        w = fp_linear.weight
        self._w8, self._s8 = quantize_w8(w)
        self._out = w.shape[0]
        self._in = w.shape[1]
        self.weight = w

    def __call__(self, x):
        orig = x.shape
        return ops.w8a8_linear(x.reshape(-1, self._in), self._w8, self._s8).reshape(
            *orig[:-1], self._out
        )


CS = 2048


def prefill_chunked(model_lm, embeds, ids):
    T = embeds.shape[1]
    cache = cache_mod.make_prompt_cache(model_lm)
    for start in range(0, T, CS):
        end = min(start + CS, T)
        model_lm(
            inputs=ids[:, start:end], inputs_embeds=embeds[:, start:end], cache=cache
        )
        mx.eval([c.state for c in cache])


def bench(label, model_lm, embeds, ids, n_warmup=3, n_iter=7):
    for _ in range(n_warmup):
        prefill_chunked(model_lm, embeds, ids)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        prefill_chunked(model_lm, embeds, ids)
        times.append((time.perf_counter() - t0) * 1000)
    med = np.median(times)
    print(f"  {label}: median={med:.0f}ms  all={[round(t) for t in times]}")
    return med


# ======= FP16 BASELINE =======
m, p = vlm_load(FP16_MODEL)
mx.eval(m.parameters())

# Prepare all 3 steps
all_steps = []
for step in range(3):
    data = build_prompt_at_step(os.path.join(ROOT_DIR, "session_data"), step)
    imgs = [Image.open(io.BytesIO(base64.b64decode(b))) for b in data["images"]]
    n = data["prompt"].count("<image>")
    parts = data["prompt"].split("<image>")
    content = []
    for j, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if j < n:
            content.append({"type": "image"})
    msgs = [
        {"role": "system", "content": data["system_prompt"]},
        {"role": "user", "content": content},
    ]
    pr = p.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = prepare_inputs(p, images=imgs, prompts=pr)
    ids = inputs["input_ids"]
    pv = inputs.get("pixel_values")
    mask_in = inputs.get("mask")
    kw = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "pixel_values", "mask")
    }
    eo = m.get_input_embeddings(ids, pv, mask=mask_in, **kw)
    embeds = eo.inputs_embeds
    mx.eval(embeds)
    all_steps.append((embeds, ids, embeds.shape[1]))
    print(f"Step {step}: {embeds.shape[1]} tokens")

print("\n=== FP16 Baseline ===")
fp_results = []
for step, (embeds, ids, ntok) in enumerate(all_steps):
    fp_results.append(bench(f"Step {step} ({ntok} tok)", m.language_model, embeds, ids))

# ======= W8A8 INDIVIDUAL =======
# Reload
m2, _ = vlm_load(FP16_MODEL)
mx.eval(m2.parameters())

# Prepare embeds with m2
all_steps2 = []
for step in range(3):
    data = build_prompt_at_step(os.path.join(ROOT_DIR, "session_data"), step)
    imgs = [Image.open(io.BytesIO(base64.b64decode(b))) for b in data["images"]]
    n = data["prompt"].count("<image>")
    parts = data["prompt"].split("<image>")
    content = []
    for j, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if j < n:
            content.append({"type": "image"})
    msgs = [
        {"role": "system", "content": data["system_prompt"]},
        {"role": "user", "content": content},
    ]
    pr = p.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = prepare_inputs(p, images=imgs, prompts=pr)
    ids = inputs["input_ids"]
    pv = inputs.get("pixel_values")
    mask_in = inputs.get("mask")
    kw = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "pixel_values", "mask")
    }
    eo = m2.get_input_embeddings(ids, pv, mask=mask_in, **kw)
    embeds = eo.inputs_embeds
    mx.eval(embeds)
    all_steps2.append((embeds, ids, embeds.shape[1]))

for layer in m2.language_model.model.layers:
    for nm in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        setattr(layer.self_attn, nm, W8A8Linear(getattr(layer.self_attn, nm)))
    for nm in ["gate_proj", "up_proj", "down_proj"]:
        setattr(layer.mlp, nm, W8A8Linear(getattr(layer.mlp, nm)))

print("\n=== W8A8 Individual (auto-cast) ===")
w8_results = []
for step, (embeds, ids, ntok) in enumerate(all_steps2):
    w8_results.append(
        bench(f"Step {step} ({ntok} tok)", m2.language_model, embeds, ids)
    )

# ======= W8A8 FUSED =======
m3, _ = vlm_load(FP16_MODEL)
mx.eval(m3.parameters())

all_steps3 = []
for step in range(3):
    data = build_prompt_at_step(os.path.join(ROOT_DIR, "session_data"), step)
    imgs = [Image.open(io.BytesIO(base64.b64decode(b))) for b in data["images"]]
    n = data["prompt"].count("<image>")
    parts = data["prompt"].split("<image>")
    content = []
    for j, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if j < n:
            content.append({"type": "image"})
    msgs = [
        {"role": "system", "content": data["system_prompt"]},
        {"role": "user", "content": content},
    ]
    pr = p.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = prepare_inputs(p, images=imgs, prompts=pr)
    ids = inputs["input_ids"]
    pv = inputs.get("pixel_values")
    mask_in = inputs.get("mask")
    kw = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "pixel_values", "mask")
    }
    eo = m3.get_input_embeddings(ids, pv, mask=mask_in, **kw)
    embeds = eo.inputs_embeds
    mx.eval(embeds)
    all_steps3.append((embeds, ids, embeds.shape[1]))

# Patch fused QKV + gate/up
for i, layer in enumerate(m3.language_model.model.layers):
    attn = layer.self_attn
    # Fused QKV
    wq = attn.q_proj.weight
    wk = attn.k_proj.weight
    wv = attn.v_proj.weight
    wt_cat = mx.concatenate([wq.T, wk.T, wv.T], axis=1)
    s = mx.max(mx.abs(wt_cat), axis=0) / 127.0
    s = mx.maximum(s, mx.array(1e-10))
    qkv_w8 = mx.clip(mx.round(wt_cat / s), -128, 127).astype(mx.int8)
    qkv_s8 = s
    mx.eval(qkv_w8, qkv_s8)
    dim_q, dim_k, dim_v = wq.shape[0], wk.shape[0], wv.shape[0]
    in_dim = wq.shape[1]

    # O proj W8A8
    o_w8, o_s8 = quantize_w8(attn.o_proj.weight)
    o_out = attn.o_proj.weight.shape[0]

    # Capture in closure
    _qkv_w8, _qkv_s8 = qkv_w8, qkv_s8
    _dim_q, _dim_k, _dim_v, _in_dim = dim_q, dim_k, dim_v, in_dim
    _o_w8, _o_s8, _o_out = o_w8, o_s8, o_out

    def make_attn_call(qw, qs, dq, dk, dv, ind, ow, os_, oo):
        def fused_attn_call(self, x, mask=None, cache=None, position_ids=None):
            B, L, D = x.shape
            out = ops.w8a8_linear(x.reshape(-1, ind), qw, qs)
            out = out.reshape(B, L, -1)
            queries = out[..., :dq]
            keys = out[..., dq : dq + dk]
            values = out[..., dq + dk :]

            queries = self.q_norm(
                queries.reshape(B, L, self.n_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(B, L, self.n_kv_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )

            kv_seq_len = keys.shape[-2]
            if position_ids is None:
                kv_seq_len += cache.offset + 1
                position_ids = mx.arange(cache.offset, cache.offset + L)
                position_ids = mx.expand_dims(position_ids, axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
            else:
                kv_seq_len += cache.offset + 1 if cache is not None else 0

            cos, sin = self.rotary_emb(values, position_ids)
            if mask is not None and isinstance(mask, mx.array):
                if isinstance(kv_seq_len, mx.array):
                    kv_seq_len = kv_seq_len.max().item()
                mask = mask[..., : int(kv_seq_len)]

            queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

            output = scaled_dot_product_attention(
                queries, keys, values, cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return ops.w8a8_linear(output.reshape(-1, oo), ow, os_).reshape(B, L, oo)

        return fused_attn_call

    attn.__call__ = types.MethodType(
        make_attn_call(
            _qkv_w8, _qkv_s8, _dim_q, _dim_k, _dim_v, _in_dim, _o_w8, _o_s8, _o_out
        ),
        attn,
    )

    # Fused Gate+Up
    mlp = layer.mlp
    wg = mlp.gate_proj.weight
    wu = mlp.up_proj.weight
    wt_gu = mx.concatenate([wg.T, wu.T], axis=1)
    s_gu = mx.max(mx.abs(wt_gu), axis=0) / 127.0
    s_gu = mx.maximum(s_gu, mx.array(1e-10))
    gu_w8 = mx.clip(mx.round(wt_gu / s_gu), -128, 127).astype(mx.int8)
    mx.eval(gu_w8, s_gu)
    dim_g = wg.shape[0]
    mlp_in = wg.shape[1]
    down_w8, down_s8 = quantize_w8(mlp.down_proj.weight)
    down_out = mlp.down_proj.weight.shape[0]
    down_in = mlp.down_proj.weight.shape[1]

    def make_mlp_call(gw, gs, dg, mi, dw, ds, do, di):
        def fused_mlp_call(self, x):
            orig = x.shape
            out = ops.w8a8_linear(x.reshape(-1, mi), gw, gs)
            out = out.reshape(*orig[:-1], -1)
            gate = out[..., :dg]
            up = out[..., dg:]
            hidden = nn.silu(gate) * up
            down = ops.w8a8_linear(hidden.reshape(-1, di), dw, ds)
            return down.reshape(*orig[:-1], do)

        return fused_mlp_call

    mlp.__call__ = types.MethodType(
        make_mlp_call(gu_w8, s_gu, dim_g, mlp_in, down_w8, down_s8, down_out, down_in),
        mlp,
    )

print("\n=== W8A8 Fused (QKV + Gate/Up) ===")
w8f_results = []
for step, (embeds, ids, ntok) in enumerate(all_steps3):
    w8f_results.append(
        bench(f"Step {step} ({ntok} tok)", m3.language_model, embeds, ids)
    )

# ======= Summary =======
print("\n" + "=" * 70)
print(
    f"{'Step':>5} {'Tok':>5} {'FP16':>8} {'W8A8':>8} {'W8A8F':>8} {'W8A8/FP':>8} {'Fuse/FP':>8} {'Fuse/W8':>8}"
)
print("-" * 70)
for step, (_, _, ntok) in enumerate(all_steps):
    sp1 = fp_results[step] / w8_results[step]
    sp2 = fp_results[step] / w8f_results[step]
    sp3 = w8_results[step] / w8f_results[step]
    print(
        f"{step:>5} {ntok:>5} {fp_results[step]:>7.0f}ms {w8_results[step]:>7.0f}ms {w8f_results[step]:>7.0f}ms {sp1:>7.2f}x {sp2:>7.2f}x {sp3:>7.2f}x"
    )
print("=" * 70)
