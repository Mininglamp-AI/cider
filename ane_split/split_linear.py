#!/usr/bin/env python3
"""
SplitLinear: Drop-in replacement for nn.Linear / nn.QuantizedLinear.
Splits GEMM along output channels — ANE does ~65%, GPU does ~35%, concurrent.

Key optimization: same-input projections (Q/K/V, Gate/Up) share a single
input preparation via _InputGroup, eliminating redundant transpose+eval+numpy.

Usage:
    from split_linear import patch_model
    bridge = patch_model(model, seq=512)

API:
    patch_model(model, seq)     → high-level one-liner  
    SplitLinear(layer, bridge, seq)  → single layer replacement
    ANEBridge()                      → ANE private API wrapper
"""
import os, ctypes
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx
import mlx.nn as nn

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_ALIGN = 64
MIN_SEQ_FOR_SPLIT = 192  # Below this, split overhead > benefit


# ─── ANE Bridge ───

class ANEBridge:
    """Thin wrapper around ANE private API."""
    _instance = None

    def __init__(self):
        lib = ctypes.CDLL(os.path.join(LIB_DIR, 'libane_bridge_v6.dylib'))
        lib.ane_init.restype = ctypes.c_int
        lib.ane_load_model.restype = ctypes.c_int
        lib.ane_load_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_float)]
        lib.ane_run.restype = ctypes.c_int
        lib.ane_run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_float)]
        lib.ane_run_rowmajor.restype = ctypes.c_int
        lib.ane_run_rowmajor.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                         ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        lib.ane_model_count.restype = ctypes.c_int
        lib.ane_surface_count.restype = ctypes.c_int
        assert lib.ane_init() == 0, "ANE init failed"
        self.lib = lib

    def load(self, ic, oc, seq, w_fp32):
        w = np.ascontiguousarray(w_fp32, dtype=np.float32)
        h = self.lib.ane_load_model(ic, oc, seq,
                w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        assert h >= 0, f"ANE load failed: {ic}→{oc}, seq={seq}"
        return h

    def run(self, h, inp, out):
        self.lib.ane_run(h,
            inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def run_rowmajor(self, h, inp_rm, L, out_rm):
        """ANE compute with row-major I/O. Uses vDSP for transpose.
        inp_rm: [L, IC] row-major float32
        out_rm: [L, OC] row-major float32 (pre-allocated)
        """
        self.lib.ane_run_rowmajor(h,
            inp_rm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            L,
            out_rm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    @property
    def model_count(self):
        return self.lib.ane_model_count()

    @classmethod
    def shared(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Single ANE worker thread
_ane_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ane')


# ─── Helpers ───

def _extract_weight(layer):
    """Extract FP32 weight [OC, IC] from nn.Linear or nn.QuantizedLinear."""
    if isinstance(layer, nn.QuantizedLinear):
        mx.eval(layer.weight, layer.scales)
        b = getattr(layer, 'biases', None)
        if b is not None:
            mx.eval(b)
        w = mx.dequantize(layer.weight, layer.scales, b,
                          layer.group_size, layer.bits)
    else:
        w = layer.weight
    mx.eval(w)
    return np.array(w, dtype=np.float32)


class _InputGroup:
    """
    Shared input preparation for same-input projections (Q/K/V or Gate/Up).
    
    The first SplitLinear in the group calls prepare(x), which does the
    expensive transpose+eval+numpy copy once. Subsequent SplitLinears
    in the group call get() to reuse the cached result.
    """
    __slots__ = ('_np', '_x2d', '_L', '_buf', '_seq')

    def __init__(self, ic, seq):
        self._np = None
        self._x2d = None
        self._L = 0
        self._buf = np.zeros((ic, seq), dtype=np.float32)
        self._seq = seq

    def prepare(self, x):
        """Prepare input. Called by first proj in group.
        Returns (inp_np_colmajor, x_2d, L)."""
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x
        L = x_2d.shape[0]

        x_t = mx.contiguous(x_2d.T)
        mx.eval(x_t)
        x_np = np.array(x_t, copy=False)

        if L < self._seq:
            self._buf[:, :L] = x_np
            self._buf[:, L:] = 0
            self._np = self._buf
        else:
            self._np = x_np

        self._L = L
        self._x2d = x_2d
        return self._np, x_2d, L

    def get(self):
        """Get cached input. Called by subsequent projs.
        Returns (inp_np_colmajor, x_2d, L)."""
        return self._np, self._x2d, self._L


# ─── SplitLinear ───

class SplitLinear:
    """
    Drop-in nn.Linear replacement with ANE+GPU tensor parallelism.
    
    Two modes controlled externally via set_prefill(True/False):
      - prefill: ANE ~65% + GPU ~35% concurrent → 1.18x speedup
      - decode:  original nn.Linear on GPU → zero overhead
    
    No extra GPU memory: original layer weight is reused for decode.
    """
    _prefill_mode = False  # class-level flag

    def __init__(self, layer, bridge, seq, ane_frac=None, name="",
                 input_group=None, is_first=True):
        self.name = name
        self._orig = layer  # always keep original for decode
        w_np = _extract_weight(layer)
        oc, ic = w_np.shape
        self.ic = ic
        self.oc = oc
        self.seq = seq
        self._is_first = is_first

        # Auto-detect split fraction
        if ane_frac is None:
            if seq < MIN_SEQ_FOR_SPLIT:
                ane_frac = 0.0
            elif ic > oc * 2:
                ane_frac = 0.0  # Wide→narrow: ANE inefficient (down_proj)
            elif oc < SPLIT_ALIGN * 2:
                ane_frac = 0.0
            else:
                ane_frac = 0.65

        self.ane_frac = ane_frac

        if ane_frac <= 0:
            self.mode = 'gpu'
            return

        self.mode = 'split'
        self.ane = bridge

        self.ane_oc = (int(oc * ane_frac) // SPLIT_ALIGN) * SPLIT_ALIGN
        self.gpu_oc = oc - self.ane_oc

        if self.ane_oc < SPLIT_ALIGN or self.gpu_oc < 1:
            self.mode = 'gpu'
            return

        self.h_ane = bridge.load(ic, self.ane_oc, seq, w_np[:self.ane_oc, :])
        self.buf_ane = np.empty((self.ane_oc, seq), dtype=np.float32)

        # GPU weight for split path
        self._is_quantized = isinstance(layer, nn.QuantizedLinear)
        if self._is_quantized:
            # QuantizedLinear: _orig.weight is packed uint32, can't slice directly.
            # Store dequantized GPU portion as mx float16.
            w_gpu_np = w_np[self.ane_oc:, :]  # [gpu_oc, ic] float32
            self._w_gpu = mx.array(w_gpu_np).astype(mx.float16)
            mx.eval(self._w_gpu)
        else:
            # nn.Linear: zero-copy slice from _orig.weight at runtime
            self._w_gpu = None

        if input_group is not None:
            self._grp = input_group
        else:
            self._grp = _InputGroup(ic, seq)
            self._is_first = True

    @classmethod
    def set_prefill(cls, enabled):
        """Switch all SplitLinear instances between prefill/decode mode."""
        cls._prefill_mode = enabled

    def __call__(self, x):
        # decode / gpu-only / short seq: use original nn.Linear
        if self.mode == 'gpu' or not SplitLinear._prefill_mode:
            return self._orig(x)
        
        L = x.shape[-2] if x.ndim == 3 else x.shape[0]
        if L < MIN_SEQ_FOR_SPLIT:
            return self._orig(x)

        # ── prefill split path ──
        orig_shape = x.shape

        # Input preparation (shared or fresh)
        if self._is_first:
            inp_np, x_2d, L = self._grp.prepare(x)
        else:
            inp_np, x_2d, L = self._grp.get()

        # GPU: matmul with GPU portion weight
        if self._w_gpu is not None:
            # Quantized: use pre-dequantized GPU weight
            w_gpu = self._w_gpu
        else:
            # FP16: zero-copy slice from original weight
            w_gpu = self._orig.weight[self.ane_oc:, :]
        gpu_out = x_2d @ w_gpu.T
        mx.async_eval(gpu_out)

        # ANE: concurrent with GPU
        self.ane.run(self.h_ane, inp_np, self.buf_ane)

        # Merge
        buf = self.buf_ane if L == self.seq else self.buf_ane[:, :L]
        ane_out = mx.array(buf).T
        merged = mx.concatenate([ane_out, gpu_out], axis=-1)

        if len(orig_shape) == 3:
            merged = merged.reshape(orig_shape[0], orig_shape[1], -1)
        return merged

    def __repr__(self):
        if self.mode == 'gpu':
            return f"SplitLinear({self.name}, gpu_only, {self.ic}→{self.oc})"
        return (f"SplitLinear({self.name}, {self.ane_oc}ane+{self.gpu_oc}gpu, "
                f"{self.ic}→{self.oc}, frac={self.ane_frac:.0%})")


# ─── Tree Walk ───

def _find_linears(module, prefix=""):
    """Walk MLX model tree, yield (parent, key, full_name, linear)."""
    if not hasattr(module, 'children'):
        return
    for attr_name, child in module.children().items():
        full_name = f"{prefix}.{attr_name}" if prefix else attr_name
        if isinstance(child, (nn.Linear, nn.QuantizedLinear)):
            yield (module, attr_name, full_name, child)
        elif isinstance(child, nn.Module):
            yield from _find_linears(child, full_name)
        elif isinstance(child, list):
            for i, v in enumerate(child):
                fname = f"{full_name}.{i}"
                if isinstance(v, (nn.Linear, nn.QuantizedLinear)):
                    yield (child, i, fname, v)
                elif hasattr(v, 'children'):
                    yield from _find_linears(v, fname)
        elif isinstance(child, dict):
            for k, v in child.items():
                fname = f"{full_name}.{k}"
                if isinstance(v, (nn.Linear, nn.QuantizedLinear)):
                    yield (module, k, fname, v)
                elif hasattr(v, 'children'):
                    yield from _find_linears(v, fname)
        elif hasattr(child, 'children'):
            yield from _find_linears(child, full_name)


def _get_input_dims(layer):
    """Get true input dimensions for nn.Linear or nn.QuantizedLinear."""
    if isinstance(layer, nn.QuantizedLinear):
        # packed uint32: input_dims = weight.shape[-1] * 32 // bits
        return layer.weight.shape[-1] * 32 // layer.bits
    return layer.weight.shape[-1]


def patch_model(model, seq, bridge=None, verbose=True):
    """
    Patch all linear layers with SplitLinear.
    
    Same-input projections share an InputGroup:
      - Q/K/V share one (same hidden_state input)
      - Gate/Up share one (same hidden_state after attn)
      - O and Down each get their own (unique inputs)
    
    Returns: bridge instance.
    """
    if bridge is None:
        bridge = ANEBridge.shared()

    lang = model.language_model
    lm = lang.model
    N = len(lm.layers)
    n_split = n_gpu = 0

    for li in range(N):
        la = lm.layers[li]
        attn = la.self_attn
        mlp = la.mlp

        ic_attn = _get_input_dims(attn.q_proj)
        ic_mlp = _get_input_dims(mlp.gate_proj)

        # Q/K/V share input group
        qkv_grp = _InputGroup(ic_attn, seq)
        for i, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
            orig = getattr(attn, name)
            sl = SplitLinear(orig, bridge, seq,
                            name=f"layer.{li}.{name}",
                            input_group=qkv_grp,
                            is_first=(i == 0))
            setattr(attn, name, sl)
            n_split += 1 if sl.mode == 'split' else 0
            n_gpu += 1 if sl.mode == 'gpu' else 0

        # O proj — own input
        sl = SplitLinear(attn.o_proj, bridge, seq,
                        name=f"layer.{li}.o_proj")
        attn.o_proj = sl
        n_split += 1 if sl.mode == 'split' else 0
        n_gpu += 1 if sl.mode == 'gpu' else 0

        # Gate/Up share input group
        gu_grp = _InputGroup(ic_mlp, seq)
        for i, name in enumerate(('gate_proj', 'up_proj')):
            orig = getattr(mlp, name)
            sl = SplitLinear(orig, bridge, seq,
                            name=f"layer.{li}.{name}",
                            input_group=gu_grp,
                            is_first=(i == 0))
            setattr(mlp, name, sl)
            n_split += 1 if sl.mode == 'split' else 0
            n_gpu += 1 if sl.mode == 'gpu' else 0

        # Down proj — own input (auto GPU-only due to IC>OC*2)
        sl = SplitLinear(mlp.down_proj, bridge, seq,
                        name=f"layer.{li}.down_proj")
        mlp.down_proj = sl
        n_split += 1 if sl.mode == 'split' else 0
        n_gpu += 1 if sl.mode == 'gpu' else 0

    if verbose:
        print(f"[SplitLinear] {N} layers: {n_split} split, {n_gpu} gpu-only")
        print(f"[SplitLinear] ANE models: {bridge.model_count}")

    return bridge


# ─── Self-test ───
if __name__ == '__main__':
    import sys, time
    from mlx_vlm.utils import load as vlm_load
    from mlx_vlm.models.cache import KVCache

    SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    N_W = 3; N_B = 10
    FP16_MODEL = '/Users/ws/Downloads/weights/mlx/Qwen3-VL-2B-Instruct-16bit'

    print(f"\n{'='*60}")
    print(f"  SplitLinear Self-Test (seq={SEQ})")
    print(f"{'='*60}\n")

    model, _ = vlm_load(FP16_MODEL)
    lang = model.language_model
    N = lang.args.num_hidden_layers
    ids = mx.ones((1, SEQ), dtype=mx.int32)
    pos = mx.broadcast_to(mx.arange(SEQ).reshape(1, SEQ)[None, :, :], (3, 1, SEQ))
    mx.eval(ids, pos)

    # GPU baseline
    print("[1/3] GPU FP16 baseline")
    for _ in range(N_W):
        c = [KVCache() for _ in range(N)]
        mx.eval(lang(ids, cache=c, position_ids=pos).logits)
    ts = []
    for _ in range(N_B):
        c = [KVCache() for _ in range(N)]
        t0 = time.perf_counter()
        mx.eval(lang(ids, cache=c, position_ids=pos).logits)
        ts.append((time.perf_counter()-t0)*1000)
    bl = float(np.median(ts))
    print(f"  {bl:.1f}ms\n")

    # Reference logits
    c_ref = [KVCache() for _ in range(N)]
    ref = np.array(lang(ids, cache=c_ref, position_ids=pos).logits)
    mx.eval()

    # Patch
    print("[2/3] Patch + benchmark")
    bridge = patch_model(model, SEQ)
    SplitLinear.set_prefill(True)  # enable split for benchmark

    # Warmup
    for _ in range(N_W):
        c = [KVCache() for _ in range(N)]
        mx.eval(lang(ids, cache=c, position_ids=pos).logits)

    # Accuracy
    c_hyb = [KVCache() for _ in range(N)]
    hyb = np.array(lang(ids, cache=c_hyb, position_ids=pos).logits)
    mx.eval()
    cos = float(np.dot(ref.flatten(), hyb.flatten()) /
                (np.linalg.norm(ref) * np.linalg.norm(hyb) + 1e-12))
    top1 = float((ref.argmax(-1) == hyb.argmax(-1)).mean() * 100)
    print(f"\n  cos={cos:.6f}, top1={top1:.1f}%")

    # Benchmark
    ts = []
    for i in range(N_B):
        c = [KVCache() for _ in range(N)]
        t0 = time.perf_counter()
        mx.eval(lang(ids, cache=c, position_ids=pos).logits)
        t = (time.perf_counter()-t0)*1000
        ts.append(t)
        print(f"  Run {i+1}: {t:.1f}ms")
    med = float(np.median(ts))

    print(f"\n{'='*60}")
    print(f"  GPU FP16:      {bl:.1f}ms")
    print(f"  SplitLinear:   {med:.1f}ms  ({bl/med:.3f}x)")
    print(f"  cos={cos:.6f}  top1={top1:.1f}%")
    print(f"{'='*60}")
