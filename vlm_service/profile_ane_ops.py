#!/usr/bin/env python3
"""
Isolate ANE eval time vs tile_transpose time.
Test: ane_run (no transpose) vs ane_run_T (with transpose)
"""
import os, sys, time, ctypes
import numpy as np

LIB_DIR = os.path.expanduser('~/.openclaw/workspace/research_bot/metal_int_gemm')

class ANEBridge:
    def __init__(self):
        lib = ctypes.CDLL(os.path.join(LIB_DIR, 'libane_bridge_v6.dylib'))
        lib.ane_init.restype = ctypes.c_int
        lib.ane_load_model.restype = ctypes.c_int
        lib.ane_load_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_float)]
        lib.ane_run_T.restype = ctypes.c_int
        lib.ane_run_T.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float)]
        lib.ane_run.restype = ctypes.c_int
        lib.ane_run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_float)]
        lib.ane_eval.restype = ctypes.c_int
        lib.ane_eval.argtypes = [ctypes.c_int]
        lib.ane_write_input.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        lib.ane_read_output.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        assert lib.ane_init() == 0
        self.lib = lib

    def load(self, ic, oc, seq, w_fp32):
        w = np.ascontiguousarray(w_fp32, dtype=np.float32)
        h = self.lib.ane_load_model(ic, oc, seq,
                w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        assert h >= 0
        return h


def main():
    SEQ = 512
    N = 20

    rng = np.random.default_rng(42)
    ane = ANEBridge()

    tests = [
        ("QKV",     2048, 4096,  SEQ),
        ("O_proj",  2048, 2048,  SEQ),
        ("GateUp",  2048, 12288, SEQ),
        ("Down",    6144, 2048,  SEQ),
    ]

    for name, ic, oc, seq in tests:
        w = (rng.standard_normal((oc, ic)) * 0.02).astype(np.float32)
        h = ane.load(ic, oc, seq, w)

        # Inputs in both layouts
        inp_seq_ic = np.ascontiguousarray(rng.standard_normal((seq, ic)).astype(np.float32) * 0.02)
        inp_ic_seq = np.ascontiguousarray(inp_seq_ic.T)  # [IC, SEQ]
        out_seq_oc = np.empty((seq, oc), dtype=np.float32)
        out_oc_seq = np.empty((oc, seq), dtype=np.float32)

        # Warmup
        for _ in range(3):
            ane.lib.ane_run_T(h,
                inp_seq_ic.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_seq_oc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            ane.lib.ane_run(h,
                inp_ic_seq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_oc_seq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        # ane_run_T (with tile_transpose)
        times_T = []
        for _ in range(N):
            t0 = time.perf_counter()
            ane.lib.ane_run_T(h,
                inp_seq_ic.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_seq_oc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            times_T.append((time.perf_counter() - t0) * 1000)

        # ane_run (no transpose, col-major I/O)
        times_R = []
        for _ in range(N):
            t0 = time.perf_counter()
            ane.lib.ane_run(h,
                inp_ic_seq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_oc_seq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            times_R.append((time.perf_counter() - t0) * 1000)

        # ane_eval only (no data copy)
        times_E = []
        # pre-load input
        ane.lib.ane_write_input(h,
            inp_ic_seq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        for _ in range(N):
            t0 = time.perf_counter()
            ane.lib.ane_eval(h)
            times_E.append((time.perf_counter() - t0) * 1000)

        # numpy transpose only
        times_NP = []
        for _ in range(N):
            t0 = time.perf_counter()
            _ = np.ascontiguousarray(inp_seq_ic.T)
            _ = np.ascontiguousarray(out_oc_seq.T)
            times_NP.append((time.perf_counter() - t0) * 1000)

        med_T = np.median(times_T)
        med_R = np.median(times_R)
        med_E = np.median(times_E)
        med_NP = np.median(times_NP)

        print(f"\n{name:8s} ({ic}→{oc}, seq={seq}):")
        print(f"  ane_run_T (tile_T):  {med_T:.3f}ms")
        print(f"  ane_run (memcpy):    {med_R:.3f}ms")
        print(f"  ane_eval only:       {med_E:.3f}ms")
        print(f"  numpy transpose:     {med_NP:.3f}ms")
        print(f"  tile_T overhead:     {med_T - med_R:.3f}ms")
        print(f"  memcpy overhead:     {med_R - med_E:.3f}ms")


if __name__ == '__main__':
    main()
