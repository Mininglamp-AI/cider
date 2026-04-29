from .code_def import ErrorCode
import time
import logging
from collections import defaultdict
from typing import List, Optional, Dict
import mlx_vlm as pm
from .custom_qwen3vl import *

logging.basicConfig(level=logging.ERROR)
import warnings
import threading
from typing import Optional

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
import torch

import numpy as np
from dataclasses import dataclass
from mlx_vlm import generate

TARGET_TYPE = torch.float16


class Timer:
    """带自动统计功能的计时器"""

    _records: Dict[str, List[float]] = defaultdict(list)
    _enabled = True

    def __init__(self, name: str = "Code block", verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        if Timer._enabled:
            Timer._records[self.name].append(self.elapsed)
        if self.verbose:
            print(f"[{self.name}] {self.elapsed:.4f}s")
        return False

    @classmethod
    def report(cls, sort_by: str = "total") -> None:
        if not cls._records:
            print("No timing records.")
            return
        print("\n" + "=" * 70)
        print(f"{'Name':<30} {'Count':>8} {'Total':>10} {'Mean':>10} {'Min':>10} {'Max':>10}")
        print("=" * 70)
        stats = []
        for name, times in cls._records.items():
            stats.append({
                "name": name, "count": len(times), "total": sum(times),
                "mean": sum(times) / len(times), "min": min(times), "max": max(times),
            })
        if sort_by in ["total", "mean", "count"]:
            stats.sort(key=lambda x: x[sort_by], reverse=True)
        elif sort_by == "name":
            stats.sort(key=lambda x: x["name"])
        for s in stats:
            print(f"{s['name']:<30} {s['count']:>8} {s['total']:>10.4f}s {s['mean']:>10.4f}s {s['min']:>10.4f}s {s['max']:>10.4f}s")
        print("=" * 70)
        print(f"Total time: {sum(s['total'] for s in stats):.4f}s")
        print()

    @classmethod
    def get_stats(cls, name: Optional[str] = None) -> Dict:
        if name:
            times = cls._records.get(name, [])
            if not times:
                return {}
            return {
                "count": len(times), "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times), "max": max(times), "times": times,
            }
        else:
            return {k: cls.get_stats(k) for k in cls._records.keys()}

    @classmethod
    def reset(cls, name: Optional[str] = None) -> None:
        if name:
            cls._records[name] = []
        else:
            cls._records.clear()

    @classmethod
    def disable(cls) -> None:
        cls._enabled = False

    @classmethod
    def enable(cls) -> None:
        cls._enabled = True


class HMInference:
    _instance: Optional['HMInference'] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 model_path,
                 temperature=1.0,
                 topk=None,
                 topp=1.0,
                 repetition_penalty=1.0,
                 max_new_tokens=1024,
                 w8a8="auto"):
        if HMInference._initialized:
            return
        with HMInference._lock:
            if HMInference._initialized:
                return
            self.model, self.processor = pm.load(model_path)

            # ── W8A8 INT8 TensorOps via cider ──
            # Replaces all Linear layers with CiderLinear:
            #   prefill mode → W8A8 INT8 TensorOps (~15-19% faster)
            #   decode mode  → original weights (zero overhead)
            self._w8a8_enabled = False
            if w8a8 != "off":
                try:
                    from cider import convert_model, set_mode, is_available
                    if w8a8 == "auto" and not is_available():
                        logging.info(
                            "[W8A8] Hardware does not support INT8 TensorOps "
                            "(requires M5+), using default inference"
                        )
                    else:
                        import mlx.core as mx
                        try:
                            stats = convert_model(self.model.language_model)
                        except:
                            stats = convert_model(self.model)
                        mx.eval(self.model.parameters())
                        self._w8a8_set_mode = set_mode
                        self._w8a8_enabled = True
                        logging.info(f"[W8A8] cider enabled: {stats}")
                except Exception as e:
                    if w8a8 == "on":
                        raise
                    logging.warning(
                        f"[W8A8] Init failed, using default inference: {e}"
                    )

            self.temperature = temperature
            self.topk = topk
            self.topp = topp
            self.repetition_penalty = repetition_penalty
            self.max_new_tokens = max_new_tokens
            HMInference._initialized = True

    def complete_stream(self, messages, images, buf_vis_feats,
                        buf_vis_stack_feats, **kwargs):
        """流式推理接口"""
        temperature = kwargs.pop("temperature", self.temperature)
        topk = kwargs.pop("topk", self.topk)
        topp = kwargs.pop("topp", self.topp)
        repetition_penalty = kwargs.pop("repetition_penalty", self.repetition_penalty)
        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        org_image_placeholder = "<image>"
        new_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        pi = len(images)
        while pi > 0:
            pi -= 1
            pos = prompt.rfind(org_image_placeholder)
            if pos >= 0:
                prompt = prompt[:pos] + prompt[pos:].replace(
                    org_image_placeholder, new_image_placeholder)
            else:
                break

        if self._w8a8_enabled:
            self._w8a8_set_mode("prefill")

        for resp in custom_stream_generate(
                self.model,
                self.processor,
                prompt,
                images,
                buf_vis_features=buf_vis_feats,
                buf_vis_stack_features=buf_vis_stack_feats,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=topp,
                top_k=topk,
                repetition_penalty=repetition_penalty,
                verbose=True,
                prefill_step_size=2048,
                on_first_token=(
                    lambda: self._w8a8_set_mode("decode")
                ) if self._w8a8_enabled else None,
        ):
            yield resp.code, resp.text, {
                "prefill_time": resp.prompt_tokens / resp.prompt_tps,
                "decode_tps": resp.generation_tps
            }

    def complete(self, messages, images, buf_vis_feats, buf_vis_stack_feats,
                 **kwargs):
        """非流式推理接口"""
        temperature = kwargs.pop("temperature", self.temperature)
        topk = kwargs.pop("topk", self.topk)
        topp = kwargs.pop("topp", self.topp)
        repetition_penalty = kwargs.pop("repetition_penalty", self.repetition_penalty)
        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        org_image_placeholder = "<image>"
        new_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        pi = len(images)
        while pi > 0:
            pi -= 1
            pos = prompt.rfind(org_image_placeholder)
            if pos >= 0:
                prompt = prompt[:pos] + prompt[pos:].replace(
                    org_image_placeholder, new_image_placeholder)
            else:
                break

        if self._w8a8_enabled:
            self._w8a8_set_mode("prefill")

        resp = custom_generate(self.model,
                               self.processor,
                               prompt,
                               images,
                               buf_vis_features=buf_vis_feats,
                               buf_vis_stack_features=buf_vis_stack_feats,
                               max_tokens=max_new_tokens,
                               temperature=temperature,
                               top_p=topp,
                               top_k=topk,
                               repetition_penalty=repetition_penalty,
                               prefill_step_size=2048,
                               on_first_token=(
                                   lambda: self._w8a8_set_mode("decode")
                               ) if self._w8a8_enabled else None)

        return resp.code, resp.text, {
            "prefill_time": resp.prompt_tokens / resp.prompt_tps,
            "decode_tps": resp.generation_tps
        }
