#!/usr/bin/env python3
"""Benchmark client: call server multiple times, report timing stats.
Uses the same request format as client.py (images array + text with <image> tags).
"""
import requests
import json
import time
import base64
import statistics

BASE_URL = "http://server_ip:8341/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer test-key",
}

def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def single_request(images_b64, run_id):
    payload = {
        "model": "qwen3-vl",
        "request_id": run_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "You are a GUI agent. You are given a task and your action history, "
                    "with screenshots. You need to perform the next action to complete the task.\n\n"
                    "## Output Format\n<think>think</think>\n<action_desp>desc</action_desp>\n<action>action</action>\n\n"
                    "## Action Space\nclick(start_box='<|box_start|>(x1,y1)<|box_end|>')\nscroll(direction='down or up')\nstop(reason='')\nfinish()\n\n"
                    "## User Instruction\n### task: 查看Wiley可持续发展目标10的图书有哪些。\n"
                    "### action history: 第1步：Click Research.，对应的截图为<image>\n"
                    "第2步：Click SDG Hub.，对应的截图为<image>\n\n"
                    "当前截图为<image>"
                ),
            },
        ],
        "images": images_b64,
        "temperature": 0.7,
        "max_tokens": 256,
    }

    t0 = time.perf_counter()
    resp = requests.post(BASE_URL, json=payload, headers=HEADERS, timeout=120)
    total = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    prefill_time = data.get("prefill_time", 0)
    decode_tps = data.get("decode_tps", 0)
    text = data["choices"][0]["message"]["content"][:80] if data.get("choices") else ""

    return {
        "total": total,
        "prefill_time": prefill_time,
        "decode_tps": decode_tps,
        "text": text,
    }

def main():
    img_dir = "~/work/images/group0"
    im0 = load_image_b64(f"{img_dir}/0.png")
    im1 = load_image_b64(f"{img_dir}/1.png")
    im2 = load_image_b64(f"{img_dir}/2_resize1.png")
    images = [im0, im1, im2]

    n_warmup = 1
    n_bench = 3

    print(f"Images: 3 (group0/0.png, 1.png, 2_resize1.png)")
    print(f"Warmup: {n_warmup}, Bench: {n_bench}")
    print()

    # Warmup
    for i in range(n_warmup):
        r = single_request(images, f"warmup_{i}")
        print(f"Warmup {i}: total={r['total']:.3f}s, prefill={r['prefill_time']:.3f}s, decode={r['decode_tps']:.1f} t/s")

    # Bench
    results = []
    for i in range(n_bench):
        r = single_request(images, f"bench_{i}")
        results.append(r)
        print(f"Run {i}: total={r['total']:.3f}s, prefill={r['prefill_time']:.3f}s, decode={r['decode_tps']:.1f} t/s")

    # Stats
    prefills = [r["prefill_time"] for r in results]
    decodes = [r["decode_tps"] for r in results]
    totals = [r["total"] for r in results]

    print()
    print(f"Prefill: median={statistics.median(prefills):.3f}s, mean={statistics.mean(prefills):.3f}s")
    print(f"Decode:  median={statistics.median(decodes):.1f} t/s, mean={statistics.mean(decodes):.1f} t/s")
    print(f"Total:   median={statistics.median(totals):.3f}s, mean={statistics.mean(totals):.3f}s")
    print(f"Response: {results[-1]['text']}...")

if __name__ == "__main__":
    main()
