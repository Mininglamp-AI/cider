#!/usr/bin/env python3
"""Replay the prompt that LocalAgent would have built at a given step.

Usage:
    python replay_prompt.py <session_dir> <step>
    python replay_prompt.py ~/mano-trajectory/result/sess-20260415-153428-1fe23a68 3

Output: JSON to stdout with keys:
    task, step, prompt, images (base64 list), system_prompt

Step is 0-indexed. Step 0 = first prediction (1 image: current screenshot).
"""

import argparse
import base64
import io
import json
import os
import re
import sys

from PIL import Image

SCREENSHOT_WIDTH = 1280
HISTORY_IMAGE_COUNT = 2

SYSTEM_PROMPT = "You are a helpful assistant."

ACTION_SPACE = """\
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
doubleclick(start_box='<|box_start|>(x1,y1)<|box_end|>')
select(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
wait()
call_user()
type(content='', start_box='<|box_start|>(x1,y1)<|box_end|>')
stop(reason='')
scroll(direction='down or up or right or left')
scrollmenu(start_box='<|box_start|>(x1,y1),(x2,y2)<|box_end|>', direction='down or up or right or left')
finish()"""

INSTRUCTION_TEMPLATE = """\
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. When outputting, output the thought process for the next action between the <think> and </think> tags, its description between the <action_desp> and </action_desp> tags, and the action itself between the <action> and </action> tags.

## Output Format
<think>think process</think>
<action_desp>next action description</action_desp>
<action>next action</action>

## Action Space
{action_space}

## User Instruction
### task: {task}
### action history: {history}
{current_screenshot}"""


def load_session(session_dir: str):
    """Load result.json and return task + history_resps."""
    result_path = os.path.join(session_dir, "result.json")
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["task"], data["history_resps"]


def parse_action_desc(resp_str: str) -> str:
    """Extract action description from history_resps entry.
    Prefer <action_desp> (matches runtime behavior), fallback to parsing <action>.
    """
    # Prefer action_desp (what LocalAgent actually stores in prompt_history)
    m = re.search(r"<action_desp>(.*?)</action_desp>", resp_str, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # Fallback: parse from <action> tag
    m = re.search(r"<action>(.*?)</action>", resp_str, re.DOTALL)
    if not m:
        return "unknown"
    action_str = m.group(1).strip()

    coord_match = re.search(r"'action':\s*'(\w+)'", action_str)
    action_name = coord_match.group(1) if coord_match else "unknown"

    coord_match2 = re.search(r"'coordinate':\s*\[(\d+),\s*(\d+)\]", action_str)
    if coord_match2:
        return f"{action_name} ({coord_match2.group(1)}, {coord_match2.group(2)})"

    text_match = re.search(r"'text':\s*'([^']*)'", action_str)
    if text_match:
        return f"{action_name}: {text_match.group(1)[:30]}"

    dir_match = re.search(r"'scroll_direction':\s*'(\w+)'", action_str)
    if dir_match:
        return f"{action_name} {dir_match.group(1)}"

    if "DONE" in action_str:
        return "DONE"
    if "FAIL" in action_str:
        return "FAIL"

    return action_name


def img_to_b64(img_path: str) -> str:
    """Load image, resize to SCREENSHOT_WIDTH, return base64."""
    img = Image.open(img_path)
    if img.width != SCREENSHOT_WIDTH:
        ratio = SCREENSHOT_WIDTH / img.width
        new_h = int(img.height * ratio)
        img = img.resize((SCREENSHOT_WIDTH, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_prompt_at_step(session_dir: str, step: int):
    """Reconstruct the prompt that LocalAgent would build at the given step."""
    task, history_resps = load_session(session_dir)
    trajectory_dir = os.path.join(session_dir, "trajectory")

    if step < 0 or step >= len(history_resps):
        sys.exit(json.dumps({"error": f"step {step} out of range (0-{len(history_resps)-1})"}))

    # Build prompt_history for steps 0..step-1 (previous steps)
    prompt_history = []
    for i in range(step):
        desc = parse_action_desc(history_resps[i])
        screenshot_path = os.path.join(trajectory_dir, f"{i}.png")
        screenshot_b64 = img_to_b64(screenshot_path) if os.path.exists(screenshot_path) else ""
        prompt_history.append({"desc": desc, "screenshot_b64": screenshot_b64})

    # Current screenshot = step.png
    current_path = os.path.join(trajectory_dir, f"{step}.png")
    if not os.path.exists(current_path):
        sys.exit(json.dumps({"error": f"{current_path} not found"}))
    current_b64 = img_to_b64(current_path)

    # Build prompt text + images list (same logic as LocalAgent._build_prompt)
    images = []
    recent = prompt_history[-(HISTORY_IMAGE_COUNT + 1):]

    history_parts = []
    for i, h in enumerate(prompt_history):
        step_num = i + 1
        desc = h["desc"]
        if h in recent and h.get("screenshot_b64"):
            images.append(h["screenshot_b64"])
            history_parts.append(f"第{step_num}步：{desc}，对应的截图为<image>")
        else:
            history_parts.append(f"第{step_num}步：{desc}")

    history_text = "\n".join(history_parts) if history_parts else "无"
    images.append(current_b64)

    prompt_text = INSTRUCTION_TEMPLATE.format(
        action_space=ACTION_SPACE,
        task=task,
        history=history_text,
        current_screenshot="当前截图为<image>",
    )

    return {
        "task": task,
        "step": step,
        "system_prompt": SYSTEM_PROMPT,
        "prompt": prompt_text,
        "images": images,
        "image_count": len(images),
        "history": [h["desc"] for h in prompt_history],
    }


def main():
    parser = argparse.ArgumentParser(description="Replay LocalAgent prompt at a given step")
    parser.add_argument("session_dir", help="Path to session directory")
    parser.add_argument("step", type=int, help="Step number (0-indexed)")
    args = parser.parse_args()

    session_dir = os.path.expanduser(args.session_dir)
    result = build_prompt_at_step(session_dir, args.step)
    with open("prompt", "w") as fo:
        json.dump(result, fo, ensure_ascii=False)


if __name__ == "__main__":
    main()
