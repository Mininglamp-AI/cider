# client_openai.py
import requests
import base64
import json


base_url="http://192.168.204.195:8341/v1/chat/completions"
def image_to_base64(image_path: str) -> str:
    """将图像转换为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def example_openai_non_stream():
    """使用 OpenAI SDK - 非流式"""
    print("=" * 60)
    print("OpenAI SDK 示例 - 非流式")
    print("=" * 60)
    im0 = image_to_base64("/Users/ws/work/images/group0/0.png")
    im1 = image_to_base64("/Users/ws/work/images/group0/1.png")
    im2 = image_to_base64("/Users/ws/work/images/group0/2_resize1.png")
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer sk-your-key"  # 如果需要认证
    }
    payload = {
        "model": "qwen3-vl",
        "messages": [
            {
                "content": "You are a helpful assistant.",
                "role": "system",
            },
            {
                "content": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. When outputting, output the thought process for the next action between the <think> and </think> tags, its description between the <action_desp> and </action_desp> tags, and the action itself between the <action> and </action> tags.\n\n## Output Format\n<think>think process</think>\n<action_desp>next action description</action_desp>\n<action>next action</action>\n\n## Action Space\nclick(start_box='<|box_start|>(x1,y1)<|box_end|>')\ndoubleclick(start_box='<|box_start|>(x1,y1)<|box_end|>')\nselect(start_box='<|box_start|>(x1,y1)<|box_end|>') # To expand a dropdown menu or select an item from it.\ndrag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>') # Drag an element from the start coordinate (x1,y1) to the end coordinate (x3,y3).\nhotkey(key='') # Trigger a keyboard shortcut.\nwait() # Sleep for 5s and take a screenshot to check for any changes.\ncall_user() # Request human assistance\ntype(content='', start_box='<|box_start|>(x1,y1)<|box_end|>') # First click on the textbox, and then type the content.\nstop(reason='') # If the item can not found in the image, give the reason\nscroll(direction='down or up or right or left') # If the all screenshot need to scroll on the `direction` side.\nscrollmenu(start_box='<|box_start|>(x1,y1),(x2,y2)<|box_end|>', direction='down or up or right or left') # If the part of screenshot need to scroll on the `direction` side and give the area need to scroll\ntable_get_data() # The table data extraction begins.\ntable_get_data_finish() # The table data extraction is completed.\nfinish() # The task is completed.\n\n## User Instruction\n### task: 查看Wiley可持续发展目标10的图书有哪些。\n### action history: 第1步：Click the Research下拉框.，对应的截图为<image>\n第2步：Click Sustainable Development Goals Hub.，对应的截图为<image>\n\n当前截图为<image>",
                "role": "user"
            },
        ],
        "images": [im0, im1, im2],
        "request_id": "tiandu",
        "temperature": 0.7,
    }
    response = requests.post(base_url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    # 打印结果
    print(f"Request ID: {result['id']}")
    print(f"AI: {result['choices'][0]['message']['content']}")
    print(f"Prefill Time: {result.get('prefill_time', 0):.3f}s")
    print(f"Decode Time: {result.get('decode_time', 0):.3f}s")
    print(f"E2E Time: {result.get('e2e_time', 0):.3f}s")


def example_openai_stream():
    """使用 OpenAI SDK - 流式"""
    print("=" * 60)
    print("OpenAI SDK 示例 - 流式")
    print("=" * 60)

    im0 = image_to_base64("/Users/ws/work/images/group0/0.png")
    im1 = image_to_base64("/Users/ws/work/images/group0/1.png")
    im2 = image_to_base64("/Users/ws/work/images/group0/2_resize1.png")
    with open("message.txt", "r") as f:
        raw = json.load(f)
    payload = {
        "model": "qwen3-vl",
        "request_id": "tiandu",
        "temperature": 0.7,
        "stream": True
    }
    messages = []
    for msg in raw:
        if msg["role"] == "system":
            messages.append({"role": "system", "content": msg["content"][0]["text"]})
        elif msg["role"] == "user":
            final_content = ""
            for content in msg["content"]:
                if content["type"] == "text":
                    final_content += content["text"]
                    if "截图" in content["text"]:
                        final_content += "<image>"
                elif content["type"] == "image_url":
                    pass
            messages.append({"role": "user", "content": final_content})
    payload["messages"] = messages
    payload["images"] = [image_to_base64("x.png")]
    # payload["request_id"] = "tiandu"

    payload = {
        "model": "qwen3-vl",
        "messages": [
            {
                "content": "You are a helpful assistant.",
                "role": "system",
            },
            {
                "content": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. When outputting, output the thought process for the next action between the <think> and </think> tags, its description between the <action_desp> and </action_desp> tags, and the action itself between the <action> and </action> tags.\n\n## Output Format\n<think>think process</think>\n<action_desp>next action description</action_desp>\n<action>next action</action>\n\n## Action Space\nclick(start_box='<|box_start|>(x1,y1)<|box_end|>')\ndoubleclick(start_box='<|box_start|>(x1,y1)<|box_end|>')\nselect(start_box='<|box_start|>(x1,y1)<|box_end|>') # To expand a dropdown menu or select an item from it.\ndrag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>') # Drag an element from the start coordinate (x1,y1) to the end coordinate (x3,y3).\nhotkey(key='') # Trigger a keyboard shortcut.\nwait() # Sleep for 5s and take a screenshot to check for any changes.\ncall_user() # Request human assistance\ntype(content='', start_box='<|box_start|>(x1,y1)<|box_end|>') # First click on the textbox, and then type the content.\nstop(reason='') # If the item can not found in the image, give the reason\nscroll(direction='down or up or right or left') # If the all screenshot need to scroll on the `direction` side.\nscrollmenu(start_box='<|box_start|>(x1,y1),(x2,y2)<|box_end|>', direction='down or up or right or left') # If the part of screenshot need to scroll on the `direction` side and give the area need to scroll\ntable_get_data() # The table data extraction begins.\ntable_get_data_finish() # The table data extraction is completed.\nfinish() # The task is completed.\n\n## User Instruction\n### task: 查看Wiley可持续发展目标10的图书有哪些。\n### action history: 第1步：Click the Research下拉框.，对应的截图为<image>\n第2步：Click Sustainable Development Goals Hub.，对应的截图为<image>\n\n当前截图为<image>",
                "role": "user"
            },
        ],
        "images": [im0, im1, im2],
        "request_id": "tiandu",
        "temperature": 0.7,
        "stream": True
    }
    response = requests.post(base_url, json=payload, stream=True)
    response.raise_for_status()
    
    print("Streaming response:")
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            print(content, end='', flush=True)
                    
                except:
                    pass
    print("\n")


if __name__ == "__main__":
    example_openai_non_stream()
    # example_openai_stream()
