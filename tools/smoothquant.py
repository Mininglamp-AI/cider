import base64
from io import BytesIO
import json
from datasets import Dataset
import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset

# Note: this is an optional utility for processing vision inputs for qwen.
# This can be installed via the "qwen" extra
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.awq import AWQModifier
import random
# Load model.
model_id = "/mnt/data/ws/project/ms-swift/output/v3_sft_4b/v0-20260421-095015/checkpoint-50349"
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

# Oneshot arguments
DATASET_ID = "/mnt/data/ws/project/ms-swift/train_v4.jsonl"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 40000

# Load dataset and preprocess.
def reservoir_sampling(file_path, k):
    """
    从大文件中均匀随机采样 k 个样本
    
    时间复杂度: O(n)
    空间复杂度: O(k)
    """
    reservoir = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 先填充前 k 个样本
        for i, line in enumerate(f):
            l = line.strip()
            if len(l) == 0:
                continue
            if i < k:
                reservoir.append(l)
            else:
                # 对于第 i 个样本，以 k/i 的概率替换蓄水池中的某个样本
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = l
    
    return reservoir


# Apply chat template and tokenize inputs.
calib_data_raw = reservoir_sampling(DATASET_ID, NUM_CALIBRATION_SAMPLES)


def preprocess_and_tokenize(example):
    # preprocess
   
    text = processor.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=True
    )
    if "<image>" not in text:
        return None
    text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    
    res = processor(
        text=[text],
        images=example["images"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )
    # tokenize
    return res

calib_data = []
for i, item in enumerate(calib_data_raw):
    print("load %d/%d items..." % (i, len(calib_data_raw)))
    u = preprocess_and_tokenize(json.loads(item))
    if u is None:
        continue
    calib_data.append(preprocess_and_tokenize(json.loads(item)))

calib_dataset = Dataset.from_list(calib_data)
# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Recipe
recipe = [
    AWQModifier(duo_scaling=False),
    QuantizationModifier(
        scheme="W8A16",
        ignore=["re:.*lm_head", "re:.*visual.*"],
    ),
]
# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=calib_dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    sequential_targets=["Qwen3VLTextDecoderLayer"],
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W8A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)