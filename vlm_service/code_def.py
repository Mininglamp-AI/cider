from enum import Enum
from typing import Optional, List
from dataclasses import dataclass

class ErrorCode(str, Enum):
    SUCCESS = "success"
    INVLID_MESSAGE_FORMAT = "invalid message format. Message should contain images not pure text."
    VIT_FAILED = "Failed to extract image features. "
    PREFILL_TOO_LONG = "The context length exceeds the maximum limit."
    DECODE_TOO_LONG = "Decoding exceeds maximum length."

@dataclass
class CustomGenerationResult:
    text: str = ""
    token: Optional[int] = None
    logprobs: Optional[List[float]] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    code: ErrorCode = ErrorCode.SUCCESS
