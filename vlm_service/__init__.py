"""cider.vlm_service — Qwen3-VL inference service with MLX patches."""

from .code_def import CustomGenerationResult, ErrorCode
from .custom_qwen3vl import custom_generate, custom_stream_generate
from .core_infer import HMInference

__all__ = [
    "CustomGenerationResult",
    "ErrorCode",
    "custom_generate",
    "custom_stream_generate",
    "HMInference",
]