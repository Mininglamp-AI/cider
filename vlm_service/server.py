import asyncio
import uuid
import json
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from queue import Queue
import uvicorn
from typing import List, Dict, Optional
import torch
import argparse
import time
import threading
from PIL import Image
from core_infer import HMInference, ErrorCode
import logging
from io import BytesIO
from config import load_config, get_config

# ============= Pydantic Models for OpenAI Compatible API =============
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def base64_to_pil(base64_string):
    """base64 转 PIL.Image"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


class Message(BaseModel):
    role: str
    content: str | List[Dict]


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-vl"
    messages: List[Message]
    images: Optional[List] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_tokens: int = 2048
    stream: bool = False
    request_id: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage
    prefill_time: float = 0.0
    decode_tps: float = 0.0


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class InferenceRequest:
    """推理请求"""

    def __init__(
        self,
        request_id: str,
        messages: List[Message],
        images: List[Image.Image],
        params: Dict,
    ):
        self.request_id = request_id
        self.messages = messages
        self.images = images
        self.params = params
        self.result_queue = Queue()
        self.stream = params.get("stream", False)


class RequestQueueManager:
    """请求队列管理器（单例模式下的请求排队）"""

    def __init__(self):
        self.queue = Queue()

    def add_request(self, request: InferenceRequest):
        """添加请求到队列"""
        self.queue.put(request)

    def get_next_request(self) -> Optional[InferenceRequest]:
        """获取下一个请求"""
        if not self.queue.empty():
            return self.queue.get(timeout=1)
        return None

    def is_empty(self) -> bool:
        return self.queue.empty()

    def size(self) -> int:
        """队列大小"""
        return self.queue.qsize()


# ============= Request Context Manager =============


class RequestContext:
    """管理单个请求的上下文（多轮对话的图像特征缓存）"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.image_features_buffer: List[torch.Tensor] = []
        self.image_stack_feature_buffer: List[torch.Tensor] = []
        self.created_at = time.time()
        self.last_accessed = self.created_at

    def get_image_features_buffer(self) -> List[torch.Tensor]:
        """获取图像特征缓存"""
        self.last_accessed = time.time()
        return self.image_features_buffer, self.image_stack_feature_buffer


class RequestContextManager:
    """管理所有请求的上下文"""

    def __init__(self, ttl: int = 3600):
        self.contexts: Dict[str, RequestContext] = {}
        self.ttl = ttl  # 上下文存活时间（秒）
        self.lock = threading.Lock()

    def get_or_create_context(self, request_id: str) -> RequestContext:
        """获取或创建请求上下文"""
        with self.lock:
            if request_id not in self.contexts:
                self.contexts[request_id] = RequestContext(request_id)
            return self.contexts[request_id]

    def cleanup_expired_contexts(self):
        """清理过期的上下文"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, ctx in self.contexts.items()
                if current_time - ctx.last_accessed > self.ttl
            ]
            for key in expired_keys:
                del self.contexts[key]
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired contexts")


class InferenceService:
    """推理服务:管理单例HMInference和请求队列"""

    _instance: Optional["InferenceService"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cfg):
        if hasattr(self, "_initialized") and self._initialized:
            return

        # 初始化HMInference单例
        self.inference_engine = HMInference(cfg.model.model_name_or_path,
                                            cfg.sampling.temperature,
                                            cfg.sampling.top_k,
                                            cfg.sampling.top_p,
                                            cfg.sampling.repetition_penalty,
                                            cfg.sampling.max_new_tokens)

        # 请求上下文管理器
        self.context_manager = RequestContextManager(cfg.server.ttl)

        # 请求队列管理器
        self.queue_manager = RequestQueueManager()

        # 启动处理线程
        self.worker_thread = threading.Thread(target=self._process_requests,
                                              daemon=True)
        self.worker_thread.start()

        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop,
                                               daemon=True)
        self.cleanup_thread.start()

        self._initialized = True
        logger.info("InferenceService initialized")

    def _cleanup_loop(self):
        """定期清理过期上下文"""
        while True:
            time.sleep(300)  # 每5分钟清理一次
            self.context_manager.cleanup_expired_contexts()

    def _process_requests(self):
        """处理请求队列（单线程串行处理）"""
        while True:
            request = self.queue_manager.get_next_request()
            if request is None:
                time.sleep(0.01)  # 避免CPU空转
                continue

            try:
                if request.stream:
                    self._process_stream_request(request)
                else:
                    self._process_non_stream_request(request)
            except Exception as e:
                logger.error(
                    f"Error processing request {request.request_id}: {e}")
                request.result_queue.put({"status": str(e)})

    def _process_non_stream_request(self, request: InferenceRequest):
        """处理非流式请求"""
        # 获取请求上下文
        ctx = self.context_manager.get_or_create_context(request.request_id)
        buf_vis_feats, buf_vis_stack_feats = ctx.get_image_features_buffer()

        # 执行推理
        code, generated_text, timing = self.inference_engine.complete(
            request.messages,
            request.images,
            buf_vis_feats,
            buf_vis_stack_feats,
            **request.params,
        )

        # 返回结果
        result = {
            "status": code,
            "text": generated_text if code == ErrorCode.SUCCESS else "",
            "prefill_time": timing["prefill_time"],
            "decode_tps": timing["decode_tps"],
        }
        if code != ErrorCode.SUCCESS:
            result["error"] = code
        request.result_queue.put(result)

    def _process_stream_request(self, request: InferenceRequest):
        """处理流式请求 - 使用 complete_stream"""
        ctx = self.context_manager.get_or_create_context(request.request_id)
        buf_vis_feats, buf_vis_stack_feats = ctx.get_image_features_buffer()

        try:
            stream_gen = self.inference_engine.complete_stream(
                request.messages,
                request.images,
                buf_vis_feats,
                buf_vis_stack_feats,
                **request.params,
            )

            for code, text, timing in stream_gen:
                if code != ErrorCode.SUCCESS:
                    request.result_queue.put({
                        "status":
                        code,
                        "done":
                        True,
                        "prefill_time":
                        timing["prefill_time"],
                        "decode_tps":
                        timing["decode_tps"],
                        "error":
                        code,
                    })
                    return
                else:
                    request.result_queue.put({
                        "text": text,
                        "done": False,
                    })
            request.result_queue.put({
                "text": "",
                "done": True,
                "prefill_time": timing["prefill_time"],
                "decode_tps": timing["decode_tps"],
            })
        except Exception as e:
            logger.error(f"Stream error: {e}")
            request.result_queue.put({"error": str(e), "done": True})

    async def submit_request(self, request: InferenceRequest) -> Queue:
        """提交请求到队列"""
        self.queue_manager.add_request(request)
        return request.result_queue


# 全局变量
inference_service: Optional[InferenceService] = None
_global_config = None


def init_config(config_path: str = "config.yaml"):
    """初始化全局配置"""
    global _global_config
    if _global_config is None:
        _global_config = load_config(config_path)
    return _global_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global inference_service

    # ✅ 安全获取配置（带默认值）
    try:
        cfg = get_config()
    except RuntimeError:
        # 如果配置还没加载，使用默认配置路径
        logger.warning("Config not loaded, using default config.yaml")
        cfg = init_config("config.yaml")

    inference_service = InferenceService(cfg)
    logger.info("Service started")
    yield
    logger.info("Service shutting down")


app = FastAPI(title="Mininglamp OpenAI Compatible API", lifespan=lifespan)


def parse_openai_messages(
    messages: List[Message],
    images: Optional[List[Image.Image]] = None
) -> tuple[List[Dict], List[Image.Image]]:
    """
    解析 OpenAI 格式的消息，提取文本和图像
    支持两种格式：
    1. content 是字符串
    2. content 是列表，包含 text 和 image_url
    """
    parsed_messages = []
    image_list = []
    if images:
        image_list = images
    for msg in messages:
        role = msg.role
        content = msg.content
        if isinstance(content, str):
            # 纯文本消息
            parsed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # 多模态消息
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item["text"])
                elif item.get("type") == "image_url":
                    # 处理图像
                    image_url = item["image_url"]["url"]
                    if image_url.startswith("data:image"):
                        # base64 编码的图像
                        img = base64_to_pil(image_url)
                        image_list.append(img)
                        text_parts.append("<image>")
                    elif image_url.startswith("http"):
                        # URL 图像（需要下载）
                        import requests

                        response = requests.get(image_url)
                        img = Image.open(BytesIO(response.content))
                        image_list.append(img)
                        text_parts.append("<image>")
                    else:
                        # 本地文件路径
                        img = Image.open(image_url)
                        image_list.append(img)
                        text_parts.append("<image>")

            if text_parts:
                parsed_messages.append({
                    "role": role,
                    "content": "".join(text_parts)
                })
        else:
            parsed_messages.append({"role": role, "content": str(content)})

    return parsed_messages, image_list


def merge_params_with_config(request: ChatCompletionRequest) -> Dict:
    """
    合并请求参数和配置默认值
    请求参数优先级高于配置
    """
    cfg = get_config()

    params = {
        "temperature": (request.temperature if request.temperature is not None
                        else cfg.sampling.temperature),
        "topp":
        request.top_p if request.top_p is not None else cfg.sampling.top_p,
        "topk":
        request.top_k if request.top_k is not None else cfg.sampling.top_k,
        "repetition_penalty":
        (request.repetition_penalty if request.repetition_penalty is not None
         else cfg.sampling.repetition_penalty),
        "stream":
        request.stream,
    }

    return params


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的聊天补全接口"""
    global inference_service

    if inference_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    request_id = request.request_id or str(uuid.uuid4())

    try:
        # ✅ 处理 images 参数（如果有的话）
        external_images = []
        if request.images:
            for img_data in request.images:
                # img_data 只能是字符串（base64 或 URL）
                if not isinstance(img_data, str):
                    raise HTTPException(
                        status_code=400,
                        detail=
                        f"Invalid image data type: {type(img_data)}. Must be base64 string or URL.",
                    )

                # 处理不同的字符串格式
                if img_data.startswith("data:image"):
                    # data:image/jpeg;base64,xxx 格式
                    img = base64_to_pil(img_data)
                elif img_data.startswith("http://") or img_data.startswith(
                        "https://"):
                    # URL 格式
                    import requests

                    response = requests.get(img_data)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                else:
                    # 纯 base64 字符串（没有 data:image 前缀）
                    try:
                        img = base64_to_pil(img_data)
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to decode image: {str(e)}")

                external_images.append(img)

        # ✅ 解析消息，传入 external_images
        parsed_messages, images = parse_openai_messages(
            request.messages, external_images)

        if not images:
            raise HTTPException(
                status_code=400,
                detail=
                "No images found in messages. This model requires images.",
            )

        params = merge_params_with_config(request)
        logger.info(
            f"Request {request_id[:8]}: params={params}, images={len(images)}")

        inference_request = InferenceRequest(
            request_id=request_id,
            messages=parsed_messages,
            images=images,
            params=params,
        )

        result_queue = await inference_service.submit_request(inference_request
                                                              )

        if request.stream:
            return StreamingResponse(
                stream_generator(result_queue, request_id, request.model),
                media_type="text/event-stream",
            )
        else:
            result = await asyncio.get_event_loop().run_in_executor(
                None, result_queue.get)

            if result.get("status") != ErrorCode.SUCCESS:
                error_msg = result.get("error", "Unknown error")
                raise HTTPException(status_code=500, detail=error_msg)

            response = ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant",
                                        content=result["text"]),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=0,
                    completion_tokens=len(result["text"]),
                    total_tokens=len(result["text"]),
                ),
                prefill_time=result.get("prefill_time", 0.0),
                decode_tps=result.get("decode_tps", 0.0),
            )

            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_generator(result_queue: Queue, request_id: str, model: str):
    """流式响应生成器"""
    try:
        while True:
            # 非阻塞获取结果
            result = await asyncio.get_event_loop().run_in_executor(
                None, result_queue.get)

            if "error" in result:
                # 错误情况
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    model=model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": f"[ERROR]: {result['error']}"},
                            finish_reason="error",
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                break

            if result.get("done", False):
                # 结束
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    model=model,
                    choices=[
                        ChatCompletionStreamChoice(index=0,
                                                   delta={},
                                                   finish_reason="stop")
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                # 如果有性能数据，额外发送一个包含性能信息的chunk
                if "prefill_time" in result:
                    perf_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [],
                        "performance": {
                            "prefill_time": result.get("prefill_time", 0.0),
                            "decode_tps": result.get("decode_tps", 0.0),
                        },
                    }
                    yield f"data: {json.dumps(perf_chunk)}\n\n"

                yield "data: [DONE]\n\n"
                break

            # 正常的token
            text = result.get("text", "")
            if text:
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    model=model,
                    choices=[
                        ChatCompletionStreamChoice(index=0,
                                                   delta={"content": text},
                                                   finish_reason=None)
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Error in stream_generator: {e}")
        error_chunk = {
            "id":
            request_id,
            "object":
            "chat.completion.chunk",
            "created":
            int(time.time()),
            "model":
            model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": f"[STREAM ERROR]: {str(e)}"
                },
                "finish_reason": "error",
            }],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object":
        "list",
        "data": [{
            "id": "qwen2.5-vl",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mininglamp",
            "author": "ws",
        }],
    }


@app.get("/v1/queue")
async def queue_status():
    """查询队列状态"""
    global inference_service

    if inference_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    queue_size = inference_service.queue_manager.size()

    return {
        "queue_size": queue_size,
        "estimated_wait_seconds": queue_size * 3,  # 假设每个请求 3 秒
        "status": "idle" if queue_size == 0 else "busy",
    }


if __name__ == "__main__":

    # 只保留配置文件路径参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    # 加载配置
    config = load_config(args.config)

    # 启动服务
    logger.info(
        f"Starting server on {config.server.host}:{config.server.port}")
    uvicorn.run(app,
                host=config.server.host,
                port=config.server.port,
                log_level="info")
