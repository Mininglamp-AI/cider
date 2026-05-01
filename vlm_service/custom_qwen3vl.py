from typing import Optional
import mlx.core as mx
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.qwen3_vl.language import LanguageModel as _OrigLanguageModel
import mlx_vlm.models.qwen3_vl as qwen_mod
import mlx.nn as nn
from tqdm import tqdm
from typing import Dict, Any, Callable, Tuple, List, Generator, Union
import functools
from mlx_vlm.generate import (
    maybe_quantize_kv_cache,
    normalize_resize_shape,
    wired_limit,
)
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_vlm.models import cache
from mlx_vlm.utils import StoppingCriteria, ThinkingBudgetCriteria, prepare_inputs
import time
from .code_def import CustomGenerationResult, ErrorCode
from transformers import PreTrainedTokenizer
import numpy as np

MAX_PROMPT_TOKENS = 8192


def find_vis_token_positions(input_ids, target_token=151655):
    if input_ids.ndim == 2:
        input_ids = input_ids[0]

    # 找到所有匹配的位置
    mask = input_ids == target_token  # [N] bool array
    start_idx = mx.argmax(mask)
    n_tokens = mx.sum(mask)
    end_idx = start_idx + n_tokens
    return start_idx.item(), end_idx.item()


class CustomLanguageModel(_OrigLanguageModel):
    """Fix get_rope_index for multi-image: upstream uses mx.sum of vision_start
    indices which collapses to a wrong scalar when >1 image is present.
    Replaced with vectorized shifted-pair matching.

    Also fix __call__ for chunked prefill: upstream else-branch uses simple
    arange+delta which is wrong for multi-image. When _position_ids is
    pre-computed, always slice from it regardless of cache_offset."""

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds=None,
        mask=None,
        cache=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        from mlx_vlm.models.qwen3_vl.language import LanguageModelOutput

        n_to_process = kwargs.get("n_to_process", None)
        if n_to_process is not None:
            visual_pos_masks = (
                visual_pos_masks[:, n_to_process:]
                if visual_pos_masks is not None
                else None
            )

        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values is not None:
            self._rope_deltas = None
            self._position_ids = None

        cache_offset = 0
        if cache and cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        rope_mask = mask
        if mask is not None and mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
            if (
                self._position_ids is not None
                and cache_offset + inputs.shape[1] <= self._position_ids.shape[2]
            ):
                # Slice from pre-computed position_ids (fixes chunked prefill)
                seq_length = inputs.shape[1]
                position_ids = self._position_ids[
                    :, :, cache_offset : cache_offset + seq_length
                ]
            elif (
                (cache is not None and cache[0] is not None and cache_offset == 0)
                or self._rope_deltas is None
                or cache is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    inputs, image_grid_thw, video_grid_thw, rope_mask
                )
                self._rope_deltas = rope_deltas
                self._position_ids = position_ids
            else:
                batch_size, seq_length = inputs.shape
                delta = mx.array(
                    cache_offset + self._rope_deltas if cache is not None else 0
                )
                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
                if cache_offset is not None:
                    if delta.ndim == 0:
                        delta = mx.expand_dims(delta, axis=0)
                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]
                position_ids = mx.add(position_ids, delta)[None, ...]
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones(
                (3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = mx.where(
                    attention_mask[i] == 1, input_ids, mx.zeros_like(input_ids)
                )
                # Vectorized shifted-pair matching (fixes multi-image bug)
                vs_mask = input_ids[:-1] == vision_start_token_id
                next_toks = input_ids[1:]
                image_nums = int(mx.sum(vs_mask & (next_toks == image_token_id)).item())
                video_nums = int(mx.sum(vs_mask & (next_toks == video_token_id)).item())
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len))
                    index = index + st_idx
                    llm_pos_ids_list.append(index)
                    t_index = mx.arange(llm_grid_t).reshape(llm_grid_t, 1)
                    t_index = mx.broadcast_to(
                        t_index, (llm_grid_t, llm_grid_h * llm_grid_w)
                    )
                    t_index = t_index.flatten()
                    h_index = mx.arange(llm_grid_h).reshape(1, llm_grid_h, 1)
                    h_index = mx.broadcast_to(
                        h_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )
                    h_index = h_index.flatten()
                    w_index = mx.arange(llm_grid_w).reshape(1, 1, llm_grid_w)
                    w_index = mx.broadcast_to(
                        w_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )
                    w_index = w_index.flatten()
                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    t_index = mx.arange(text_len).reshape(1, text_len)
                    t_index = mx.broadcast_to(t_index, (3, text_len))
                    llm_pos_ids_list.append(t_index + st_idx)
                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                mask = mx.array(attention_mask[i] == 1)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(
                    expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
                )
                updated_position_ids = mx.concatenate(
                    [
                        position_ids[:, :i, :],
                        new_positions,
                        position_ids[:, i + 1 :, :],
                    ],
                    axis=1,
                )
                position_ids = updated_position_ids
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = mx.array(mrope_position_deltas)[0]
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
                position_ids = mx.where(
                    attention_mask == 0, mx.ones_like(position_ids), position_ids
                )
                position_ids = mx.expand_dims(position_ids[0], axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
                max_position_ids = position_ids.max(0, keepdims=False)[0].max(
                    -1, keepdims=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
                position_ids = mx.broadcast_to(
                    position_ids, (3, input_ids.shape[0], input_ids.shape[1])
                )
                mrope_position_deltas = mx.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas


class CustomQwen3VLModel(qwen_mod.Model):

    def __init__(self, config: qwen_mod.ModelConfig):
        nn.Module.__init__(config)
        self.enable_pruning = False
        self.config = config
        self.vision_tower = qwen_mod.VisionModel(config.vision_config)
        # Replace upstream LanguageModel with fixed version
        self.language_model = CustomLanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            # Reset position state for text-only generation
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        # 获取缓存的特征
        buf_vis_features = kwargs.get("buf_vis_features", None)
        buf_vis_stack_features = kwargs.get("buf_vis_stack_features", None)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states, deepstack_image_embeds = self.vision_tower(
            pixel_values, grid_thw
        )
        if buf_vis_features is not None and buf_vis_stack_features is not None:
            # 检查 buf_vis_stack_features 是否是包含3个元素的列表
            assert isinstance(buf_vis_features, list) and isinstance(
                buf_vis_stack_features, list
            )
            if len(buf_vis_stack_features) == 3 and len(buf_vis_features) > 0:

                # 1. concat hidden_states (shape: (1, seq_len, d))
                # buf_vis_features 应该也是 (1, seq_len', d)
                axis = 0
                assert hidden_states.ndim == buf_vis_features[0].ndim
                if hidden_states.ndim == 3:
                    axis = 1
                hidden_states = mx.concatenate(
                    [buf_vis_features[0], hidden_states], axis=axis
                )

                # 就地更新 buf_vis_features
                buf_vis_features[0] = hidden_states

                # 2. concat deepstack_image_embeds (列表，每个元素 shape: (seq_len, d))
                # 对应位置的元素进行concat
                for i in range(3):
                    if i < len(deepstack_image_embeds):
                        # 在 axis=0 (seq_len维度) 上concat
                        axis = 0 if deepstack_image_embeds[i].ndim == 2 else 1
                        concatenated = mx.concatenate(
                            [buf_vis_stack_features[i], deepstack_image_embeds[i]],
                            axis=axis,
                        )
                        # 就地覆盖 buf_vis_stack_features
                        buf_vis_stack_features[i] = concatenated

                # 使用concat后的特征
                deepstack_image_embeds = buf_vis_stack_features
            else:
                buf_vis_features.append(hidden_states[:])  # 或者 hidden_states.copy()
                for embed in deepstack_image_embeds:
                    buf_vis_stack_features.append(embed[:])  # 或者 embed.copy()
        visual_pos_masks = None
        deepstack_visual_embeds = None

        # Insert special image tokens in the input_ids
        inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        mx.eval(deepstack_image_embeds)
        deepstack_visual_embeds = deepstack_image_embeds

        # Pre-calculate position_ids for chunked prefill
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        kwargs.update(
            {
                "pixel_values": pixel_values,
                **input_embeddings_features.to_dict(),
            }
        )

        logits = self.language_model(input_ids, mask=mask, cache=cache, **kwargs)
        return logits


qwen_mod.Model = CustomQwen3VLModel


def custom_generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[Any]] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prefill_step_size: Optional[int] = 2048,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        input_ids (mx.array): The input prompt token ids.
        model (nn.Module): The model to use for generation.
        pixel_values: The pixel values for vision models (optional).
        mask: The attention mask (optional).
        max_tokens (int): Maximum number of tokens to generate. Default: ``256``.
        temperature (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nucleus sampling, higher means model considers
          more less likely words.
        logit_bias (dictionary, optional): Additive logit bias.
        prompt_cache (list, optional): Pre-existing KV cache for the prompt.
        max_kv_size (int, optional): Maximum KV cache size.
        kv_bits (int, optional): Number of bits for KV cache quantization.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Start index for quantized KV cache. Default: ``0``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        prefill_step_size (int): Number of tokens to process per prefill step.
          Chunked prefill processes prompts in smaller chunks to reduce peak
          memory usage. Default: ``2048``.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    if sampler is None:
        sampler = make_sampler(temperature, top_p)

    processors = make_logits_processors(
        logit_bias, repetition_penalty, repetition_context_size
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    y = input_ids
    tokens = mx.array([], dtype=input_ids.dtype)

    thinking_budget_criteria = kwargs.pop("thinking_budget_criteria", None)

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    def _step(y, inputs_embeds=None):
        nonlocal tokens, kwargs

        with mx.stream(mx.gpu):
            if "decoder_input_ids" in kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **kwargs,
                )
            else:
                outputs = model.language_model(
                    y,
                    inputs_embeds=inputs_embeds,
                    cache=prompt_cache,
                    **kwargs,
                )

            logits = outputs.logits[:, -1, :]

            if len(processors) > 0 and len(y) > 0:
                tokens = mx.concat([tokens, y.flatten()])

                for processor in processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits)
            y = sampler(logprobs)

            if outputs.cross_attention_states is not None:
                kwargs = {"cross_attention_states": outputs.cross_attention_states}
            elif outputs.encoder_outputs is not None:
                kwargs = {"encoder_outputs": outputs.encoder_outputs}
            else:
                kwargs = {}

            return y, logprobs.squeeze(0)

    with mx.stream(mx.gpu):

        # Get input embeddings (handles both multimodal and text-only)
        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        inputs_embeds = embedding_output.inputs_embeds

        kwargs.update(
            {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )
        if prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size:
            # Pre-compute RoPE position_ids with FULL input_ids before chunking.
            # Without this, get_rope_index sees only the first chunk and computes
            # wrong positions when vision tokens span across chunk boundaries.
            image_grid_thw = kwargs.get("image_grid_thw", None)
            video_grid_thw = kwargs.get("video_grid_thw", None)
            position_ids, rope_deltas = model.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, None
            )
            model.language_model._position_ids = position_ids
            model.language_model._rope_deltas = rope_deltas

            # Remove pixel_values/image_grid_thw from kwargs to prevent
            # language_model.__call__ from resetting _position_ids/_rope_deltas
            # (it clears them when pixel_values is not None).
            # Vision encoding is already done in get_input_embeddings above.
            kwargs.pop("pixel_values", None)
            kwargs.pop("image_grid_thw", None)
            kwargs.pop("video_grid_thw", None)

            # Chunked prefill with embeddings
            # Must slice visual_pos_masks and deepstack per-chunk to match h's seq_len
            full_vpm = kwargs.pop("visual_pos_masks", None)
            full_dse = kwargs.pop("deepstack_visual_embeds", None)
            total_tokens = inputs_embeds.shape[1]
            chunk_offset = 0
            with tqdm(total=total_tokens, desc="Prefill", unit="tok") as pbar:
                while inputs_embeds.shape[1] > 1:
                    n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                    # Slice visual_pos_masks for this chunk
                    chunk_vpm = None
                    chunk_dse = None
                    if full_vpm is not None:
                        chunk_vpm = full_vpm[
                            :, chunk_offset : chunk_offset + n_to_process
                        ]
                    if full_dse is not None:
                        # deepstack embeds need to be sliced to match image tokens in this chunk
                        # Count image token positions in this chunk from full_vpm
                        if chunk_vpm is not None:

                            chunk_mask = np.array(chunk_vpm[0])
                            n_img_in_chunk = int(chunk_mask.sum())
                            if n_img_in_chunk > 0:
                                # Find the global image token index range for this chunk
                                full_mask = np.array(full_vpm[0])
                                img_before = int(full_mask[:chunk_offset].sum())
                                chunk_dse = [
                                    e[img_before : img_before + n_img_in_chunk]
                                    for e in full_dse
                                ]
                            else:
                                chunk_dse = None
                    model.language_model(
                        inputs=input_ids[:, :n_to_process],
                        inputs_embeds=inputs_embeds[:, :n_to_process],
                        cache=prompt_cache,
                        visual_pos_masks=chunk_vpm,
                        deepstack_visual_embeds=chunk_dse,
                        **kwargs,
                    )
                    quantize_cache_fn(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                    chunk_offset += n_to_process
                    inputs_embeds = inputs_embeds[:, n_to_process:]
                    input_ids = input_ids[:, n_to_process:]
                    mx.clear_cache()
                    pbar.update(n_to_process)

            # Last token goes through _step; need remaining vpm/dse
            if full_vpm is not None:
                kwargs["visual_pos_masks"] = full_vpm[:, chunk_offset:]
                if full_dse is not None:
                    full_mask = np.array(full_vpm[0])
                    img_before = int(full_mask[:chunk_offset].sum())
                    n_img_remain = int(full_mask[chunk_offset:].sum())
                    if n_img_remain > 0:
                        kwargs["deepstack_visual_embeds"] = [
                            e[img_before : img_before + n_img_remain] for e in full_dse
                        ]
            input_ids = input_ids[:, -1:]

        y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)

    mx.async_eval(y)

    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y[None])
            mx.async_eval(next_y)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break

        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()

        if thinking_budget_criteria is not None:
            next_y = thinking_budget_criteria.apply_forced_token(next_y)
        y, logprobs = next_y, next_logprobs
        n += 1


def custom_stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        processor (PreTrainedTokenizer): The tokenizer/processor.
        prompt (str): The input prompt text.
        image (Union[str, List[str]], optional): Image path(s) or URL(s).
        audio (Union[str, List[str]], optional): Audio file path(s).
        prefill_step_size (int, optional): Number of tokens to process per prefill
          step. When set, enables chunked prefill which processes long prompts in
          smaller chunks to reduce peak memory usage.
        kwargs: Additional options passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[GenerationResult]: A generator producing GenerationResult objects
          containing the generated text, tokens, and statistics.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Set up thinking budget criteria if requested
    thinking_budget = kwargs.pop("thinking_budget", None)
    thinking_end_token = kwargs.pop("thinking_end_token", "<think>")
    thinking_start_token = kwargs.pop("thinking_start_token", "</think>")
    enable_thinking = kwargs.pop("enable_thinking", False)

    # Skip special tokens
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    skip_special_token_ids = (
        set(tokenizer.all_special_ids)
        if skip_special_tokens and hasattr(tokenizer, "all_special_ids")
        else []
    )

    add_special_tokens = (
        not hasattr(processor, "chat_template")
        if model.config.model_type in ["gemma3", "gemma3n", "gemma4"]
        else True
    )

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    image_token_index = getattr(model.config, "image_token_index", None)

    if kwargs.get("input_ids", None) is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
    else:
        inputs = prepare_inputs(
            processor,
            images=image,
            audio=audio,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        input_ids = inputs.get("input_ids", None)
        pixel_values = inputs.get("pixel_values", None)
        mask = inputs.get("attention_mask", None)
        data_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        kwargs.update(data_kwargs)

    if thinking_budget is not None:
        thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        enable_thinking = enable_thinking and (
            thinking_start_token_id in input_ids.flatten().tolist()
        )
        tokenizer.thinking_budget_criteria = ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=thinking_budget,
            thinking_end_token=thinking_end_token,
            thinking_start_token=thinking_start_token,
            enable_thinking=enable_thinking,
        )
        kwargs["thinking_budget_criteria"] = tokenizer.thinking_budget_criteria
    else:
        tokenizer.thinking_budget_criteria = None

    if input_ids.size > MAX_PROMPT_TOKENS:
        # yield一个错误结果并直接返回
        yield CustomGenerationResult(
            text=f"Error: Input token length ({input_ids.size}) exceeds maximum allowed ({MAX_PROMPT_TOKENS})",
            token=None,
            logprobs=None,
            prompt_tokens=input_ids.size,
            generation_tokens=0,
            total_tokens=input_ids.size,
            prompt_tps=0.0,
            generation_tps=0.0,
            peak_memory=0.0,
            code=ErrorCode.PREFILL_TOO_LONG,  # 可以添加一个error标志
        )
        return  # 直接返回，不进行生成

    with wired_limit(model):
        detokenizer = processor.detokenizer
        detokenizer.reset()
        thinking_criteria = getattr(tokenizer, "thinking_budget_criteria", None)
        gen = custom_generate_step(input_ids, model, pixel_values, mask, **kwargs)
        tic = time.perf_counter()

        for n, (token, logprobs) in enumerate(gen):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = input_ids.size / prompt_time
                tic = time.perf_counter()

            # Check thinking budget and force token if needed
            if thinking_criteria is not None:
                thinking_criteria(token)

            # Stop generation if the token is in the eos_token_ids
            if tokenizer.stopping_criteria(token):
                break

            detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)

            # Yield the last segment if streaming
            yield CustomGenerationResult(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=input_ids.size,
                generation_tokens=n + 1,
                total_tokens=input_ids.size + n + 1,
                prompt_tps=prompt_tps,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
            )

        detokenizer.finalize()
        yield CustomGenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=input_ids.size,
            generation_tokens=n + 1,
            total_tokens=input_ids.size + n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
        )
        # Cleanup after generation
        mx.clear_cache()


def custom_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    **kwargs,
):
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temperature (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """

    text = ""
    last_response = None

    eos_tokens = kwargs.get("eos_tokens", None)
    stopping_criteria = kwargs.get("stopping_criteria", None)

    # Get the tokenizer
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Add custom EOS tokens to the stopping criteria
    if eos_tokens is not None:
        tokenizer.stopping_criteria.add_eos_token_ids(eos_tokens)

    # Use custom stopping criteria
    elif stopping_criteria is not None:
        if isinstance(stopping_criteria, StoppingCriteria) or callable(
            stopping_criteria
        ):
            tokenizer.stopping_criteria = stopping_criteria
        else:
            raise ValueError(
                "stopping_criteria must be an instance of StoppingCriteria or a callable"
            )
    else:
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

    for response in custom_stream_generate(
        model, processor, prompt, image, audio, **kwargs
    ):
        text += response.text
        last_response = response

    if len(text) == 0:
        return CustomGenerationResult(
            text=text,
            token=None,
            logprobs=None,
            prompt_tokens=0,
            generation_tokens=0,
            total_tokens=0,
            prompt_tps=0.0,
            generation_tps=0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
        )

    return CustomGenerationResult(
        text=text,
        token=last_response.token,
        logprobs=last_response.logprobs,
        prompt_tokens=last_response.prompt_tokens,
        generation_tokens=last_response.generation_tokens,
        total_tokens=last_response.total_tokens,
        prompt_tps=last_response.prompt_tps,
        generation_tps=last_response.generation_tps,
        peak_memory=last_response.peak_memory,
    )
