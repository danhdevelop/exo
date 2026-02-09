import base64
import json
import re
import time
from collections.abc import Generator
from functools import cache
from typing import Any, Callable, Literal

import mlx.core as mx
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)
from pydantic import ValidationError

from exo.shared.constants import EXO_MAX_CHUNK_SIZE
from exo.shared.models.model_cards import ModelId, ModelTask
from exo.shared.types.api import ChatCompletionMessageText, ImageGenerationStats
from exo.shared.types.chunks import ErrorChunk, ImageChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ChatCompletion,
    ConnectToGroup,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ImageGenerationResponse,
    PartialImageResponse,
    ToolCallItem,
    ToolCallResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.image import (
    DistributedImageModel,
    generate_image,
    initialize_image_model,
    warmup_image_generator,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
)
from exo.worker.runner.bootstrap import logger

# Note: Tinygrad engine support was previously here but has been temporarily
# removed in favor of upstream's MLX-only implementation. May be re-added later.


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
    instance, runner_id, shard_metadata = (
        bound_instance.instance,
        bound_instance.bound_runner_id,
        bound_instance.bound_shard,
    )
    device_rank = shard_metadata.device_rank
    logger.info("hello from the runner")
    if getattr(shard_metadata, "immediate_exception", False):
        raise Exception("Fake exception - runner failed to spin up.")
    if timeout := getattr(shard_metadata, "should_timeout", 0):
        time.sleep(timeout)

    setup_start_time = time.time()

    model: Model | DistributedImageModel | None = None
    tokenizer = None
    group = None
    kv_prefix_cache: KVPrefixCache | None = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )
    seen = set[TaskId]()
    with task_receiver as tasks:
        for task in tasks:
            if task.task_id in seen:
                logger.warning("repeat task - potential error")
            seen.add(task.task_id)
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
            event_sender.send(TaskAcknowledged(task_id=task.task_id))
            match task:
                case ConnectToGroup() if isinstance(
                    current_status, (RunnerIdle, RunnerFailed)
                ):
                    logger.info("runner connecting")
                    current_status = RunnerConnecting()
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    group = initialize_mlx(bound_instance)
                    logger.info("runner connected")
                    current_status = RunnerConnected()

                # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
                case LoadModel() if (
                    isinstance(current_status, RunnerConnected) and group is not None
                ) or (isinstance(current_status, RunnerIdle) and group is None):
                    current_status = RunnerLoading()
                    logger.info("runner loading")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    def on_model_load_timeout() -> None:
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id,
                                runner_status=RunnerFailed(
                                    error_message="Model loading timed out"
                                ),
                            )
                        )
                        time.sleep(0.5)

                    if ModelTask.TextGeneration in shard_metadata.model_card.tasks:
                        model, tokenizer = load_mlx_items(
                            bound_instance, group, on_timeout=on_model_load_timeout
                        )
                        logger.info(
                            f"model has_tool_calling={tokenizer.has_tool_calling}"
                        )
                        model_id = bound_instance.bound_shard.model_card.model_id
                        kv_prefix_cache = KVPrefixCache(tokenizer, group, model_id=model_id)

                    elif (
                        ModelTask.TextToImage in shard_metadata.model_card.tasks
                        or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                    ):
                        model = initialize_image_model(bound_instance)
                    else:
                        raise ValueError(
                            f"Unknown model task(s): {shard_metadata.model_card.tasks}"
                        )
                    current_status = RunnerLoaded()
                    logger.info("runner loaded")
                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    assert model

                    current_status = RunnerWarmingUp()
                    logger.info("runner warming up")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    logger.info(f"warming up inference for instance: {instance}")
                    if ModelTask.TextGeneration in shard_metadata.model_card.tasks:
                        assert not isinstance(model, DistributedImageModel)
                        assert tokenizer

                        toks = warmup_inference(
                            model=model,
                            tokenizer=tokenizer,
                            # kv_prefix_cache=kv_prefix_cache,  # supply for warmup-time prefix caching
                        )
                        logger.info(f"warmed up by generating {toks} tokens")
                        logger.info(
                            f"runner initialized in {time.time() - setup_start_time} seconds"
                        )
                    elif (
                        ModelTask.TextToImage in shard_metadata.model_card.tasks
                        or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                    ):
                        assert isinstance(model, DistributedImageModel)
                        image = warmup_image_generator(model=model)
                        if image is not None:
                            logger.info(f"warmed up by generating {image.size} image")
                        else:
                            logger.info("warmup completed (non-primary node)")

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ChatCompletion(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    logger.info(f"received chat request: {task}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    assert model and not isinstance(model, DistributedImageModel)
                    assert tokenizer
                    assert task_params.messages[0].content is not None

                    try:
                        _check_for_debug_prompts(task_params.messages[0].content)

                        # Build prompt once - used for both generation and thinking detection
                        prompt = apply_chat_template(tokenizer, task_params)

                        # Generate responses using the actual MLX generation
                        mlx_generator = mlx_generate(
                            model=model,
                            tokenizer=tokenizer,
                            task=task_params,
                            prompt=prompt,
                            kv_prefix_cache=kv_prefix_cache,
                        )

                        # For other thinking models (GLM, etc.), check if we need to
                        # prepend the thinking tag that was consumed by the chat template
                        if detect_thinking_prompt_suffix(prompt, tokenizer):
                            mlx_generator = parse_thinking_models(
                                mlx_generator, tokenizer
                            )

                        # Kimi-K2 has tool call sections and K2-specific tool format
                        # Only apply K2 patching for K2 models, NOT Kimi VL (vision-language)
                        if "kimi" in shard_metadata.model_card.model_id.lower() and "k2" in shard_metadata.model_card.model_id.lower():
                            mlx_generator = filter_kimi_tokens(mlx_generator)
                            patch_kimi_tokenizer(tokenizer)

                        # GLM models need patched parser (upstream has bug with None regex match)
                        elif "glm" in shard_metadata.model_card.model_id.lower():
                            patch_glm_tokenizer(tokenizer)

                        # Qwen models need XML tool call parser
                        elif "qwen" in shard_metadata.model_card.model_id.lower():
                            patch_qwen_tokenizer(tokenizer)

                        # GPT-OSS specific parsing to match other model formats.
                        elif isinstance(model, GptOssModel):
                            mlx_generator = parse_gpt_oss(mlx_generator)
                        # Qwen thinking models parsing
                        elif isinstance(model, (Qwen3NextModel, Qwen3MoeModel)):
                            # Check if it's a thinking model (model name contains "thinking")
                            # For now, apply parse_qwen_thinking
                            mlx_generator = parse_qwen_thinking(mlx_generator)

                        if tokenizer.has_tool_calling and not isinstance(
                            model, GptOssModel
                        ):
                            assert tokenizer.tool_call_start
                            assert tokenizer.tool_call_end
                            assert tokenizer.tool_parser  # pyright: ignore[reportAny]
                            mlx_generator = parse_tool_calls(
                                mlx_generator,
                                tokenizer.tool_call_start,
                                tokenizer.tool_call_end,
                                tokenizer.tool_parser,  # pyright: ignore[reportAny]
                            )

                        completion_tokens = 0
                        for response in mlx_generator:
                            match response:
                                case GenerationResponse():
                                    completion_tokens += 1
                                    if (
                                        device_rank == 0
                                        and response.finish_reason == "error"
                                    ):
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=ErrorChunk(
                                                    error_message=response.text,
                                                    model=shard_metadata.model_card.model_id,
                                                ),
                                            )
                                        )

                                    elif device_rank == 0:
                                        assert response.finish_reason not in (
                                            "error",
                                            "tool_calls",
                                            "function_call",
                                        )
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=TokenChunk(
                                                    model=shard_metadata.model_card.model_id,
                                                    text=response.text,
                                                    token_id=response.token,
                                                    usage=response.usage,
                                                    finish_reason=response.finish_reason,
                                                    stats=response.stats,
                                                ),
                                            )
                                        )
                                case ToolCallResponse():
                                    if device_rank == 0:
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=ToolCallChunk(
                                                    tool_calls=response.tool_calls,
                                                    model=shard_metadata.model_card.model_id,
                                                    usage=response.usage,
                                                ),
                                            )
                                        )

                    # can we make this more explicit?
                    except Exception as e:
                        if device_rank == 0:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=ErrorChunk(
                                        model=shard_metadata.model_card.model_id,
                                        finish_reason="error",
                                        error_message=str(e),
                                    ),
                                )
                            )
                        raise

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ImageGeneration(
                    task_params=task_params, command_id=command_id
                ) if isinstance(current_status, RunnerReady):
                    assert isinstance(model, DistributedImageModel)
                    logger.info(f"received image generation request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    try:
                        # Generate images using the image generation backend
                        # Track image_index for final images only
                        image_index = 0
                        for response in generate_image(model=model, task=task_params):
                            if (
                                shard_metadata.device_rank
                                == shard_metadata.world_size - 1
                            ):
                                match response:
                                    case PartialImageResponse():
                                        logger.info(
                                            f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                        )
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                    case ImageGenerationResponse():
                                        logger.info("sending final ImageChunk")
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                        image_index += 1
                    # can we make this more explicit?
                    except Exception as e:
                        if shard_metadata.device_rank == shard_metadata.world_size - 1:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=ErrorChunk(
                                        model=shard_metadata.model_card.model_id,
                                        finish_reason="error",
                                        error_message=str(e),
                                    ),
                                )
                            )
                        raise

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ImageEdits(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    assert isinstance(model, DistributedImageModel)
                    logger.info(f"received image edits request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    try:
                        image_index = 0
                        for response in generate_image(model=model, task=task_params):
                            if (
                                shard_metadata.device_rank
                                == shard_metadata.world_size - 1
                            ):
                                match response:
                                    case PartialImageResponse():
                                        logger.info(
                                            f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                        )
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                    case ImageGenerationResponse():
                                        logger.info("sending final ImageChunk")
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                        image_index += 1
                    except Exception as e:
                        if shard_metadata.device_rank == shard_metadata.world_size - 1:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=ErrorChunk(
                                        model=shard_metadata.model_card.model_id,
                                        finish_reason="error",
                                        error_message=str(e),
                                    ),
                                )
                            )
                        raise

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case Shutdown():
                    current_status = RunnerShuttingDown()
                    logger.info("runner shutting down")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    current_status = RunnerShutdown()
                case _:
                    raise ValueError(
                        f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                    )
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete)
            )
            event_sender.send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
            )
            if isinstance(current_status, RunnerShutdown):
                del model, tokenizer, group
                mx.clear_cache()
                import gc

                gc.collect()
                break


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def filter_kimi_tokens(
    responses: Generator[GenerationResponse | ToolCallResponse],
) -> Generator[GenerationResponse]:
    for resp in responses:
        assert isinstance(resp, GenerationResponse)
        if (
            resp.text == "<|tool_calls_section_begin|>"
            or resp.text == "<|tool_calls_section_end|>"
        ):
            continue
        yield resp


def parse_gpt_oss(
    responses: Generator[GenerationResponse | ToolCallResponse],
) -> Generator[GenerationResponse | ToolCallResponse]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []
    token_count = 0
    parser_active = False
    first_error_logged = False

    for response in responses:
        assert isinstance(response, GenerationResponse)
        token_count += 1
        stream.process(response.token)

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=current_tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = recipient

        # If inside a tool call, accumulate arguments
        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            continue

        try:
            stream.process(response.token)
            parser_active = True

            delta = stream.last_content_delta
            ch = stream.current_channel

            if ch == "analysis" and not thinking:
                thinking = True
                yield response.model_copy(update={"text": "<think>"})

            if ch != "analysis" and thinking:
                thinking = False
                yield response.model_copy(update={"text": "</think>"})

            if delta:
                yield response.model_copy(update={"text": delta})

        except Exception as e:
            # If parser fails on first few tokens, the model isn't using Harmony format
            # This happens when the chat template doesn't produce proper Harmony format prompt
            if not parser_active and token_count <= 3:
                if not first_error_logged:
                    logger.warning(
                        f"GPT-OSS parser failed on token {token_count}: {e}. "
                        f"Model is not generating in Harmony format. "
                        f"Falling back to passthrough mode. "
                        f"This usually means the chat template didn't include Harmony format tokens."
                    )
                    first_error_logged = True
                use_passthrough = True
                yield response
            elif parser_active:
                # If parser was active and then failed, this is a real error
                logger.error(f"GPT-OSS parser failed after {token_count} tokens: {e}")
                raise
            else:
                # Parser never activated, keep trying for a few more tokens
                if token_count > 10:
                    logger.warning(
                        f"GPT-OSS parser hasn't activated after {token_count} tokens. "
                        f"Switching to passthrough mode. Error: {e}"
                    )
                    use_passthrough = True
                yield response

        if response.finish_reason is not None:
            if thinking:
                yield response.model_copy(update={"text": "</think>"})
            yield response


def parse_thinking_models(
    responses: Generator[GenerationResponse | ToolCallResponse],
    tokenizer: TokenizerWrapper,
) -> Generator[GenerationResponse | ToolCallResponse]:
    """
    For models that inject thinking tags in the prompt (like GLM-4.7),
    prepend the thinking tag to the output stream so the frontend
    can properly parse thinking content.
    """
    first = True
    for response in responses:
        if isinstance(response, ToolCallResponse):
            yield response
            continue
        if first:
            first = False
            yield response.model_copy(
                update={
                    "text": tokenizer.think_start,
                    "token": tokenizer.think_start_id,  # type: ignore
                }
            )
        yield response


def _send_image_chunk(
    encoded_data: str,
    command_id: CommandId,
    model_id: ModelId,
    event_sender: MpSender[Event],
    image_index: int,
    is_partial: bool,
    partial_index: int | None = None,
    total_partials: int | None = None,
    stats: ImageGenerationStats | None = None,
    image_format: Literal["png", "jpeg", "webp"] | None = None,
) -> None:
    """Send base64-encoded image data as chunks via events."""
    data_chunks = [
        encoded_data[i : i + EXO_MAX_CHUNK_SIZE]
        for i in range(0, len(encoded_data), EXO_MAX_CHUNK_SIZE)
    ]
    total_chunks = len(data_chunks)
    for chunk_index, chunk_data in enumerate(data_chunks):
        # Only include stats on the last chunk of the final image
        chunk_stats = (
            stats if chunk_index == total_chunks - 1 and not is_partial else None
        )
        event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=ImageChunk(
                    model=model_id,
                    data=chunk_data,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    image_index=image_index,
                    is_partial=is_partial,
                    partial_index=partial_index,
                    total_partials=total_partials,
                    stats=chunk_stats,
                    format=image_format,
                ),
            )
        )


def _process_image_response(
    response: ImageGenerationResponse | PartialImageResponse,
    command_id: CommandId,
    shard_metadata: ShardMetadata,
    event_sender: MpSender[Event],
    image_index: int,
) -> None:
    """Process a single image response and send chunks."""
    encoded_data = base64.b64encode(response.image_data).decode("utf-8")
    is_partial = isinstance(response, PartialImageResponse)
    # Extract stats from final ImageGenerationResponse if available
    stats = response.stats if isinstance(response, ImageGenerationResponse) else None
    _send_image_chunk(
        encoded_data=encoded_data,
        command_id=command_id,
        model_id=shard_metadata.model_card.model_id,
        event_sender=event_sender,
        image_index=response.image_index,
        is_partial=is_partial,
        partial_index=response.partial_index if is_partial else None,
        total_partials=response.total_partials if is_partial else None,
        stats=stats,
        image_format=response.format,
    )


def parse_tool_calls(
    responses: Generator[GenerationResponse | ToolCallResponse],
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> Generator[GenerationResponse | ToolCallResponse]:
    """Parse tool calls from a token stream.

    Uses buffer-based string matching so tool_call_start / tool_call_end
    do NOT need to be single tokens.  Handles multi-token start/end tags,
    text before/after tool calls, and incomplete tool calls gracefully.
    """
    in_tool_call = False
    tool_call_buf = ""       # text accumulated inside <tool_call>…</tool_call>
    text_buf = ""            # text accumulated outside, checked for start marker
    last_resp: GenerationResponse | None = None

    def _make_text(text: str, ref: GenerationResponse) -> GenerationResponse:
        """Clone *ref* with only the text changed (clear finish_reason)."""
        return GenerationResponse(
            text=text,
            token=ref.token,
            finish_reason=None,
            stats=None,
            tool_calls=None,
            usage=ref.usage,
        )

    def _try_parse(raw: str, ref: GenerationResponse):
        """Attempt to parse tool-call text; yield result or fallback."""
        try:
            parsed = tool_parser(raw.strip())
            logger.info(f"parsed tool_call text into {parsed=}")
            if isinstance(parsed, list):
                tools = [_validate_single_tool(t) for t in parsed]
            else:
                tools = [_validate_single_tool(parsed)]
            return ToolCallResponse(tool_calls=tools, usage=ref.usage)
        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            AttributeError,
        ) as e:
            logger.opt(exception=e).warning("tool call parsing failed")
            return _make_text(tool_call_start + raw + tool_call_end, ref)

    for response in responses:
        assert isinstance(response, GenerationResponse)
        last_resp = response

        # ── inside a tool call: accumulate until end marker ──
        if in_tool_call:
            tool_call_buf += response.text

            if tool_call_end in tool_call_buf:
                end_idx = tool_call_buf.index(tool_call_end)
                tool_text = tool_call_buf[:end_idx]
                after_end = tool_call_buf[end_idx + len(tool_call_end):]

                yield _try_parse(tool_text, response)

                in_tool_call = False
                tool_call_buf = ""

                # feed leftover text back through the outer-loop logic
                if after_end:
                    text_buf += after_end
                    # check immediately for another tool call
                    if tool_call_start in text_buf:
                        si = text_buf.index(tool_call_start)
                        prefix = text_buf[:si]
                        if prefix:
                            yield _make_text(prefix, response)
                        tool_call_buf = text_buf[si + len(tool_call_start):]
                        text_buf = ""
                        in_tool_call = True
            continue

        # ── outside a tool call ──
        text_buf += response.text

        # full start marker found
        if tool_call_start in text_buf:
            si = text_buf.index(tool_call_start)
            prefix = text_buf[:si]
            after_start = text_buf[si + len(tool_call_start):]

            if prefix:
                yield _make_text(prefix, response)

            in_tool_call = True
            tool_call_buf = after_start
            text_buf = ""
            continue

        # check whether the tail of text_buf could be a *partial* start marker
        partial_len = 0
        for i in range(min(len(text_buf), len(tool_call_start)), 0, -1):
            if text_buf.endswith(tool_call_start[:i]):
                partial_len = i
                break

        if partial_len:
            safe = text_buf[:-partial_len]
            text_buf = text_buf[-partial_len:]
            if safe:
                yield _make_text(safe, response)
        else:
            # nothing pending – emit everything
            yield _make_text(text_buf, response)
            text_buf = ""

    # ── flush whatever is left ──
    if in_tool_call and tool_call_buf and last_resp is not None:
        # incomplete tool call (output truncated) – emit as plain text
        logger.warning("tool call incomplete (output truncated), emitting as text")
        yield _make_text(tool_call_start + tool_call_buf, last_resp)
    elif text_buf and last_resp is not None:
        yield _make_text(text_buf, last_resp)


def patch_kimi_tokenizer(tokenizer: TokenizerWrapper):
    """
    Version of to-be-upstreamed kimi-k2 tool parser
    """
    import ast
    import json
    from typing import Any

    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0 <|tool_call_argument_begin|> {"a": 2, "b": 3}
    _func_name_regex = re.compile(
        r"^\s*(.+):\d+\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)

    # kimi has a tool_calls_section - we're leaving this up to the caller to handle
    tool_call_start = "<|tool_call_begin|>"
    tool_call_end = "<|tool_call_end|>"

    def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
        try:
            return json.loads(value)  # pyright: ignore[reportAny]
        except Exception:
            pass

        try:
            return ast.literal_eval(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        return value

    def parse_tool_call(text: str, tools: Any | None = None):
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError(f"Could not parse function name from tool call: {text!r}")
        func_name = func_name_match.group(1)
        # strip off the `functions.` prefix, if it exists.
        func_name = func_name[func_name.find(".") + 1 :]

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError(f"Could not parse function args from tool call: {text!r}")
        func_args = func_args_match.group(1)
        # the args should be valid json - no need to check against our tools to deserialize
        arg_dct = _deserialize(func_args)  # pyright: ignore[reportAny]

        return dict(name=func_name, arguments=arg_dct)  # pyright: ignore[reportAny]

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = parse_tool_call


def patch_glm_tokenizer(tokenizer: TokenizerWrapper):
    """
    Fixed version of mlx_lm's glm47 tool parser that handles regex match failures.
    """
    import ast
    import json
    from typing import Any

    import regex as re

    _func_name_regex = re.compile(r"^(.*?)<arg_key>", re.DOTALL)
    _func_arg_regex = re.compile(
        r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
        re.DOTALL,
    )

    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: list[Any] | None,
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:  # pyright: ignore[reportAny]
            func = tool["function"]  # pyright: ignore[reportAny]
            if func["name"] == tool_name:
                params = func["parameters"]  # pyright: ignore[reportAny]
                if params is None:
                    return False
                props = params.get("properties", {})  # pyright: ignore[reportAny]
                arg_props = props.get(arg_name, {})  # pyright: ignore[reportAny]
                arg_type = arg_props.get("type", None)  # pyright: ignore[reportAny]
                return arg_type == "string"  # pyright: ignore[reportAny]
        return False

    def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
        try:
            return json.loads(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        try:
            return ast.literal_eval(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        return value

    def parse_tool_call(text: str, tools: list[Any] | None = None):
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError(f"Could not parse function name from tool call: {text!r}")
        func_name = func_name_match.group(1)

        pairs = _func_arg_regex.findall(text)
        arg_dct: dict[str, Any] = {}
        for key, value in pairs:  # pyright: ignore[reportAny]
            arg_key = key.strip()  # pyright: ignore[reportAny]
            arg_val = value.strip()  # pyright: ignore[reportAny]
            if not _is_string_type(func_name, arg_key, tools):  # pyright: ignore[reportAny]
                arg_val = _deserialize(arg_val)  # pyright: ignore[reportAny]
            arg_dct[arg_key] = arg_val
        return dict(name=func_name, arguments=arg_dct)

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = parse_tool_call


def _parse_parameters_state_machine(content: str) -> dict[str, str]:
    """
    Parse XML parameters using character-by-character state machine.

    Handles both correct and broken formats:
    - Correct: <parameter=name>value</parameter>
    - Broken: <parameter=name=value</parameter> (AI puts value in attribute)

    Works with any content including source code, special characters, and nested XML.

    Args:
        content: XML content containing parameter tags

    Returns:
        Dictionary mapping parameter names to values
    """
    arguments: dict[str, str] = {}
    i = 0

    while i < len(content):
        # Look for <parameter= tag
        if content[i:i+11] == '<parameter=':
            i += 11  # Move past <parameter=

            # Extract parameter name (until we hit >, =, or <)
            param_name_chars = []
            while i < len(content) and content[i] not in ['>', '=', '<']:
                param_name_chars.append(content[i])
                i += 1

            param_name = ''.join(param_name_chars).strip()

            if i >= len(content):
                break

            # Determine format by checking next character
            if content[i] == '>':
                # Correct format: <parameter=name>value</parameter>
                i += 1  # Skip >

                # Extract value until </parameter>
                value_chars = []
                depth = 0  # Track nesting for XML-like content in values
                while i < len(content):
                    if content[i:i+12] == '</parameter>':
                        if depth == 0:
                            break
                        else:
                            # This is part of the value content
                            value_chars.append(content[i])
                            i += 1
                    elif content[i] == '<' and i+1 < len(content) and content[i+1] != '/':
                        # Potential nested tag in value
                        depth += 1
                        value_chars.append(content[i])
                        i += 1
                    elif content[i:i+2] == '</':
                        # Potential closing tag
                        if depth > 0:
                            depth -= 1
                        value_chars.append(content[i])
                        i += 1
                    else:
                        value_chars.append(content[i])
                        i += 1

                arguments[param_name] = ''.join(value_chars).strip()
                i += 12  # Skip </parameter>

            elif content[i] == '=':
                # Broken format: <parameter=name=value</parameter>
                i += 1  # Skip =

                # Extract value until </parameter>
                value_chars = []
                while i < len(content):
                    if content[i:i+12] == '</parameter>':
                        break
                    value_chars.append(content[i])
                    i += 1

                arguments[param_name] = ''.join(value_chars).strip()
                i += 12  # Skip </parameter>

            elif content[i:i+12] == '</parameter>':
                # Malformed: <parameter=name</parameter> with no value
                arguments[param_name] = ""
                i += 12  # Skip </parameter>
        else:
            i += 1

    return arguments


def _parse_qwen_xml_tool_call(content: str, call_index: int) -> dict[str, Any] | None:
    """
    Parse Qwen3-Coder style pure XML tool calls.

    Format:
    <function=calculator>
    <parameter=operation>multiply</parameter>
    <parameter=a>5</parameter>
    </function>

    Returns OpenAI-compatible tool call dict or None if parsing fails.

    Uses state machine parser to handle both correct and broken XML formats,
    including arbitrary content (source code, HTML/JSX tags, special chars).
    """
    import re

    # Extract function name from <function=name> tag
    function_match = re.search(r"<function=([^>]+)>", content)
    if not function_match:
        return None

    function_name = function_match.group(1).strip()

    # Parse parameters using state machine parser
    # This handles both correct and broken XML formats robustly
    arguments = _parse_parameters_state_machine(content)

    # Format as OpenAI-compatible tool call
    return {
        "id": f"call_{call_index}",
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(arguments),
        },
    }


def patch_qwen_tokenizer(tokenizer: TokenizerWrapper):
    """
    Patch Qwen tokenizer to use the qwen3_coder XML tool parser.
    Qwen models generate tool calls in XML format like:
    <tool_call>
    <function=name>
    <parameter=arg1>value1</parameter>
    </function>
    </tool_call>
    """
    from mlx_lm.tool_parsers.qwen3_coder import (
        tool_call_end,
        tool_call_start,
    )

    def custom_qwen_parser(raw_text: str) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Custom parser for Qwen XML tool calls that uses the working implementation.
        This avoids the bug in mlx_lm.tool_parsers.qwen3_coder.parse_tool_call.
        """
        # The raw_text is the content between <tool_call> and </tool_call>
        # Try parsing as pure XML (Qwen3-Coder format)
        result = _parse_qwen_xml_tool_call(raw_text, call_index=0)
        if result is None:
            raise ValueError(f"Failed to parse Qwen tool call: {raw_text}")
        return result

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = custom_qwen_parser


def _validate_single_tool(obj: dict[str, Any]) -> ToolCallItem:
    # Handle OpenAI format with nested 'function' object
    if "function" in obj:
        func = obj["function"]
        name = func.get("name")
        args = func.get("arguments")
    else:
        # Handle flat format
        name = obj.get("name")
        args = obj.get("arguments")

    if name is not None and args is not None and isinstance(name, str):
        # If args is already a string (as in OpenAI format), use it directly
        if isinstance(args, str):
            return ToolCallItem(name=name, arguments=args)
        else:
            return ToolCallItem(name=name, arguments=json.dumps(args))
    else:
        raise ValueError(f"Invalid tool call structure: {obj}")


def parse_qwen_thinking(
    responses: Generator[GenerationResponse],
) -> Generator[GenerationResponse]:
    """
    Parser for Qwen thinking models.
    Currently a pass-through - needs implementation for thinking token detection.
    """
    # TODO: Implement thinking token detection for Qwen models
    # Qwen thinking models use special tokens <think> and </think>
    # Need to detect these tokens and emit appropriate thinking markers
    for response in responses:
        yield response


def parse_qwen_tool_calls(
    responses: Generator[GenerationResponse],
) -> Generator[GenerationResponse]:
    """
    Parser for Qwen tool calls in XML format.

    Supports two formats:
    1. JSON-in-XML (Qwen2.5/Qwen3-Instruct):
       <tool_call>
       {"name": "function_name", "arguments": {"arg1": "value1"}}
       </tool_call>

    2. Pure XML (Qwen3-Coder):
       <tool_call>
       <function=calculator>
       <parameter=operation>multiply</parameter>
       <parameter=a>5</parameter>
       </function>
       </tool_call>

    This parser detects these XML tags, extracts the tool call information,
    and converts to OpenAI-compatible format.
    """
    import regex as re

    accumulated_text = ""
    tool_calls: list[dict[str, Any]] = []
    tool_call_buffer = ""
    in_tool_call = False
    tool_call_index = 0

    # Pattern to match tool call XML tags
    tool_call_start_pattern = re.compile(r"<tool_call>")
    tool_call_end_pattern = re.compile(r"</tool_call>")

    for response in responses:
        accumulated_text += response.text
        current_text = response.text
        visible_text = current_text

        # Check if we're entering a tool call
        if tool_call_start_pattern.search(current_text):
            in_tool_call = True
            # Remove the opening tag from the visible text
            visible_text = tool_call_start_pattern.sub("", visible_text)

        if in_tool_call:
            tool_call_buffer += current_text

            # Check if we're exiting a tool call
            if tool_call_end_pattern.search(tool_call_buffer):
                # Remove the opening and closing tags
                content = tool_call_start_pattern.sub("", tool_call_buffer)
                content = tool_call_end_pattern.sub("", content).strip()

                tool_call = None

                # Try parsing as JSON first (Qwen2.5/Qwen3-Instruct format)
                try:
                    tool_call_data = json.loads(content)

                    # Format as OpenAI-compatible tool call
                    tool_call = {
                        "id": f"call_{tool_call_index}",
                        "type": "function",
                        "function": {
                            "name": str(tool_call_data.get("name", "")),  # pyright: ignore[reportAny]
                            "arguments": json.dumps(tool_call_data.get("arguments", {})),  # pyright: ignore[reportAny]
                        },
                    }

                except json.JSONDecodeError:
                    # Try parsing as pure XML (Qwen3-Coder format)
                    tool_call = _parse_qwen_xml_tool_call(content, tool_call_index)

                if tool_call is not None:
                    tool_calls.append(tool_call)
                    tool_call_index += 1

                # Reset for next tool call
                tool_call_buffer = ""
                in_tool_call = False

                # Don't yield the tool call XML content as visible text
                continue

        # If we're not in a tool call, yield the response normally
        if not in_tool_call:
            # Check if this is the final response
            if response.finish_reason is not None:
                # If we have tool calls, set finish_reason to "tool_calls"
                if tool_calls:
                    yield GenerationResponse(
                        text="",
                        token=response.token,
                        finish_reason="tool_calls",
                        stats=response.stats,
                        tool_calls=tool_calls,
                    )
                else:
                    yield response
            else:
                yield response

EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(
    prompt: str | ChatCompletionMessageText | list[ChatCompletionMessageText],
):
    if isinstance(prompt, list):
        if len(prompt) == 0:
            logger.debug("Empty message prompt received in debug prompt")
            return
        prompt = prompt[0]

    if isinstance(prompt, ChatCompletionMessageText):
        prompt = prompt.text

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        if engine_force_oom:
            engine_force_oom()
        else:
            logger.warning("OOM test not supported for current engine")
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)
