import json
import re
import time
from collections.abc import Generator
from functools import cache
from typing import Any

import mlx.core as mx
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.chunks import TokenChunk
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
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
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
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
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

    model = None
    tokenizer = None
    group = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )
    with task_receiver as tasks:
        for task in tasks:
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

                    model, tokenizer = load_mlx_items(
                        bound_instance, group, on_timeout=on_model_load_timeout
                    )

                    current_status = RunnerLoaded()
                    logger.info("runner loaded")
                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    assert model
                    assert tokenizer
                    current_status = RunnerWarmingUp()
                    logger.info("runner warming up")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    logger.info(f"warming up inference for instance: {instance}")
                    toks = warmup_inference(
                        model=model,
                        tokenizer=tokenizer,
                        # kv_prefix_cache=kv_prefix_cache,  # supply for warmup-time prefix caching
                    )
                    logger.info(f"warmed up by generating {toks} tokens")
                    logger.info(
                        f"runner initialized in {time.time() - setup_start_time} seconds"
                    )
                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ChatCompletion(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    logger.info(f"received chat request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    assert model
                    assert tokenizer
                    assert task_params.messages[0].content is not None

                    try:
                        _check_for_debug_prompts(task_params.messages[0].content)

                        # Generate responses using the actual MLX generation
                        mlx_generator = mlx_generate(
                            model=model,
                            tokenizer=tokenizer,
                            task=task_params,
                        )

                        # GPT-OSS specific parsing to match other model formats.
                        # GPT-OSS specific parsing to match other model formats.
                        if isinstance(model, GptOssModel):
                            mlx_generator = parse_gpt_oss(mlx_generator)
                        # Qwen thinking models parsing
                        elif isinstance(model, (Qwen3NextModel, Qwen3MoeModel)):
                            # Check if it's a thinking model (model name contains "thinking")
                            # For now, apply parse_qwen_thinking
                            mlx_generator = parse_qwen_thinking(mlx_generator)

                        # Apply tool call parser for Qwen models
                        if isinstance(model, (Qwen3NextModel, Qwen3MoeModel)):
                            mlx_generator = parse_qwen_tool_calls(mlx_generator)

                        for response in mlx_generator:
                            match response:
                                case GenerationResponse():
                                    if device_rank == 0:
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=TokenChunk(
                                                    idx=response.token,
                                                    model=shard_metadata.model_meta.model_id,
                                                    text=response.text,
                                                    token_id=response.token,
                                                    finish_reason=response.finish_reason,
                                                    stats=response.stats,
                                                    tool_calls=response.tool_calls,
                                                ),
                                            )
                                        )

                    # can we make this more explicit?
                    except Exception as e:
                        if device_rank == 0:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=TokenChunk(
                                        idx=0,
                                        model=shard_metadata.model_meta.model_id,
                                        text="",
                                        token_id=0,
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


def parse_gpt_oss(
    responses: Generator[GenerationResponse],
) -> Generator[GenerationResponse]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    parser_active = False
    use_passthrough = False
    token_count = 0
    first_error_logged = False

    for response in responses:
        token_count += 1

        # If we've switched to passthrough mode, just yield tokens directly
        if use_passthrough:
            yield response
            if response.finish_reason is not None:
                break
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
            break


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


def _parse_qwen_xml_tool_call(content: str, call_index: int) -> dict[str, Any] | None:
    """
    Parse Qwen3-Coder style pure XML tool calls.

    Format:
    <function=calculator>
    <parameter=operation>multiply</parameter>
    <parameter=a>5</parameter>
    </function>

    Returns OpenAI-compatible tool call dict or None if parsing fails.
    """
    # Extract function name from <function=name> tag
    function_match = re.search(r"<function=([^>]+)>", content)
    if not function_match:
        return None

    function_name = function_match.group(1).strip()

    # Extract all parameters
    arguments: dict[str, str] = {}
    parameter_pattern = re.compile(r"<parameter=([^>]+)>\s*([^<]*?)\s*</parameter>", re.DOTALL)

    for match in parameter_pattern.finditer(content):
        param_name = match.group(1).strip()
        param_value = match.group(2).strip()
        arguments[param_name] = param_value

    # Format as OpenAI-compatible tool call
    return {
        "id": f"call_{call_index}",
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(arguments),
        },
    }
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
