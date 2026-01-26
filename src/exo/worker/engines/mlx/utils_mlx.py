import json
import os
import resource
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

# Monkey-patch for transformers 5.x compatibility
# Kimi's tokenization_kimi.py imports bytes_to_unicode from the old location
# which was moved in transformers 5.0.0rc2
try:
    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[attr-defined]
except ImportError:
    pass  # transformers < 5.0 or bytes_to_unicode not available

from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE,
    KV_CACHE_BITS,
    TRUST_REMOTE_CODE,
)

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer
import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.runner.bootstrap import logger

Group = mx.distributed.Group
# Needed for 8 bit model
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))


# TODO: Test this
#  ALSO https://github.com/exo-explore/exo/pull/233#discussion_r2549683673
def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_meta.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


class ModelLoadingTimeoutError(Exception):
    pass


TimeoutCallback = Callable[[], None]


def eval_with_timeout(
    mlx_item: Any,  # pyright: ignore[reportAny]
    timeout_seconds: float = 60.0,
    on_timeout: TimeoutCallback | None = None,
) -> None:
    """Evaluate MLX item with a hard timeout.

    If on_timeout callback is provided, it will be called before terminating
    the process. This allows the runner to send a failure event before exit.
    """
    completed = threading.Event()

    def watchdog() -> None:
        if not completed.wait(timeout=timeout_seconds):
            logger.error(
                f"mlx_item evaluation timed out after {timeout_seconds:.0f}s. "
                "This may indicate an issue with FAST_SYNCH and tensor parallel sharding. "
                "Terminating process."
            )
            if on_timeout is not None:
                on_timeout()
            os._exit(1)

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()

    try:
        mx.eval(mlx_item)  # pyright: ignore[reportAny]
    finally:
        completed.set()


def mx_barrier(group: Group | None = None):
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0),
            stream=mx.default_stream(mx.Device(mx.cpu)),
            group=group,
        )
    )


def broadcast_from_zero(value: int, group: Group | None = None):
    if group is None:
        return value

    if group.rank() == 0:
        a = mx.array([value], dtype=mx.int32)
    else:
        a = mx.array([0], dtype=mx.int32)

    m = mx.distributed.all_sum(a, stream=mx.Device(mx.DeviceType.cpu), group=group)
    mx.eval(m)
    return int(m.item())


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> Group:
    """
    Initialize MLX distributed.
    """
    rank = bound_instance.bound_shard.device_rank
    logger.info(f"Starting initialization for rank {rank}")

    coordination_file = None
    try:
        # TODO: singleton instances
        match bound_instance.instance:
            case MlxRingInstance(hosts_by_node=hosts_by_node, ephemeral_port=_):
                coordination_file = (
                    f"./hosts_{bound_instance.instance.instance_id}_{rank}.json"
                )
                hosts_for_node = hosts_by_node[bound_instance.bound_node_id]
                hosts_json = HostList.from_hosts(hosts_for_node).model_dump_json()

                with open(coordination_file, "w") as f:
                    _ = f.write(hosts_json)

                logger.info(
                    f"rank {rank} hostfile: {coordination_file} hosts: {hosts_json}"
                )

                os.environ["MLX_HOSTFILE"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_RING_VERBOSE"] = "1"
                group = mx.distributed.init(backend="ring", strict=True)

            case MlxJacclInstance(
                ibv_devices=ibv_devices, jaccl_coordinators=jaccl_coordinators
            ):
                # Use RDMA connectivity matrix
                coordination_file = (
                    f"./hosts_{bound_instance.instance.instance_id}_{rank}.json"
                )
                ibv_devices_json = json.dumps(ibv_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(ibv_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                logger.info(f"rank {rank} MLX_IBV_DEVICES: {ibv_devices_json}")
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_IBV_DEVICES"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_JACCL_COORDINATOR"] = jaccl_coordinator
                group = mx.distributed.init(backend="jaccl", strict=True)

        logger.info(f"Rank {rank} mlx distributed initialization complete")

        return group
    finally:
        with contextlib.suppress(FileNotFoundError):
            if coordination_file:
                os.remove(coordination_file)


def initialize_mlx(
    bound_instance: BoundInstance,
) -> Group:
    # should we unseed it?
    # TODO: pass in seed from params
    mx.random.seed(42)

    assert len(bound_instance.instance.shard_assignments.node_to_runner) > 1, (
        "Tried to initialize mlx for a single node instance"
    )
    return mlx_distributed_init(bound_instance)


def load_mlx_items(
    bound_instance: BoundInstance,
    group: Group | None,
    on_timeout: TimeoutCallback | None = None,
) -> tuple[Model, TokenizerWrapper]:
    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(bound_instance.bound_shard.model_meta.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, strict=True)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = shard_and_load(
            bound_instance.bound_shard, group=group, on_timeout=on_timeout
        )
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    return cast(Model, model), tokenizer


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: Group,
    on_timeout: TimeoutCallback | None = None,
) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(shard_metadata.model_meta.model_id)

    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.debug(model)
    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # TODO: See if we should quantize the model.
        # def is_attention_layer(path: str) -> bool:
        #     path = path.lower()

        #     return "self_attn" in path and "layernorm" not in path

        # def quant_predicate(path: str, module: nn.Module):
        #     if not isinstance(module, nn.Linear):
        #         return False

        #     return is_attention_layer(path)
        # model, config = quantize_model(
        #        model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = tensor_auto_parallel(model, group)
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = pipeline_auto_parallel(model, group, shard_metadata)

    # Estimate timeout based on model size
    base_timeout = float(os.environ.get("EXO_MODEL_LOAD_TIMEOUT", "60"))
    model_size_gb = get_weights_size(shard_metadata).in_bytes / (1024**3)
    timeout_seconds = base_timeout + model_size_gb / 5
    logger.info(
        f"Evaluating model parameters with timeout of {timeout_seconds:.0f}s "
        f"(model size: {model_size_gb:.1f}GB)"
    )
    eval_with_timeout(model.parameters(), timeout_seconds, on_timeout)

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata) -> TokenizerWrapper:
    """Load tokenizer for a model shard. Delegates to load_tokenizer_for_model_id."""
    return load_tokenizer_for_model_id(shard_metadata.model_meta.model_id, model_path)


def get_eos_token_ids_for_model(model_id: str) -> list[int] | None:
    """
    Get the EOS token IDs for a model based on its ID.

    Some models require explicit EOS token configuration that isn't in their
    tokenizer config. This function returns the known EOS token IDs for such models.

    Args:
        model_id: The HuggingFace model ID

    Returns:
        List of EOS token IDs, or None if the model uses standard tokenizer config
    """
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm" in model_id_lower:
        return [151336, 151329, 151338]
    elif "gpt-oss" in model_id_lower or "gpt_oss" in model_id_lower:
        # GPT-OSS uses <|return|> token (200002) as EOS
        # Also include <|endoftext|> (199999) as fallback
        return [200002, 199999]
    return None


def load_tokenizer_for_model_id(model_id: str, model_path: Path) -> TokenizerWrapper:
    """
    Load tokenizer for a model given its ID and local path.

    This is the core tokenizer loading logic, handling special cases for different
    model families (Kimi, GLM, GPT-OSS, etc.) and transformers 5.x compatibility.

    Args:
        model_id: The HuggingFace model ID (e.g., "moonshotai/Kimi-K2-Instruct")
        model_path: Local path where the model/tokenizer files are stored

    Returns:
        TokenizerWrapper instance configured for the model
    """
    model_id_lower = model_id.lower()
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # Kimi uses a custom TikTokenTokenizer that transformers 5.x can't load via AutoTokenizer
    if "kimi-k2" in model_id_lower:
        sys.path.insert(0, str(model_path))
        from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

        # Patch encode to use internal tiktoken model directly
        # transformers 5.x has a bug in the encode->pad path for slow tokenizers
        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            # Pass allowed_special="all" to handle special tokens like <|im_user|>
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        hf_tokenizer.encode = _patched_encode
        return TokenizerWrapper(hf_tokenizer, eos_token_ids=eos_token_ids)

    # GPT-OSS uses o200k_harmony tokenizer with special Harmony format
    if "gpt-oss" in model_id_lower or "gpt_oss" in model_id_lower:
        logger.info(f"Loading GPT-OSS tokenizer for {model_id}")
        tokenizer = load_tokenizer(
            model_path,
            tokenizer_config_extra={"trust_remote_code": TRUST_REMOTE_CODE},
            eos_token_ids=eos_token_ids,
        )

        # Verify chat template is loaded
        if hasattr(tokenizer, '_tokenizer') and hasattr(tokenizer._tokenizer, 'chat_template'):
            if tokenizer._tokenizer.chat_template is None:  # pyright: ignore[reportUnknownMemberType]
                logger.warning(f"GPT-OSS tokenizer loaded but chat_template is None! Model may not generate in Harmony format.")
            else:
                logger.info("GPT-OSS chat template loaded successfully")
                # Log a sample of the template for debugging
                template_preview = str(tokenizer._tokenizer.chat_template)[:200]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                logger.debug(f"Chat template preview: {template_preview}...")

        return tokenizer

    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={"trust_remote_code": TRUST_REMOTE_CODE},
        eos_token_ids=eos_token_ids,
    )

    return tokenizer


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
    # Now we can properly access the messages
    messages = chat_task_data.messages
    model_id = chat_task_data.model.lower()
    is_gpt_oss = "gpt-oss" in model_id or "gpt_oss" in model_id

    formatted_messages: list[dict[str, Any]] = []
    for message in messages:
        # Convert content to appropriate plain Python type
        if isinstance(message.content, ChatCompletionMessageText):
            message.content = message.content.text
        if isinstance(message.content, list):
            if len(message.content) == 0:
                logger.warning("Received prompt with no content, skipping")
                continue
            message.content = "\n".join(c.text for c in message.content).strip()

        # Skip messages with no content and no thinking
        if message.content is None and message.thinking is None:
            continue

        # Build message dict manually with only plain Python types
        # This ensures the Jinja template receives only simple types
        msg_dict: dict[str, Any] = {
            "role": message.role,
        }

        # Add content if present
        if message.content is not None:
            msg_dict["content"] = message.content

        # Add thinking if present (GPT-OSS specific)
        if message.thinking is not None:
            msg_dict["thinking"] = message.thinking

        # Add optional fields only if they're present
        if message.name is not None:
            msg_dict["name"] = message.name

        if message.tool_calls is not None:
            # Convert tool_calls to plain Python types (in case they're Pydantic models)
            import json
            tool_calls_list = []
            if isinstance(message.tool_calls, list):  # pyright: ignore[reportUnnecessaryIsInstance]
                for tc in message.tool_calls:
                    tc_dict = tc.model_dump(mode='python') if hasattr(tc, 'model_dump') else tc  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

                    # Qwen3-Coder template expects arguments as dict, not JSON string
                    # Convert OpenAI format (string) to Qwen format (dict)
                    if isinstance(tc_dict, dict) and "function" in tc_dict:
                        func = tc_dict["function"]
                        if isinstance(func, dict) and "arguments" in func:
                            args = func["arguments"]
                            # If arguments is a JSON string, parse it to dict
                            if isinstance(args, str):
                                try:
                                    func["arguments"] = json.loads(args)
                                except (json.JSONDecodeError, TypeError):
                                    logger.warning(f"Failed to parse tool_call arguments as JSON: {args}")
                                    func["arguments"] = {}

                    tool_calls_list.append(tc_dict)
                msg_dict["tool_calls"] = tool_calls_list
            else:
                msg_dict["tool_calls"] = message.tool_calls

        if message.tool_call_id is not None:
            msg_dict["tool_call_id"] = message.tool_call_id

        if message.function_call is not None:
            # Convert function_call to plain Python types (in case it's a Pydantic model)
            import json
            if hasattr(message.function_call, 'model_dump'):
                func_call_dict = message.function_call.model_dump(mode='python')  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            else:
                func_call_dict = message.function_call

            # Qwen3-Coder template expects arguments as dict, not JSON string
            if isinstance(func_call_dict, dict) and "arguments" in func_call_dict:
                args = func_call_dict["arguments"]
                if isinstance(args, str):
                    try:
                        func_call_dict["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse function_call arguments as JSON: {args}")
                        func_call_dict["arguments"] = {}

            msg_dict["function_call"] = func_call_dict

        formatted_messages.append(msg_dict)

    # Debug: Log the formatted messages to see what we're passing to the template
    logger.debug(f"Formatted messages for template: {formatted_messages}")

    # Ensure all items are plain dicts with plain Python types
    import json
    try:
        # This will fail if there are non-serializable objects
        json.dumps(formatted_messages)
    except (TypeError, ValueError) as e:
        logger.error(f"Messages contain non-serializable objects: {e}")
        logger.error(f"Messages: {formatted_messages}")
        raise

    # Prepare template kwargs - GPT-OSS specific parameters
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    # Add tools if present - convert to plain Python types
    if chat_task_data.tools is not None:
        # Convert tools to plain Python dicts to avoid Pydantic model issues in Jinja2
        # Use JSON round-trip to ensure all nested objects are plain Python types
        import json
        try:
            tools_json = json.dumps(
                chat_task_data.tools,
                default=lambda o: o.model_dump(mode='python') if hasattr(o, 'model_dump') else str(o)  # pyright: ignore[reportAny]
            )
            tools_data = json.loads(tools_json)  # pyright: ignore[reportAny]

            # Clean tools for Qwen3-Coder template compatibility
            for tool in tools_data:  # pyright: ignore[reportAny]
                if isinstance(tool, dict) and "function" in tool:
                    func = tool["function"]  # pyright: ignore[reportUnknownVariableType]
                    if "parameters" in func and isinstance(func["parameters"], dict):  # pyright: ignore[reportUnknownArgumentType]
                        params = func["parameters"]  # pyright: ignore[reportUnknownVariableType]

                        # Ensure properties is a dict
                        if "properties" in params:  # pyright: ignore[reportUnknownArgumentType]
                            if isinstance(params["properties"], list):  # pyright: ignore[reportUnknownArgumentType]
                                props_dict = {}
                                for prop in params["properties"]:  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
                                    if isinstance(prop, dict) and len(prop) == 1:
                                        props_dict.update(prop)  # pyright: ignore[reportUnknownArgumentType]
                                params["properties"] = props_dict  # pyright: ignore[reportUnknownArgumentType]
                            elif not isinstance(params["properties"], dict):  # pyright: ignore[reportUnknownArgumentType]
                                logger.warning(f"Tool properties is not a dict: {type(params['properties'])}")  # pyright: ignore[reportUnknownArgumentType]
                                params["properties"] = {}  # pyright: ignore[reportUnknownArgumentType]

                            # Clean each property - remove additionalProperties that confuse Qwen template
                            for prop_name, prop_spec in params["properties"].items():  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType,reportUnknownArgumentType]
                                # Remove additionalProperties from individual properties
                                # Qwen template iterates over properties and gets confused by this field
                                if isinstance(prop_spec, dict) and "additionalProperties" in prop_spec:
                                    logger.debug(f"Removing additionalProperties from {func['name']}.{prop_name}")  # pyright: ignore[reportUnknownArgumentType]
                                    del prop_spec["additionalProperties"]

                        # Remove top-level additionalProperties from parameters if present
                        if "additionalProperties" in params:  # pyright: ignore[reportUnknownArgumentType]
                            logger.debug(f"Removing top-level additionalProperties from {func['name']}")  # pyright: ignore[reportUnknownArgumentType]
                            del params["additionalProperties"]  # pyright: ignore[reportUnknownArgumentType]

            template_kwargs["tools"] = tools_data
            logger.debug("Cleaned tools for Qwen template")
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to convert tools to plain Python types: {e}, passing as-is")
            template_kwargs["tools"] = chat_task_data.tools

    # GPT-OSS specific parameters from tokenizer_config.json
    if is_gpt_oss:
        logger.info("Applying GPT-OSS specific chat template parameters")
        # reasoning_effort: "low", "medium", "high" - affects thinking depth
        # We can infer this from temperature or use default "medium"
        if hasattr(chat_task_data, 'temperature') and chat_task_data.temperature is not None:
            if chat_task_data.temperature < 0.3:
                template_kwargs["reasoning_effort"] = "low"
            elif chat_task_data.temperature > 0.7:
                template_kwargs["reasoning_effort"] = "high"
            else:
                template_kwargs["reasoning_effort"] = "medium"

        # builtin_tools: can include "browser" and "python"
        # Only enable if tools are present
        if chat_task_data.tools:
            template_kwargs["builtin_tools"] = []  # Start empty, model will use custom tools

    try:
        prompt: str = tokenizer.apply_chat_template(  # pyright: ignore[reportAny]
            formatted_messages,
            **template_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to apply chat template: {e}")
        logger.error(f"Model: {chat_task_data.model}")

        # Log messages in detail to debug format issues
        import json
        try:
            logger.error(f"Messages (formatted): {json.dumps(formatted_messages, indent=2)}")
        except Exception:
            logger.error(f"Messages (cannot serialize): {formatted_messages}")

        # Log tools in detail to debug format issues
        if "tools" in template_kwargs:
            logger.error(f"Tools (formatted): {json.dumps(template_kwargs['tools'], indent=2)}")
        logger.error(f"Template kwargs keys: {template_kwargs.keys()}")

        # Check for non-dict objects in messages
        for i, msg in enumerate(formatted_messages):
            if not isinstance(msg, dict):
                logger.error(f"Message {i} is not a dict: {type(msg)} = {msg}")
            elif "content" in msg and msg["content"] is not None:
                if isinstance(msg["content"], list):
                    for j, content_item in enumerate(msg["content"]):
                        if not isinstance(content_item, dict):
                            logger.error(f"Message {i}, content[{j}] is not a dict: {type(content_item)}")
                elif not isinstance(msg["content"], str):
                    logger.error(f"Message {i}, content is neither string nor list: {type(msg['content'])}")

        # Try progressively simpler parameters
        if chat_task_data.tools is not None:
            logger.warning("Retrying without tools parameter...")
            template_kwargs.pop("tools", None)
            template_kwargs.pop("builtin_tools", None)
            try:
                prompt = tokenizer.apply_chat_template(  # pyright: ignore[reportAny]
                    formatted_messages,
                    **template_kwargs
                )
            except Exception as e2:
                logger.error(f"Retry without tools failed: {e2}")
                # Last resort: minimal parameters
                logger.warning("Retrying with minimal parameters...")
                try:
                    prompt = tokenizer.apply_chat_template(
                        formatted_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as e3:
                    logger.error(f"All retries failed: {e3}")
                    raise Exception(
                        f"Chat template failed: {str(e)}. "
                        f"Message format may be incompatible with this model. "
                        f"Original error: {str(e)}"
                    ) from e
        else:
            raise Exception(
                f"Chat template failed: {str(e)}. "
                f"Message format may be incompatible with this model."
            ) from e

    # Log the prompt for debugging (especially important for GPT-OSS)
    if is_gpt_oss:
        logger.info("="*80)
        logger.info("GPT-OSS Harmony Format Prompt:")
        logger.info("="*80)
        # Check if prompt contains Harmony format tokens
        if "<|start|>" in prompt:
            logger.info("✓ Prompt contains <|start|> token (Harmony format detected)")
        else:
            logger.warning("✗ Prompt does NOT contain <|start|> token - model may not generate in Harmony format!")

        if "<|channel|>" in prompt:
            logger.info("✓ Prompt contains <|channel|> token")
        else:
            logger.warning("✗ Prompt does NOT contain <|channel|> token")

        # Log the actual prompt
        logger.info(prompt)
        logger.info("="*80)
    else:
        logger.info(f"Chat template applied successfully for {chat_task_data.model}")
        logger.debug(f"Prompt: {prompt}")

    return prompt


class NullKVCache(KVCache):
    """
    A KVCache that pretends to exist but holds zero tokens.
    It satisfies .state/.meta_state and never allocates real keys/values.
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # zero-length K/V so shapes/dtypes are defined but empty
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> list[KVCache | RotatingKVCache | QuantizedKVCache]:
    assert hasattr(model, "layers")

    # TODO: Do this for all models
    if hasattr(model, "make_cache") and isinstance(model, GptOssModel):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]


def mlx_force_oom(size: int = 40000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        return

    model_bytes = model_size.in_bytes
    max_rec_size = int(mx.metal.device_info()["max_recommended_working_set_size"])
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        logger.warning(
            f"Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    # Allocate 10% of model size for KV cache to support larger context windows
    kv_bytes = int(0.10 * model_bytes)
    target_cache = int(1.20 * (model_bytes + kv_bytes))
    target_cache = min(target_cache, max_rec_size)
    mx.set_cache_limit(target_cache)
    mx.set_wired_limit(max_rec_size)
    logger.info(f"Wired limit set to {max_rec_size}.")


def mlx_cleanup(
    model: Model | None, tokenizer: TokenizerWrapper | None, group: Group | None
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()
