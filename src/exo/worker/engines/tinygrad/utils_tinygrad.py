import asyncio
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from tinygrad import Tensor, nn, Context
from tinygrad.nn.state import load_state_dict

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.tinygrad.models.llama import Transformer, TransformerShard, convert_from_huggingface, fix_bf16
from exo.worker.engines.tinygrad.tinygrad_helpers import concat_weights, load
from exo.worker.engines.tinygrad.constants import MODEL_PARAMS
from exo.worker.runner.bootstrap import logger

# Singleton executor so tinygrad always runs on the same thread
_executor = ThreadPoolExecutor(max_workers=1)

# Disable gradient computation by default
Tensor.no_grad = True


def build_transformer(model_path: Path, shard: ShardMetadata, model_size="8B", device=None):
    """Build a transformer model for the given shard"""
    # Build model
    linear = nn.Linear
    model = Transformer(
        **MODEL_PARAMS[model_size]["args"],
        linear=linear,
        max_context=8192,
        jit=True,
        shard=shard
    )

    # Load weights
    if model_path.is_dir():
        if (model_path / "model.safetensors.index.json").exists():
            weights = load(str(model_path / "model.safetensors.index.json"), shard)
        elif (model_path / "model.safetensors").exists():
            weights = load(str(model_path / "model.safetensors"), shard)
        else:
            weights = concat_weights(
                [load(str(model_path / f"consolidated.{i:02d}.pth"), shard)
                 for i in range(MODEL_PARAMS[model_size]["files"])],
                device[0] if isinstance(device, tuple) else device
            )
    else:
        weights = load(str(model_path), shard)

    weights = convert_from_huggingface(
        weights, model,
        MODEL_PARAMS[model_size]["args"]["n_heads"],
        MODEL_PARAMS[model_size]["args"]["n_kv_heads"]
    )
    weights = fix_bf16(weights)

    with Context(BEAM=0):
        # Replace weights in model
        load_state_dict(model, weights, strict=False, consume=False)
        model = TransformerShard(shard, model)

    return model


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata):
    """Get tokenizer for the model"""
    # Import here to avoid circular dependencies
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


async def initialize_tinygrad(bound_instance: BoundInstance) -> tuple[Any, Any, Any]:
    """
    Initialize the tinygrad model, tokenizer, and sampler.
    Returns: (model, tokenizer, sampler)
    """
    logger.info("Initializing tinygrad engine")

    shard_metadata = bound_instance.bound_shard
    model_path = build_model_path(shard_metadata.model_meta.model_id)

    # Determine model size from model_id
    model_size = "1B" if "1b" in shard_metadata.model_meta.model_id.lower() else \
                 "3B" if "3b" in shard_metadata.model_meta.model_id.lower() else \
                 "8B" if "8b" in shard_metadata.model_meta.model_id.lower() else "70B"

    # Build model in executor (tinygrad thread)
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(
        _executor,
        build_transformer,
        model_path,
        shard_metadata,
        model_size
    )

    # Load tokenizer
    tokenizer_path = model_path if model_path.is_dir() else model_path.parent
    tokenizer = get_tokenizer(tokenizer_path, shard_metadata)

    # Simple sampler (will be replaced with actual sampling logic)
    sampler = lambda logits: logits  # Placeholder

    logger.info(f"Tinygrad model initialized: {model_size}")

    return model, tokenizer, sampler


def get_tinygrad_executor():
    """Get the singleton executor for tinygrad operations"""
    return _executor
