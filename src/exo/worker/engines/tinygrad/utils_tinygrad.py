import asyncio
import os
import json
from pathlib import Path
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor

from tinygrad import Tensor, nn, Context, Device
from tinygrad.nn.state import load_state_dict

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.tinygrad.models.llama import (
    Transformer,
    TransformerShard,
    convert_from_huggingface,
    fix_bf16,
)
from exo.worker.engines.tinygrad.tinygrad_helpers import concat_weights, load
from exo.worker.engines.tinygrad.constants import MODEL_PARAMS
from exo.worker.runner.bootstrap import logger

# Singleton executor so tinygrad always runs on the same thread
_executor = ThreadPoolExecutor(max_workers=1)

# Disable gradient computation by default
Tensor.no_grad = True


def detect_best_device() -> str:
    """
    Detect the best available device for tinygrad inference.

    Priority order:
    1. QCOM (Qualcomm - for Snapdragon X Elite and other Qualcomm chips)
    2. WEBGPU (Uses Vulkan backend on most systems, good for wide hardware support)
    3. CL (OpenCL - can use Vulkan drivers on some systems)
    4. METAL (Apple Silicon)
    5. CUDA (NVIDIA)
    6. HIP (AMD on Linux)
    7. AMD (AMD legacy)
    8. CPU (fallback)

    Returns:
        str: The best available device name (e.g., 'QCOM', 'WEBGPU', 'CPU')
    """
    # Allow manual override via environment variable
    manual_device = os.getenv('TINYGRAD_DEVICE') or os.getenv('DEVICE')
    if manual_device:
        logger.info(f"Using manually specified device: {manual_device}")
        return manual_device

    # Get available devices
    available_devices = list(Device.get_available_devices())
    logger.info(f"Tinygrad available devices: {available_devices}")

    # Due to WebGPU compatibility issues with wgpu-native on Snapdragon X Elite,
    # default to CPU until GPU support matures
    logger.warning("GPU acceleration disabled due to WebGPU/wgpu-native compatibility issues")
    logger.warning("Using CPU mode - set DEVICE=WEBGPU to attempt GPU acceleration")
    return 'CPU'


def setup_device() -> str:
    """
    Setup and configure the best device for tinygrad.
    This should be called before any tinygrad operations.

    Returns:
        str: The configured device name
    """
    device = detect_best_device()

    # Set the DEVICE environment variable for tinygrad
    os.environ['DEVICE'] = device

    # Only set Device.DEFAULT for CPU to avoid GPU initialization issues
    # GPU devices can be manually enabled with DEVICE=WEBGPU
    if device == 'CPU':
        Device.DEFAULT = device

    # Log device information
    logger.info(f"Tinygrad configured to use device: {device}")
    if device == 'CPU':
        logger.info(f"Device.DEFAULT set to: {device}")

    # Try to get more information about the device if possible
    try:
        if device == 'QCOM':
            logger.info(
                "Using Qualcomm backend (Vulkan-based) - "
                "optimized for Snapdragon processors"
            )
        elif device == 'WEBGPU':
            logger.info(
                "Using WebGPU backend (Vulkan-based) - "
                "wide hardware compatibility"
            )
        elif device == 'CL':
            logger.info("Using OpenCL backend - may use Vulkan drivers")
        elif device == 'CPU':
            logger.info(
                "Using CPU backend - "
                "consider installing GPU drivers for better performance"
            )
    except Exception as e:
        logger.debug(f"Could not get additional device info: {e}")

    return device


def detect_model_config(model_path: Path) -> Dict[str, Any]:
    """
    Detect model configuration from config.json.
    Returns model args compatible with MODEL_PARAMS format.
    """
    config_path = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"

    if not config_path.exists():
        logger.warning(f"No config.json found at {config_path}, using default 8B config")
        return MODEL_PARAMS["8B"]["args"]

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Extract configuration
        dim = config.get('hidden_size', 4096)
        n_heads = config.get('num_attention_heads', 32)
        n_kv_heads = config.get('num_key_value_heads', config.get('num_attention_heads', 32))
        n_layers = config.get('num_hidden_layers', 32)
        vocab_size = config.get('vocab_size', 128256)
        hidden_dim = config.get('intermediate_size', 14336)
        norm_eps = config.get('rms_norm_eps', 1e-5)
        rope_theta = config.get('rope_theta', 500000)

        # Check for rope_scaling
        rope_scaling = config.get('rope_scaling', None)

        # Check for tied embeddings
        tie_word_embeddings = config.get('tie_word_embeddings', False)

        # Check for explicit head_dim (some models like Qwen3 specify this)
        head_dim = config.get('head_dim', None)

        logger.info(f"Detected model config: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
                   f"n_layers={n_layers}, vocab_size={vocab_size}, hidden_dim={hidden_dim}, head_dim={head_dim}")

        model_args = {
            "dim": dim,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "n_layers": n_layers,
            "norm_eps": norm_eps,
            "rope_theta": rope_theta,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
        }

        if rope_scaling:
            model_args["rope_scaling"] = rope_scaling

        if tie_word_embeddings:
            model_args["tie_word_embeddings"] = tie_word_embeddings

        if head_dim:
            model_args["head_dim"] = head_dim

        return model_args

    except Exception as e:
        logger.error(f"Failed to read config.json: {e}")
        logger.warning("Falling back to default 8B config")
        return MODEL_PARAMS["8B"]["args"]


def build_transformer(
    model_path: Path, shard: ShardMetadata, model_size="8B", device=None
):
    """Build a transformer model for the given shard"""
    # Detect model configuration from config.json instead of relying on model_size string
    try:
        model_args = detect_model_config(model_path)
        logger.info(f"Using detected model configuration: {model_args.get('dim')} dim, "
                   f"{model_args.get('n_heads')} heads, {model_args.get('n_layers')} layers")
    except Exception as e:
        logger.warning(f"Failed to detect model config, falling back to MODEL_PARAMS[{model_size}]: {e}")
        model_args = MODEL_PARAMS[model_size]["args"]

    # Get quantization config if present
    config_path = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
    group_size = 64  # Default group size
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            quant_config = config.get('quantization', {}) or config.get('quantization_config', {})
            group_size = quant_config.get('group_size', 64)
            if quant_config:
                logger.info(f"Detected quantization config: bits={quant_config.get('bits', 'N/A')}, group_size={group_size}")
        except Exception as e:
            logger.debug(f"Could not read quantization config: {e}")

    # Build model
    linear = nn.Linear
    model = Transformer(
        **model_args,
        linear=linear,
        max_context=8192,
        jit=True,
        shard=shard
    )

    # Load weights (with automatic dequantization if needed)
    if model_path.is_dir():
        if (model_path / "model.safetensors.index.json").exists():
            weights = load(str(model_path / "model.safetensors.index.json"), shard, group_size)
        elif (model_path / "model.safetensors").exists():
            weights = load(str(model_path / "model.safetensors"), shard, group_size)
        else:
            # Determine number of files from model_size fallback
            num_files = MODEL_PARAMS.get(model_size, {}).get("files", 1)
            weights = concat_weights(
                [load(str(model_path / f"consolidated.{i:02d}.pth"), shard, group_size)
                 for i in range(num_files)],
                device[0] if isinstance(device, tuple) else device
            )
    else:
        weights = load(str(model_path), shard, group_size)

    # Use the detected model args for weight conversion
    weights = convert_from_huggingface(
        weights, model,
        model_args["n_heads"],
        model_args["n_kv_heads"]
    )
    weights = fix_bf16(weights)

    with Context(BEAM=0):
        # Replace weights in model
        # Set realize=False since weights are already realized during dequantization
        load_state_dict(model, weights, strict=False, consume=False, realize=False)
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

    # Setup the best available device (with Vulkan/GPU support)
    setup_device()

    shard_metadata = bound_instance.bound_shard
    model_path = build_model_path(shard_metadata.model_meta.model_id)

    # Determine model size from model_id (for fallback only)
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

    # Get actual model info from detected config
    n_params = model.layers[0].attention.wq.weight.shape[0] * model.layers[0].attention.wq.weight.shape[1] * len(model.layers)
    n_params_b = n_params / 1e9
    target_device = os.getenv('DEVICE', 'CPU')
    logger.info(f"Tinygrad model initialized: ~{n_params_b:.1f}B parameters, {len(model.layers)} layers")
    logger.info(f"Weights loaded on CPU, will be transferred to {target_device} during inference")

    return model, tokenizer, sampler


def get_tinygrad_executor():
    """Get the singleton executor for tinygrad operations"""
    return _executor
