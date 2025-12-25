from tinygrad.nn.state import safe_load, torch_load
from tinygrad import Tensor, dtypes
from pathlib import Path
import json
from typing import List, Dict
from exo.shared.types.worker.shards import ShardMetadata
from fnmatch import fnmatch
import re

DEBUG = 0  # TODO: Get this from environment or config


def dequantize_mlx_4bit(weight: Tensor, scales: Tensor, biases: Tensor, group_size: int = 64) -> Tensor:
  """
  Dequantize MLX 4-bit quantized weights.

  Args:
    weight: Packed 4-bit weights as uint32
            - Regular: shape (out_features, in_features // 8)
            - MoE: shape (num_experts, out_features, in_features // 8)
    scales: Dequantization scales
            - Regular: shape (out_features, n_groups)
            - MoE: shape (num_experts, out_features, n_groups)
    biases: Dequantization biases (same shape as scales)
    group_size: Size of each quantization group (default: 64)

  Returns:
    Dequantized weight tensor in fp16/bf16
  """
  from tinygrad import Device

  # Handle both regular and MoE (Mixture of Experts) weights
  weight_shape = weight.shape
  is_moe = len(weight_shape) == 3  # MoE has (num_experts, out_features, in_features // 8)

  if is_moe:
    num_experts, out_features, packed_in = weight_shape
    n_groups = scales.shape[-1]  # Last dimension is always n_groups
  else:
    out_features, packed_in = weight_shape
    n_groups = scales.shape[-1]

  # Each uint32 contains 8 4-bit values
  in_features = packed_in * 8

  # Move to CPU for dequantization if not already there
  # DISK tensors need to be realized to CPU first
  weight_cpu = weight.to('CPU').realize()
  scales_cpu = scales.to('CPU').realize()
  biases_cpu = biases.to('CPU').realize()

  # Unpack 4-bit values from uint32
  # Each uint32 packs 8 values: [v0 v1 v2 v3 v4 v5 v6 v7] where each v is 4 bits
  unpacked = []
  for i in range(8):
    # Extract i-th 4-bit value from each uint32
    # Shift right by (i * 4) bits and mask with 0xF
    shift = i * 4
    vals = (weight_cpu >> shift) & 0xF
    unpacked.append(vals)

  # Stack and reshape based on whether this is MoE or regular
  if is_moe:
    # (8, num_experts, out_features, packed_in) -> (num_experts, out_features, in_features)
    w = Tensor.stack(*unpacked, dim=-1).reshape(num_experts, out_features, in_features)
    # Reshape for group-wise dequantization
    # (num_experts, out_features, in_features) -> (num_experts, out_features, n_groups, group_size)
    w = w.reshape(num_experts, out_features, n_groups, group_size)
    # Expand scales and biases for broadcasting
    scales_expanded = scales_cpu.reshape(num_experts, out_features, n_groups, 1)
    biases_expanded = biases_cpu.reshape(num_experts, out_features, n_groups, 1)
  else:
    # (8, out_features, packed_in) -> (out_features, in_features)
    w = Tensor.stack(*unpacked, dim=-1).reshape(out_features, in_features)
    # Reshape for group-wise dequantization
    # (out_features, in_features) -> (out_features, n_groups, group_size)
    w = w.reshape(out_features, n_groups, group_size)
    # Expand scales and biases for broadcasting
    scales_expanded = scales_cpu.reshape(out_features, n_groups, 1)
    biases_expanded = biases_cpu.reshape(out_features, n_groups, 1)

  # Dequantize: y = x * scale + bias (MLX formula)
  w_float = w.cast(scales_cpu.dtype)
  dequantized = w_float * scales_expanded + biases_expanded

  # Reshape back to original dimensions
  if is_moe:
    result = dequantized.reshape(num_experts, out_features, in_features)
  else:
    result = dequantized.reshape(out_features, in_features)

  # Keep on CPU after dequantization
  # Tinygrad will automatically move to target device during forward pass
  # Trying to move to WEBGPU here causes ctypes issues with device handles
  return result


def is_quantized_weight(weight_map: Dict[str, Tensor], base_name: str) -> bool:
  """Check if a weight is quantized (has corresponding scales and biases)"""
  return f"{base_name}.scales" in weight_map and f"{base_name}.biases" in weight_map


def maybe_dequantize(weight_map: Dict[str, Tensor], key: str, group_size: int = 64) -> Tensor:
  """
  Dequantize a weight if it's quantized, otherwise return as-is.

  Args:
    weight_map: Dictionary of all weights
    key: Key of the weight to potentially dequantize
    group_size: Quantization group size (from config)

  Returns:
    Dequantized or original weight tensor (on CPU)
  """
  tensor = weight_map[key]

  if not key.endswith(".weight"):
    # Not a weight tensor (e.g., norm layer), move from DISK to CPU
    if tensor.device.startswith('DISK:'):
      tensor = tensor.to('CPU').realize()
    return tensor

  base_name = key[:-7]  # Remove ".weight"

  if is_quantized_weight(weight_map, base_name):
    # This is a quantized weight - dequantize it
    weight = weight_map[key]
    scales = weight_map[f"{base_name}.scales"]
    biases = weight_map[f"{base_name}.biases"]

    if DEBUG >= 1:
      print(f"Dequantizing {key}: weight={weight.shape} scales={scales.shape} biases={biases.shape}")

    dequantized = dequantize_mlx_4bit(weight, scales, biases, group_size)

    if DEBUG >= 1:
      print(f"Dequantized {key}: {dequantized.shape} dtype={dequantized.dtype} device={dequantized.device}")

    return dequantized
  else:
    # Not quantized, but still need to move from DISK to CPU
    if tensor.device.startswith('DISK:'):
      tensor = tensor.to('CPU').realize()
    return tensor


# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)

  return {name: convert(name) for name in {name: None for model in models for name in model}}


def load(fn: str, shard: ShardMetadata, group_size: int = 64):
  if fn.endswith('.index.json'):
    with open(fn) as fp:
      weight_map_index = json.load(fp)['weight_map']
    parts = {}
    filtered_weight_map = {}
    # TODO: Re-implement get_allow_patterns if needed for tinygrad
    allow_patterns = None
    for k, n in weight_map_index.items():
      if allow_patterns is not None and not any(fnmatch(n, r) for r in allow_patterns):
        continue
      if k.startswith("model.layers."):
        layer_num = int(k.split('.')[2])
        if layer_num < shard.start_layer or layer_num > shard.end_layer:
          continue

      # Skip biases and scales - they'll be consumed during dequantization
      if k.endswith('.biases') or k.endswith('.scales'):
        continue

      # Load part files (this will dequantize automatically)
      if n not in parts:
        parts[n] = load(str(Path(fn).parent/Path(n).name), shard, group_size)

      filtered_weight_map[k] = n

    if DEBUG >= 2: print(f"Excluded model param keys for {shard=}: {sorted(set(weight_map_index.keys()) - set(filtered_weight_map.keys()))}")

    # Build final weight map, only including keys that exist after dequantization
    final_weights = {}
    for k, n in filtered_weight_map.items():
      if k in parts[n]:
        final_weights[k] = parts[n][k]
      elif DEBUG >= 1:
        print(f"Warning: Key {k} not found in {n} after dequantization")

    return final_weights
  elif fn.endswith(".safetensors"):
    weight_map = safe_load(fn)
    for k in list(weight_map):
      if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
          del weight_map[k]

    # Dequantize weights if they're quantized
    dequantized_map = {}
    processed_keys = set()  # Track which keys we've processed

    for k in weight_map.keys():
      if k in processed_keys:
        continue

      # Skip scales and biases - they'll be processed with their weights
      if k.endswith('.scales') or k.endswith('.biases'):
        processed_keys.add(k)
        continue

      # Dequantize if needed
      dequantized_map[k] = maybe_dequantize(weight_map, k, group_size)
      processed_keys.add(k)

      # Mark scales and biases as processed if they exist
      if k.endswith('.weight'):
        base_name = k[:-7]
        if is_quantized_weight(weight_map, base_name):
          processed_keys.add(f"{base_name}.scales")
          processed_keys.add(f"{base_name}.biases")

    return dequantized_map
  else:
    return torch_load(fn)
