from tinygrad import Tensor, Variable, dtypes
from tinygrad.helpers import getenv
from collections import OrderedDict
from typing import List, Optional

def create_kv_cache(x: Tensor, layer, dtype=None):
  """Create KV cache with the correct dtype.

  Args:
    x: Input tensor (used for batch size and device)
    layer: Model layer (used for max_context, n_kv_heads, head_dim)
    dtype: Data type for the cache (if None, uses model embedding dtype)
  """
  # Use the provided dtype, or default to half (fp16)
  # Token IDs are long/int, but embeddings and KV cache should be fp16/bf16
  cache_dtype = dtype if dtype is not None else dtypes.half

  # Create cache - don't specify device to avoid WebGPU ctypes issues
  # TinyGrad will handle device placement automatically
  cache_kv = Tensor.zeros(2, x.shape[0], layer.max_context, layer.n_kv_heads, layer.head_dim, dtype=cache_dtype).contiguous().realize()
  if isinstance(x.device, tuple):
    # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
    cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()
  return cache_kv.realize()

class ModelState:
  cache: List[Tensor]
  start: int
  def __init__(self, cache: List[Tensor], start: int = 0):
    self.cache = cache
    self.start = start

def make_prompt_state(x: Tensor, model):
  """Create initial prompt state with KV cache.

  Args:
    x: Input token IDs tensor
    model: The model (used to get embedding dtype)
  """
  # Get dtype from model's token embeddings
  # This ensures cache dtype matches the model's computation dtype
  embedding_dtype = model.tok_embeddings.weight.dtype

  cache = [create_kv_cache(x, l.attention, dtype=embedding_dtype) for l in model.layers]

  return ModelState(cache)
