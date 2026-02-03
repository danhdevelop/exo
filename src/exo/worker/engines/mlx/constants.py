# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = 4  # 4-bit quantization during generation for 128K support
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 4000  # Sliding window: keep last 4K tokens (RotatingKVCache doesn't support quantization)
KEEP_KV_SIZE: int | None = 2000  # Keep 2K tokens when rotating
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = 4  # 4-bit quantized KV cache for memory efficiency
TEMPERATURE: float = 1.0

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
