# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = 4  # 4-bit quantization during generation for 128K support
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 8192
MAX_KV_SIZE: int | None = None  # Unlimited KV cache for 128K token support
KEEP_KV_SIZE: int | None = None  # Not using rotating cache
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = 4  # 4-bit quantized KV cache for memory efficiency
TEMPERATURE: float = 1.0

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
