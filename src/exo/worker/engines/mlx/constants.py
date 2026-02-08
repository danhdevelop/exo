# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None  # Auto-detect from model (None = no quantization, or detect from model config)
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200  # Sliding window: keep last 3.2K tokens (reduces memory usage)
KEEP_KV_SIZE: int | None = 1600  # Keep 1.6K tokens when rotating
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None  # Auto-detect from model (None = auto-detect, or set to 4/8 to force)

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
