import json
from pathlib import Path
from typing import Any, cast
from loguru import logger
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.utils.pydantic_ext import CamelCaseModel


class ModelCard(CamelCaseModel):
    short_id: str
    model_id: ModelId
    name: str
    description: str
    tags: list[str]
    metadata: ModelMetadata


MODEL_CARDS: dict[str, ModelCard] = {
    # deepseek v3
    "deepseek-coder-v2-lite-instruct-mlx-6bit": ModelCard(
        short_id="deepseek-coder-v2-lite-instruct-mlx-6bit",
        model_id=ModelId("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-6bit"),
        name="DeepSeek Coder (6-bit)",
        description="""DeepSeek Coder is a large language model trained on the DeepSeek Coder dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-6bit"),
            pretty_name="DeepSeek Coder (6-bit)",
            storage_size=Memory.from_gb(12.8),
            n_layers=27,
            hidden_size=2048,
            supports_tensor=False,
        ),
    ),
    "deepseek-v3.1-4bit": ModelCard(
        short_id="deepseek-v3.1-4bit",
        model_id=ModelId("mlx-community/DeepSeek-V3.1-4bit"),
        name="DeepSeek V3.1 (4-bit)",
        description="""DeepSeek V3.1 is a large language model trained on the DeepSeek V3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-V3.1-4bit"),
            pretty_name="DeepSeek V3.1 (4-bit)",
            storage_size=Memory.from_gb(378),
            n_layers=61,
            hidden_size=7168,
            supports_tensor=True,
        ),
    ),
    "deepseek-v3.1-8bit": ModelCard(
        short_id="deepseek-v3.1-8bit",
        model_id=ModelId("mlx-community/DeepSeek-V3.1-8bit"),
        name="DeepSeek V3.1 (8-bit)",
        description="""DeepSeek V3.1 is a large language model trained on the DeepSeek V3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-V3.1-8bit"),
            pretty_name="DeepSeek V3.1 (8-bit)",
            storage_size=Memory.from_gb(713),
            n_layers=61,
            hidden_size=7168,
            supports_tensor=True,
        ),
    ),
    # kimi k2
    "kimi-k2-instruct-4bit": ModelCard(
        short_id="kimi-k2-instruct-4bit",
        model_id=ModelId("mlx-community/Kimi-K2-Instruct-4bit"),
        name="Kimi K2 Instruct (4-bit)",
        description="""Kimi K2 is a large language model trained on the Kimi K2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Kimi-K2-Instruct-4bit"),
            pretty_name="Kimi K2 Instruct (4-bit)",
            storage_size=Memory.from_gb(578),
            n_layers=61,
            hidden_size=7168,
            supports_tensor=True,
        ),
    ),
    "kimi-k2-thinking": ModelCard(
        short_id="kimi-k2-thinking",
        model_id=ModelId("mlx-community/Kimi-K2-Thinking"),
        name="Kimi K2 Thinking (4-bit)",
        description="""Kimi K2 Thinking is the latest, most capable version of open-source thinking model.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Kimi-K2-Thinking"),
            pretty_name="Kimi K2 Thinking (4-bit)",
            storage_size=Memory.from_gb(658),
            n_layers=61,
            hidden_size=7168,
            supports_tensor=True,
        ),
    ),
    # llama-3.1
    "llama-3.1-8b": ModelCard(
        short_id="llama-3.1-8b",
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        name="Llama 3.1 8B (4-bit)",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
            pretty_name="Llama 3.1 8B (4-bit)",
            storage_size=Memory.from_mb(4423),
            n_layers=32,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "llama-3.1-8b-8bit": ModelCard(
        short_id="llama-3.1-8b-8bit",
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"),
        name="Llama 3.1 8B (8-bit)",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"),
            pretty_name="Llama 3.1 8B (8-bit)",
            storage_size=Memory.from_mb(8540),
            n_layers=32,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "llama-3.1-8b-bf16": ModelCard(
        short_id="llama-3.1-8b-bf16",
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"),
        name="Llama 3.1 8B (BF16)",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"),
            pretty_name="Llama 3.1 8B (BF16)",
            storage_size=Memory.from_mb(16100),
            n_layers=32,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "llama-3.1-70b": ModelCard(
        short_id="llama-3.1-70b",
        model_id=ModelId("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"),
        name="Llama 3.1 70B (4-bit)",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"),
            pretty_name="Llama 3.1 70B (4-bit)",
            storage_size=Memory.from_mb(38769),
            n_layers=80,
            hidden_size=8192,
            supports_tensor=True,
        ),
    ),
    # llama-3.2
    "llama-3.2-1b": ModelCard(
        short_id="llama-3.2-1b",
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        name="Llama 3.2 1B (4-bit)",
        description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
            pretty_name="Llama 3.2 1B (4-bit)",
            storage_size=Memory.from_mb(696),
            n_layers=16,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "llama-3.2-3b": ModelCard(
        short_id="llama-3.2-3b",
        model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
        name="Llama 3.2 3B (4-bit)",
        description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
            pretty_name="Llama 3.2 3B (4-bit)",
            storage_size=Memory.from_mb(1777),
            n_layers=28,
            hidden_size=3072,
            supports_tensor=True,
        ),
    ),
    "llama-3.2-3b-8bit": ModelCard(
        short_id="llama-3.2-3b-8bit",
        model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-8bit"),
        name="Llama 3.2 3B (8-bit)",
        description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-8bit"),
            pretty_name="Llama 3.2 3B (8-bit)",
            storage_size=Memory.from_mb(3339),
            n_layers=28,
            hidden_size=3072,
            supports_tensor=True,
        ),
    ),
    # llama-3.3
    "llama-3.3-70b": ModelCard(
        short_id="llama-3.3-70b",
        model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
        name="Llama 3.3 70B (4-bit)",
        description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
            pretty_name="Llama 3.3 70B",
            storage_size=Memory.from_mb(38769),
            n_layers=80,
            hidden_size=8192,
            supports_tensor=True,
        ),
    ),
    "llama-3.3-70b-8bit": ModelCard(
        short_id="llama-3.3-70b-8bit",
        model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-8bit"),
        name="Llama 3.3 70B (8-bit)",
        description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-8bit"),
            pretty_name="Llama 3.3 70B (8-bit)",
            storage_size=Memory.from_mb(73242),
            n_layers=80,
            hidden_size=8192,
            supports_tensor=True,
        ),
    ),
    "llama-3.3-70b-fp16": ModelCard(
        short_id="llama-3.3-70b-fp16",
        model_id=ModelId("mlx-community/llama-3.3-70b-instruct-fp16"),
        name="Llama 3.3 70B (FP16)",
        description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/llama-3.3-70b-instruct-fp16"),
            pretty_name="Llama 3.3 70B (FP16)",
            storage_size=Memory.from_mb(137695),
            n_layers=80,
            hidden_size=8192,
            supports_tensor=True,
        ),
    ),
    # qwen3
    "qwen3-0.6b": ModelCard(
        short_id="qwen3-0.6b",
        model_id=ModelId("mlx-community/Qwen3-0.6B-4bit"),
        name="Qwen3 0.6B (4-bit)",
        description="""Qwen3 0.6B is a large language model trained on the Qwen3 0.6B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-0.6B-4bit"),
            pretty_name="Qwen3 0.6B (4-bit)",
            storage_size=Memory.from_mb(327),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
        ),
    ),
    "qwen3-0.6b-8bit": ModelCard(
        short_id="qwen3-0.6b-8bit",
        model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        name="Qwen3 0.6B (8-bit)",
        description="""Qwen3 0.6B is a large language model trained on the Qwen3 0.6B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
            pretty_name="Qwen3 0.6B (8-bit)",
            storage_size=Memory.from_mb(666),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
        ),
    ),
    "qwen3-coder-30b-4bit": ModelCard(
        short_id="qwen3-coder-30b-4bit",
        model_id=ModelId("lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit"),
        name="Qwen3 Coder 30B A3B (4-bit)",
        description="""Qwen3 Coder 30B is a large language model trained on the Qwen3 Coder 30B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit"),
            pretty_name="Qwen3 Coder 30B A3B (4-bit)",
            storage_size=Memory.from_gb(17.2),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-coder-30b-6bit": ModelCard(
        short_id="qwen3-coder-30b-6bit",
        model_id=ModelId("lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit"),
        name="Qwen3 Coder 30B A3B (6-bit)",
        description="""Qwen3 Coder 30B is a large language model trained on the Qwen3 Coder 30B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit"),
            pretty_name="Qwen3 Coder 30B A3B (6-bit)",
            storage_size=Memory.from_gb(17.8),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-80b-a3B-4bit": ModelCard(
        short_id="qwen3-80b-a3B-4bit",
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"),
        name="Qwen3 80B A3B (4-bit)",
        description="""Qwen3 80B""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"),
            pretty_name="Qwen3 80B A3B (4-bit)",
            storage_size=Memory.from_mb(44800),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-80b-a3B-8bit": ModelCard(
        short_id="qwen3-80b-a3B-8bit",
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit"),
        name="Qwen3 80B A3B (8-bit)",
        description="""Qwen3 80B""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit"),
            pretty_name="Qwen3 80B A3B (8-bit)",
            storage_size=Memory.from_mb(84700),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-80b-a3B-thinking-4bit": ModelCard(
        short_id="qwen3-80b-a3B-thinking-4bit",
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit"),
        name="Qwen3 80B A3B Thinking (4-bit)",
        description="""Qwen3 80B Reasoning model""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit"),
            pretty_name="Qwen3 80B A3B (4-bit)",
            storage_size=Memory.from_mb(84700),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-80b-a3B-thinking-8bit": ModelCard(
        short_id="qwen3-80b-a3B-thinking-8bit",
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit"),
        name="Qwen3 80B A3B Thinking (8-bit)",
        description="""Qwen3 80B Reasoning model""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit"),
            pretty_name="Qwen3 80B A3B (8-bit)",
            storage_size=Memory.from_mb(84700),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-235b-a22b-4bit": ModelCard(
        short_id="qwen3-235b-a22b-4bit",
        model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
        name="Qwen3 235B A22B (4-bit)",
        description="""Qwen3 235B (Active 22B) is a large language model trained on the Qwen3 235B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
            pretty_name="Qwen3 235B A22B (4-bit)",
            storage_size=Memory.from_gb(132),
            n_layers=94,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "qwen3-235b-a22b-8bit": ModelCard(
        short_id="qwen3-235b-a22b-8bit",
        model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit"),
        name="Qwen3 235B A22B (8-bit)",
        description="""Qwen3 235B (Active 22B) is a large language model trained on the Qwen3 235B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit"),
            pretty_name="Qwen3 235B A22B (8-bit)",
            storage_size=Memory.from_gb(250),
            n_layers=94,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "qwen3-coder-480b-a35b-4bit": ModelCard(
        short_id="qwen3-coder-480b-a35b-4bit",
        model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"),
        name="Qwen3 Coder 480B A35B (4-bit)",
        description="""Qwen3 Coder 480B (Active 35B) is a large language model trained on the Qwen3 Coder 480B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"),
            pretty_name="Qwen3 Coder 480B A35B (4-bit)",
            storage_size=Memory.from_gb(270),
            n_layers=62,
            hidden_size=6144,
            supports_tensor=True,
        ),
    ),
    "qwen3-coder-480b-a35b-8bit": ModelCard(
        short_id="qwen3-coder-480b-a35b-8bit",
        model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit"),
        name="Qwen3 Coder 480B A35B (8-bit)",
        description="""Qwen3 Coder 480B (Active 35B) is a large language model trained on the Qwen3 Coder 480B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit"),
            pretty_name="Qwen3 Coder 480B A35B (8-bit)",
            storage_size=Memory.from_gb(540),
            n_layers=62,
            hidden_size=6144,
            supports_tensor=True,
        ),
    ),
    # gpt-oss
    "gpt-oss-120b-MXFP4-Q8": ModelCard(
        short_id="gpt-oss-120b-MXFP4-Q8",
        model_id=ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8"),
        name="GPT-OSS 120B (MXFP4-Q8, MLX)",
        description="""OpenAI's GPT-OSS 120B is a 117B-parameter Mixture-of-Experts model designed for high-reasoning and general-purpose use; this variant is a 4-bit MLX conversion for Apple Silicon.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8"),
            pretty_name="GPT-OSS 120B (MXFP4-Q8, MLX)",
            storage_size=Memory.from_kb(68_996_301),
            n_layers=36,
            hidden_size=2880,
            supports_tensor=True,
        ),
    ),
    "gpt-oss-20b-MXFP4-Q8": ModelCard(
        short_id="gpt-oss-20b-MXFP4-Q8",
        model_id=ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
        name="GPT-OSS 20B (MXFP4-Q8, MLX)",
        description="""OpenAI's GPT-OSS 20B is a medium-sized MoE model for lower-latency and local or specialized use cases; this variant is a 4-bit MLX conversion for Apple Silicon.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
            pretty_name="GPT-OSS 20B (MXFP4-Q8, MLX)",
            storage_size=Memory.from_kb(11_744_051),
            n_layers=24,
            hidden_size=2880,
            supports_tensor=True,
        ),
    ),
    # glm 4.5
    "glm-4.5-air-8bit": ModelCard(
        # Needs to be quantized g32 or g16 to work with tensor parallel
        short_id="glm-4.5-air-8bit",
        model_id=ModelId("mlx-community/GLM-4.5-Air-8bit"),
        name="GLM 4.5 Air 8bit",
        description="""GLM 4.5 Air 8bit""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/GLM-4.5-Air-8bit"),
            pretty_name="GLM 4.5 Air 8bit",
            storage_size=Memory.from_gb(114),
            n_layers=46,
            hidden_size=4096,
            supports_tensor=False,
        ),
    ),
    "glm-4.5-air-bf16": ModelCard(
        short_id="glm-4.5-air-bf16",
        model_id=ModelId("mlx-community/GLM-4.5-Air-bf16"),
        name="GLM 4.5 Air bf16",
        description="""GLM 4.5 Air bf16""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/GLM-4.5-Air-bf16"),
            pretty_name="GLM 4.5 Air bf16",
            storage_size=Memory.from_gb(214),
            n_layers=46,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    # glm 4.7
    "glm-4.7-4bit": ModelCard(
        short_id="glm-4.7-4bit",
        model_id=ModelId("mlx-community/GLM-4.7-4bit"),
        name="GLM 4.7 4bit",
        description="GLM 4.7 4bit",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/GLM-4.7-4bit"),
            pretty_name="GLM 4.7 4bit",
            storage_size=Memory.from_bytes(198556925568),
            n_layers=91,
            hidden_size=5120,
            supports_tensor=True,
        ),
    ),
    "glm-4.7-6bit": ModelCard(
        short_id="glm-4.7-6bit",
        model_id=ModelId("mlx-community/GLM-4.7-6bit"),
        name="GLM 4.7 6bit",
        description="GLM 4.7 6bit",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/GLM-4.7-6bit"),
            pretty_name="GLM 4.7 6bit",
            storage_size=Memory.from_bytes(286737579648),
            n_layers=91,
            hidden_size=5120,
            supports_tensor=True,
        ),
    ),
    "glm-4.7-8bit-gs32": ModelCard(
        short_id="glm-4.7-8bit-gs32",
        model_id=ModelId("mlx-community/GLM-4.7-8bit-gs32"),
        name="GLM 4.7 8bit (gs32)",
        description="GLM 4.7 8bit (gs32)",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/GLM-4.7-8bit-gs32"),
            pretty_name="GLM 4.7 8bit (gs32)",
            storage_size=Memory.from_bytes(396963397248),
            n_layers=91,
            hidden_size=5120,
            supports_tensor=True,
        ),
    ),
    # minimax-m2
    "minimax-m2.1-8bit": ModelCard(
        short_id="minimax-m2.1-8bit",
        model_id=ModelId("mlx-community/MiniMax-M2.1-8bit"),
        name="MiniMax M2.1 8bit",
        description="MiniMax M2.1 8bit",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/MiniMax-M2.1-8bit"),
            pretty_name="MiniMax M2.1 8bit",
            storage_size=Memory.from_bytes(242986745856),
            n_layers=61,
            hidden_size=3072,
            supports_tensor=True,
        ),
    ),
    "minimax-m2.1-3bit": ModelCard(
        short_id="minimax-m2.1-3bit",
        model_id=ModelId("mlx-community/MiniMax-M2.1-3bit"),
        name="MiniMax M2.1 3bit",
        description="MiniMax M2.1 3bit",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/MiniMax-M2.1-3bit"),
            pretty_name="MiniMax M2.1 3bit",
            storage_size=Memory.from_bytes(100086644736),
            n_layers=61,
            hidden_size=3072,
            supports_tensor=True,
        ),
    ),
}

# Persistent storage for custom models - use EXO_HOME for consistency
def _get_custom_models_path() -> Path:
    """Get the path to custom models JSON, using EXO_HOME if set."""
    import os
    exo_home = os.environ.get("EXO_HOME")
    if exo_home:
        base_path = Path(exo_home)
    else:
        base_path = Path.home() / ".exo"
    return base_path / "custom_models.json"

PERSISTENT_FILE_PATH = _get_custom_models_path()
custom_models_loaded = False

def load_custom_models_once() -> None:
    """Loads custom models from persistent storage (called once on first access)."""
    global custom_models_loaded
    if custom_models_loaded:
        return
    custom_models_loaded = True

    logger.debug(f"Attempting to load custom models from: {PERSISTENT_FILE_PATH}")
    
    if not PERSISTENT_FILE_PATH.exists():
        logger.debug(f"Custom models file does not exist yet: {PERSISTENT_FILE_PATH}")
        return
    
    try:
        with open(PERSISTENT_FILE_PATH, "r") as f:
            data = cast(dict[str, dict[str, Any]], json.load(f))
            logger.debug(f"Loaded {len(data)} entries from custom_models.json")
            
            loaded_count = 0
            for key, card_data in data.items():
                try:
                    card = ModelCard.model_validate(card_data)
                    if key not in MODEL_CARDS:
                        MODEL_CARDS[key] = card
                        loaded_count += 1
                        logger.debug(f"Loaded custom model: {key} ({card.model_id})")
                    else:
                        logger.debug(f"Skipping {key} - already in MODEL_CARDS")
                except Exception as e:
                    logger.warning(f"Failed to load model card for {key}: {e}")
            
            logger.info(f"✓ Loaded {loaded_count} custom models from {PERSISTENT_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error loading custom models from {PERSISTENT_FILE_PATH}: {e}")

def save_custom_models() -> None:
    """Saves custom models from MODEL_CARDS to persistent storage."""
    logger.debug(f"Attempting to save custom models to: {PERSISTENT_FILE_PATH}")
    
    try:
        PERSISTENT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {PERSISTENT_FILE_PATH.parent}")
        
        custom_models = {
            key: card.model_dump(mode="json") 
            for key, card in MODEL_CARDS.items() 
            if "custom" in card.tags
        }
        
        logger.debug(f"Found {len(custom_models)} custom models to save")
        for key in custom_models.keys():
            logger.debug(f"  - {key}")
        
        with open(PERSISTENT_FILE_PATH, "w") as f:
            json.dump(custom_models, f, indent=4)
        
        logger.info(f"✓ Saved {len(custom_models)} custom models to {PERSISTENT_FILE_PATH}")
    except Exception as e:
        logger.error(f"Failed to save custom models to {PERSISTENT_FILE_PATH}: {e}")

def get_model_cards() -> dict[str, ModelCard]:
    """Returns MODEL_CARDS, loading custom models on first access."""
    load_custom_models_once()
    return MODEL_CARDS
