"""
Model capability registry that defines what features each model family supports.

This module provides a central registry of model capabilities, parameter constraints,
and special token configurations for different LLM families. It enables the adapter
system to properly handle model-specific features and format differences.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from exo.shared.models.model_cards import ModelId


class OutputFormat(str, Enum):
    """Different output format types that models may produce."""

    STANDARD = "standard"  # Standard text completion
    THINKING = "thinking"  # Explicit thinking tokens (e.g., GPT-OSS, DeepSeek-R1)
    TOOL_CALLS = "tool_calls"  # Native tool calling support
    HARMONY = "harmony"  # GPT-OSS Harmony format with <think> tags
    JSON_MODE = "json_mode"  # Structured JSON output


class ToolCallFormat(str, Enum):
    """Different tool call format conventions."""

    OPENAI = "openai"  # Standard OpenAI format with function.arguments as JSON string
    DICT = "dict"  # Arguments as dict (GLM, some Qwen variants)
    CUSTOM = "custom"  # Model-specific format requiring custom parser


class ThinkingTokenType(str, Enum):
    """Different thinking/reasoning token conventions."""

    NONE = "none"  # No thinking support
    XML_TAGS = "xml_tags"  # <think>...</think> or <thinking>...</thinking>
    SPECIAL_TOKENS = "special_tokens"  # Special tokenizer tokens
    STRUCTURED_FIELD = "structured_field"  # Separate field in response


@dataclass(frozen=True)
class ParameterConstraints:
    """Parameter constraints for a model."""

    max_tokens_limit: int | None = None
    temperature_min: float = 0.0
    temperature_max: float = 2.0
    top_p_min: float = 0.0
    top_p_max: float = 1.0
    top_k_min: int = 1
    top_k_max: int | None = None
    supports_frequency_penalty: bool = True
    supports_presence_penalty: bool = True
    supports_repetition_penalty: bool = True
    supports_logit_bias: bool = True
    supports_logprobs: bool = False
    supports_seed: bool = True


@dataclass(frozen=True)
class SpecialTokens:
    """Special tokens used by a model for various purposes."""

    thinking_start: str | None = None
    thinking_end: str | None = None
    tool_call_start: str | None = None
    tool_call_end: str | None = None
    reasoning_start: str | None = None
    reasoning_end: str | None = None
    eos_tokens: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ModelCapabilities:
    """Comprehensive capability definition for a model family."""

    model_family: str
    supports_tools: bool = False
    supports_thinking: bool = False
    supports_streaming: bool = True
    supports_system_message: bool = True
    supports_parallel_tool_calls: bool = False
    output_formats: list[OutputFormat] = field(
        default_factory=lambda: [OutputFormat.STANDARD]
    )
    tool_call_format: ToolCallFormat = ToolCallFormat.OPENAI
    thinking_token_type: ThinkingTokenType = ThinkingTokenType.NONE
    parameter_constraints: ParameterConstraints = field(
        default_factory=ParameterConstraints
    )
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
    requires_custom_parser: bool = False
    custom_parser_hints: dict[str, Any] = field(default_factory=dict)


# Model capability registry - maps model ID patterns to capabilities
MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    "deepseek": ModelCapabilities(
        model_family="deepseek",
        supports_tools=True,
        supports_thinking=True,  # DeepSeek-R1 and newer support reasoning
        supports_streaming=True,
        supports_system_message=True,
        supports_parallel_tool_calls=True,
        output_formats=[
            OutputFormat.STANDARD,
            OutputFormat.THINKING,
            OutputFormat.TOOL_CALLS,
        ],
        tool_call_format=ToolCallFormat.OPENAI,
        thinking_token_type=ThinkingTokenType.XML_TAGS,
        parameter_constraints=ParameterConstraints(
            max_tokens_limit=8192,
            temperature_max=2.0,
            supports_logprobs=False,
        ),
        special_tokens=SpecialTokens(
            thinking_start="<think>", thinking_end="</think>"
        ),
        requires_custom_parser=True,
        custom_parser_hints={"check_reasoning_tags": True},
    ),
    "kimi": ModelCapabilities(
        model_family="kimi",
        supports_tools=True,
        supports_thinking=True,  # Kimi-K2-Thinking variant
        supports_streaming=True,
        supports_system_message=True,
        supports_parallel_tool_calls=True,
        output_formats=[
            OutputFormat.STANDARD,
            OutputFormat.THINKING,
            OutputFormat.TOOL_CALLS,
        ],
        tool_call_format=ToolCallFormat.OPENAI,
        thinking_token_type=ThinkingTokenType.XML_TAGS,
        parameter_constraints=ParameterConstraints(max_tokens_limit=8192),
        special_tokens=SpecialTokens(
            thinking_start="<thinking>",
            thinking_end="</thinking>",
            eos_tokens=[163586],
        ),
        requires_custom_parser=True,
        custom_parser_hints={"extract_thinking_blocks": True},
    ),
    "llama": ModelCapabilities(
        model_family="llama",
        supports_tools=True,
        supports_thinking=False,
        supports_streaming=True,
        supports_system_message=True,
        supports_parallel_tool_calls=True,
        output_formats=[OutputFormat.STANDARD, OutputFormat.TOOL_CALLS],
        tool_call_format=ToolCallFormat.OPENAI,
        thinking_token_type=ThinkingTokenType.NONE,
        parameter_constraints=ParameterConstraints(
            max_tokens_limit=8192, supports_logprobs=False
        ),
        requires_custom_parser=False,
    ),
    "qwen-coder": ModelCapabilities(
        model_family="qwen-coder",
        supports_tools=True,
        supports_thinking=False,  # Standard Qwen Coder doesn't have thinking
        supports_streaming=True,
        supports_system_message=True,
        supports_parallel_tool_calls=True,
        output_formats=[OutputFormat.STANDARD, OutputFormat.TOOL_CALLS],
        tool_call_format=ToolCallFormat.DICT,  # Qwen uses dict format
        thinking_token_type=ThinkingTokenType.NONE,
        parameter_constraints=ParameterConstraints(
            max_tokens_limit=8192, supports_logprobs=False
        ),
        requires_custom_parser=True,
        custom_parser_hints={
            "tool_args_as_dict": True,
            "qwen_coder": True,
        },
    ),
    "qwen-thinking": ModelCapabilities(
        model_family="qwen-thinking",
        supports_tools=True,
        supports_thinking=True,  # Qwen-Thinking variants have explicit thinking
        supports_streaming=True,
        supports_system_message=True,
        supports_parallel_tool_calls=True,
        output_formats=[
            OutputFormat.STANDARD,
            OutputFormat.THINKING,
            OutputFormat.TOOL_CALLS,
        ],
        tool_call_format=ToolCallFormat.DICT,  # Qwen uses dict format
        thinking_token_type=ThinkingTokenType.XML_TAGS,
        parameter_constraints=ParameterConstraints(
            max_tokens_limit=8192, supports_logprobs=False
        ),
        special_tokens=SpecialTokens(
            thinking_start="<think>", thinking_end="</think>"
        ),
        requires_custom_parser=True,
        custom_parser_hints={
            "tool_args_as_dict": True,
            "extract_thinking_blocks": True,
        },
    ),
    "qwen": ModelCapabilities(
        model_family="qwen",
        supports_tools=True,
        supports_thinking=False,  # Standard Qwen3 doesn't have thinking
        supports_streaming=True,
        supports_system_message=True,
        supports_parallel_tool_calls=True,
        output_formats=[OutputFormat.STANDARD, OutputFormat.TOOL_CALLS],
        tool_call_format=ToolCallFormat.DICT,  # Qwen uses dict format
        thinking_token_type=ThinkingTokenType.NONE,
        parameter_constraints=ParameterConstraints(
            max_tokens_limit=8192, supports_logprobs=False
        ),
        requires_custom_parser=True,
        custom_parser_hints={
            "tool_args_as_dict": True,
        },
    ),
    "gpt-oss": ModelCapabilities(
        model_family="gpt-oss",
        supports_tools=True,
        supports_thinking=True,  # Native Harmony format support
        supports_streaming=True,
        supports_system_message=True,
        supports_parallel_tool_calls=False,
        output_formats=[
            OutputFormat.STANDARD,
            OutputFormat.HARMONY,
            OutputFormat.TOOL_CALLS,
        ],
        tool_call_format=ToolCallFormat.OPENAI,
        thinking_token_type=ThinkingTokenType.STRUCTURED_FIELD,
        parameter_constraints=ParameterConstraints(max_tokens_limit=8192),
        special_tokens=SpecialTokens(
            thinking_start="<think>",
            thinking_end="</think>",
            eos_tokens=[200002, 199999],  # <|return|>, <|endoftext|>
        ),
        requires_custom_parser=True,
        custom_parser_hints={
            "harmony_format": True,
            "thinking_as_field": True,
            "check_think_tags": True,
        },
    ),
    "glm": ModelCapabilities(
        model_family="glm",
        supports_tools=True,
        supports_thinking=False,
        supports_streaming=True,
        supports_system_message=True,
        supports_parallel_tool_calls=True,
        output_formats=[OutputFormat.STANDARD, OutputFormat.TOOL_CALLS],
        tool_call_format=ToolCallFormat.DICT,  # GLM expects dict format
        thinking_token_type=ThinkingTokenType.NONE,
        parameter_constraints=ParameterConstraints(
            max_tokens_limit=8192, supports_logprobs=False
        ),
        special_tokens=SpecialTokens(
            eos_tokens=[151336, 151329, 151338]  # GLM-4.5/4.7
        ),
        requires_custom_parser=True,
        custom_parser_hints={"tool_args_as_dict": True, "glm_format": True},
    ),
    "minimax": ModelCapabilities(
        model_family="minimax",
        supports_tools=True,
        supports_thinking=False,
        supports_streaming=True,
        supports_system_message=True,
        supports_parallel_tool_calls=True,
        output_formats=[OutputFormat.STANDARD, OutputFormat.TOOL_CALLS],
        tool_call_format=ToolCallFormat.OPENAI,
        thinking_token_type=ThinkingTokenType.NONE,
        parameter_constraints=ParameterConstraints(max_tokens_limit=8192),
        requires_custom_parser=False,
    ),
}


def get_model_capabilities(model_id: ModelId) -> ModelCapabilities:
    """
    Get capabilities for a model based on its ID.

    Args:
        model_id: The model identifier (e.g., "deepseek-v3.1-4bit")

    Returns:
        ModelCapabilities object for the model family

    Raises:
        ValueError: If model family is not recognized
    """
    model_id_lower = model_id.lower()

    # Match against known model families with priority order
    # Check more specific variants first, then general families

    # Qwen variants (check specific before general)
    if "qwen" in model_id_lower and "coder" in model_id_lower:
        return MODEL_CAPABILITIES["qwen-coder"]
    elif "qwen" in model_id_lower and "thinking" in model_id_lower:
        return MODEL_CAPABILITIES["qwen-thinking"]
    elif "qwen" in model_id_lower:
        return MODEL_CAPABILITIES["qwen"]

    # Other model families
    for family_key in ["deepseek", "kimi", "llama", "gpt-oss", "glm", "minimax"]:
        if family_key in model_id_lower:
            return MODEL_CAPABILITIES[family_key]

    # Default capabilities for unknown models
    return ModelCapabilities(
        model_family="unknown",
        supports_tools=False,
        supports_thinking=False,
        supports_streaming=True,
        supports_system_message=True,
        output_formats=[OutputFormat.STANDARD],
        parameter_constraints=ParameterConstraints(),
    )


def supports_feature(model_id: ModelId, feature: str) -> bool:
    """
    Check if a model supports a specific feature.

    Args:
        model_id: The model identifier
        feature: Feature name (e.g., "tools", "thinking", "streaming")

    Returns:
        True if the feature is supported, False otherwise
    """
    capabilities = get_model_capabilities(model_id)
    return getattr(capabilities, f"supports_{feature}", False)


def get_parameter_constraints(model_id: ModelId) -> ParameterConstraints:
    """Get parameter constraints for a model."""
    capabilities = get_model_capabilities(model_id)
    return capabilities.parameter_constraints


def get_special_tokens(model_id: ModelId) -> SpecialTokens:
    """Get special tokens for a model."""
    capabilities = get_model_capabilities(model_id)
    return capabilities.special_tokens
