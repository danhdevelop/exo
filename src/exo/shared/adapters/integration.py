"""
Integration utilities for connecting adapters to the API layer.

This module provides helper functions to integrate the input adapter and output parser
into the existing API flow with minimal code changes.
"""

from typing import Any

from loguru import logger

from exo.shared.adapters.input_adapter import adapt_parameters
from exo.shared.adapters.output_parser import ParsedOutput, parse_complete_output
from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.tasks import ChatCompletionTaskParams


def adapt_chat_completion_request(
    params: ChatCompletionTaskParams,
) -> ChatCompletionTaskParams:
    """
    Adapt incoming chat completion parameters for the target model.

    This should be called at the API entry point before creating the ChatCompletion command.

    Args:
        params: Original request parameters

    Returns:
        Adapted parameters suitable for the model
    """
    try:
        model_id = ModelId(params.model)
        adapted = adapt_parameters(model_id, params)
        return adapted
    except Exception as e:
        logger.warning(f"Failed to adapt parameters for {params.model}: {e}")
        # Return original on error to avoid breaking existing functionality
        return params


def parse_token_chunk_with_adapter(
    chunk: TokenChunk, accumulated_text: str = ""
) -> ParsedOutput:
    """
    Parse a token chunk using the model-specific output parser.

    This extracts thinking/reasoning tokens and normalizes the content.

    Args:
        chunk: Token chunk from the generator
        accumulated_text: Previously accumulated text for context

    Returns:
        ParsedOutput with extracted fields
    """
    try:
        model_id = ModelId(chunk.model)
        # Parse the complete accumulated text
        full_text = accumulated_text + chunk.text
        parsed = parse_complete_output(model_id, full_text)

        # Calculate delta for streaming
        if accumulated_text:
            prev_parsed = parse_complete_output(model_id, accumulated_text)

            # Content delta
            content_delta = None
            if parsed.content and prev_parsed.content:
                if len(parsed.content) > len(prev_parsed.content):
                    content_delta = parsed.content[len(prev_parsed.content) :]
            elif parsed.content:
                content_delta = parsed.content

            # Thinking delta
            thinking_delta = None
            if parsed.thinking and prev_parsed.thinking:
                if len(parsed.thinking) > len(prev_parsed.thinking):
                    thinking_delta = parsed.thinking[len(prev_parsed.thinking) :]
            elif parsed.thinking:
                thinking_delta = parsed.thinking

            return ParsedOutput(
                content=content_delta,
                thinking=thinking_delta,
                tool_calls=parsed.tool_calls,  # Tool calls are usually complete
                finish_reason=parsed.finish_reason,
                metadata=parsed.metadata,
            )

        return parsed

    except Exception as e:
        logger.warning(
            f"Failed to parse chunk for {chunk.model}: {e}. Using raw text."
        )
        # Fallback to raw text on error
        return ParsedOutput(content=chunk.text)


def enrich_chat_message_with_parsing(
    message: ChatCompletionMessage, model_id: ModelId, full_text: str
) -> ChatCompletionMessage:
    """
    Enrich a chat completion message by parsing its content for special formats.

    This is useful for non-streaming responses where we have the complete text.

    Args:
        message: Original message
        model_id: Model that generated the message
        full_text: Complete generated text

    Returns:
        Enriched message with thinking/tool calls extracted
    """
    try:
        parsed = parse_complete_output(model_id, full_text)

        # Update message fields
        if parsed.thinking:
            message.thinking = parsed.thinking
        if parsed.content is not None:
            message.content = parsed.content
        if parsed.tool_calls:
            message.tool_calls = parsed.tool_calls

        return message

    except Exception as e:
        logger.warning(f"Failed to enrich message for {model_id}: {e}")
        return message


def should_extract_thinking(model_id: ModelId) -> bool:
    """
    Check if a model requires thinking extraction.

    Args:
        model_id: Model identifier

    Returns:
        True if thinking extraction should be attempted
    """
    from exo.shared.models.model_capabilities import get_model_capabilities

    capabilities = get_model_capabilities(model_id)
    return capabilities.supports_thinking


def validate_model_supports_features(
    model_id: ModelId, requested_features: dict[str, Any]
) -> list[str]:
    """
    Validate that a model supports the requested features.

    Args:
        model_id: Model identifier
        requested_features: Dict of feature names to their values (e.g., {"tools": [...], "stream": True})

    Returns:
        List of warning messages for unsupported features (empty if all supported)
    """
    from exo.shared.models.model_capabilities import get_model_capabilities

    warnings = []
    capabilities = get_model_capabilities(model_id)

    if requested_features.get("tools") and not capabilities.supports_tools:
        warnings.append(f"Model {model_id} does not support tool calling")

    if requested_features.get("stream") and not capabilities.supports_streaming:
        warnings.append(f"Model {model_id} may not support streaming properly")

    if requested_features.get("parallel_tool_calls") and not capabilities.supports_parallel_tool_calls:
        warnings.append(f"Model {model_id} does not support parallel tool calls")

    return warnings
