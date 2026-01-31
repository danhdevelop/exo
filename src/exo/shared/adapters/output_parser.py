"""
Output format parser for normalizing model-specific outputs to OpenAI format.

This module handles parsing and normalization of outputs from different LLM models,
including thinking tokens, tool calls, and other special formats. It ensures all
outputs conform to the OpenAI API response structure.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from exo.shared.models.model_capabilities import (
    ThinkingTokenType,
    get_model_capabilities,
    get_special_tokens,
)
from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import ChatCompletionMessage, ToolCall, ToolCallItem


class ParsedOutput:
    """Container for parsed model output."""

    def __init__(
        self,
        content: str | None = None,
        thinking: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.thinking = thinking
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason
        self.metadata = metadata or {}

    def to_message(self) -> ChatCompletionMessage:
        """Convert parsed output to ChatCompletionMessage."""
        return ChatCompletionMessage(
            role="assistant",
            content=self.content,
            thinking=self.thinking,
            tool_calls=self.tool_calls,
        )


class BaseOutputParser(ABC):
    """Base class for model-specific output parsers."""

    def __init__(self, model_id: ModelId):
        self.model_id = model_id
        self.capabilities = get_model_capabilities(model_id)
        self.special_tokens = get_special_tokens(model_id)

    @abstractmethod
    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """
        Parse a chunk of output text.

        Args:
            text: Current chunk text
            accumulated: Previously accumulated text for context

        Returns:
            ParsedOutput with extracted fields
        """
        pass

    @abstractmethod
    def parse_complete(self, text: str) -> ParsedOutput:
        """
        Parse complete output text.

        Args:
            text: Complete output text

        Returns:
            ParsedOutput with all extracted fields
        """
        pass

    def extract_thinking_tokens(self, text: str) -> tuple[str | None, str]:
        """
        Extract thinking/reasoning tokens from text.

        Args:
            text: Input text that may contain thinking tokens

        Returns:
            Tuple of (thinking_content, text_without_thinking)
        """
        if not self.capabilities.supports_thinking:
            return None, text

        if self.capabilities.thinking_token_type == ThinkingTokenType.XML_TAGS:
            return self._extract_xml_thinking(text)
        elif (
            self.capabilities.thinking_token_type == ThinkingTokenType.STRUCTURED_FIELD
        ):
            # Thinking is already a separate field, no extraction needed
            return None, text
        else:
            return None, text

    def _extract_xml_thinking(self, text: str) -> tuple[str | None, str]:
        """Extract thinking from XML-style tags, handling JSON content inside."""
        if not self.special_tokens.thinking_start or not self.special_tokens.thinking_end:
            # Try common patterns
            patterns: list[tuple[str, str, str]] = [
                (r"<think>(.*?)</think>", "<think>", "</think>"),
                (r"<thinking>(.*?)</thinking>", "<thinking>", "</thinking>"),
                (r"<reason>(.*?)</reason>", "<reason>", "</reason>"),
            ]
        else:
            start = re.escape(self.special_tokens.thinking_start)
            end = re.escape(self.special_tokens.thinking_end)
            patterns = [(rf"{start}(.*?){end}", self.special_tokens.thinking_start, self.special_tokens.thinking_end)]

        thinking_parts: list[str] = []
        remaining_text = text

        for pattern, _start_tag, _end_tag in patterns:
            matches = list(re.finditer(pattern, remaining_text, re.DOTALL))
            if matches:
                for match in matches:
                    inner_content = match.group(1).strip()
                    # Try to parse as JSON if it looks like JSON
                    processed_content = self._process_xml_inner_content(inner_content)
                    thinking_parts.append(processed_content)
                # Remove thinking tags from text
                remaining_text = re.sub(pattern, "", remaining_text, flags=re.DOTALL)

        if thinking_parts:
            thinking_content = "\n\n".join(thinking_parts)
            return thinking_content, remaining_text.strip()

        return None, text

    def _process_xml_inner_content(self, content: str) -> str:
        """
        Process content inside XML tags, handling JSON gracefully.

        If content is valid JSON, we extract and format it nicely.
        If it's not JSON, return as-is.
        """
        content = content.strip()

        # Check if it looks like JSON (starts with { or [)
        if not (content.startswith("{") or content.startswith("[")):
            return content

        try:
            # Try to parse as JSON
            parsed: Any = json.loads(content)  # type: ignore[reportAny]

            # If it's a dict with common reasoning fields, extract those
            if isinstance(parsed, dict):
                # Check for common reasoning structure fields
                reasoning: Any = parsed.get("reasoning")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
                conclusion: Any = parsed.get("conclusion")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
                thought: Any = parsed.get("thought")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
                thinking: Any = parsed.get("thinking")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]

                if reasoning is not None and conclusion is not None:
                    return f"{reasoning}\n\nConclusion: {conclusion}"
                elif thought is not None:
                    return str(thought)  # type: ignore[reportUnknownArgumentType]
                elif thinking is not None:
                    return str(thinking)  # type: ignore[reportUnknownArgumentType]
                # If it's a simple dict, format it nicely
                else:
                    return json.dumps(parsed, indent=2)

            # For arrays or other JSON, return formatted
            return json.dumps(parsed, indent=2)

        except json.JSONDecodeError:
            # Not valid JSON, return as-is
            # This handles partial JSON or JSON-like text
            return content

    def normalize_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """
        Normalize tool calls to OpenAI format.

        Args:
            tool_calls: Raw tool calls from model output

        Returns:
            List of normalized ToolCall objects
        """
        normalized: list[ToolCall] = []

        for idx, tc in enumerate(tool_calls):
            # Handle different tool call formats
            tool_id = tc.get("id")
            if tool_id is None or not isinstance(tool_id, str):
                tool_id = f"call_{idx}"

            tool_type_raw: Any = tc.get("type", "function")  # type: ignore[reportAny]
            # Only support "function" type for now
            if tool_type_raw != "function":
                continue

            function_data: Any = tc.get("function")
            if not isinstance(function_data, dict):
                continue

            name: Any = function_data.get("name", "")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
            if not isinstance(name, str):
                name = ""

            arguments: Any = function_data.get("arguments", "{}")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]

            # Normalize arguments to JSON string
            if isinstance(arguments, dict):
                arguments_str: str = json.dumps(arguments)
            elif isinstance(arguments, str):
                arguments_str = arguments
            else:
                arguments_str = str(arguments)  # type: ignore[reportUnknownArgumentType]

            normalized.append(
                ToolCall(
                    id=tool_id,
                    index=idx,
                    type="function",
                    function=ToolCallItem(name=name, arguments=arguments_str),
                )
            )

        return normalized

    def _detect_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Detect tool calls in output text, including JSON inside XML tags."""
        tool_calls: list[dict[str, Any]] = []

        # First, check if tool calls are wrapped in XML tags
        xml_wrapped_calls = self._extract_json_from_xml_tags(text)
        for call_data in xml_wrapped_calls:
            if isinstance(call_data, dict) and call_data.get("type") == "function":
                tool_calls.append(call_data)
            elif isinstance(call_data, list):
                # Multiple tool calls in one XML block
                for item in call_data:  # type: ignore[reportAny]
                    if isinstance(item, dict) and item.get("type") == "function":  # type: ignore[reportUnknownMemberType]
                        tool_calls.append(item)  # type: ignore[reportUnknownArgumentType]

        # Also check for tool calls directly in text (not in XML)
        # Pattern for tool calls in various formats
        json_pattern = r"\{[^\}]*\"type\":\s*\"function\"[^\}]*\}"
        matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                tool_call_data: Any = json.loads(match.group(0))  # type: ignore[reportAny]
                # Avoid duplicates from XML extraction
                if isinstance(tool_call_data, dict) and tool_call_data not in tool_calls:
                    tool_calls.append(tool_call_data)  # type: ignore[reportUnknownArgumentType]
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _extract_json_from_xml_tags(self, text: str) -> list[dict[str, Any] | list[Any]]:
        """
        Extract JSON content from common XML wrapper tags.

        Handles cases like:
        <tool_call>{"type": "function", ...}</tool_call>
        <function_call>[{...}, {...}]</function_call>
        <response>{"data": {...}}</response>
        """
        json_data: list[dict[str, Any] | list[Any]] = []

        # Common XML tags that might contain JSON
        xml_tags = [
            "tool_call",
            "tool_calls",
            "function_call",
            "function_calls",
            "response",
            "result",
            "data",
        ]

        for tag in xml_tags:
            pattern = rf"<{tag}>(.*?)</{tag}>"
            matches = re.finditer(pattern, text, re.DOTALL)

            for match in matches:
                inner_content = match.group(1).strip()

                # Try to parse as JSON
                if inner_content.startswith("{") or inner_content.startswith("["):
                    try:
                        parsed: Any = json.loads(inner_content)  # type: ignore[reportAny]
                        if isinstance(parsed, (dict, list)):
                            json_data.append(parsed)  # type: ignore[reportUnknownArgumentType]
                    except json.JSONDecodeError:
                        # Not valid JSON, skip
                        continue

        return json_data

    def _remove_tool_call_text(self, text: str) -> str:
        """Remove tool call JSON from text content, including XML-wrapped calls."""
        # Remove XML-wrapped tool calls first
        xml_tags = [
            "tool_call",
            "tool_calls",
            "function_call",
            "function_calls",
        ]

        cleaned = text
        for tag in xml_tags:
            # Remove complete XML blocks containing JSON
            pattern = rf"<{tag}>.*?</{tag}>"
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

        # Remove JSON objects that look like tool calls (not in XML)
        cleaned = re.sub(
            r'\{[^\}]*"type":\s*"function"[^\}]*\}', "", cleaned, flags=re.DOTALL
        )

        return cleaned.strip()


class StandardOutputParser(BaseOutputParser):
    """Parser for standard text output without special formatting."""

    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """Parse a standard text chunk."""
        return ParsedOutput(content=text)

    def parse_complete(self, text: str) -> ParsedOutput:
        """Parse complete standard output."""
        # Try to extract thinking even for standard models
        thinking, content = self.extract_thinking_tokens(text)
        return ParsedOutput(content=content, thinking=thinking)


class ThinkingOutputParser(BaseOutputParser):
    """Parser for models that output explicit thinking tokens."""

    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """
        Parse a chunk that may contain thinking tokens.

        For streaming, we need to be careful about partial tags.
        """
        # For streaming, we might have incomplete tags
        # Check if we're inside a thinking block
        full_text = accumulated + text

        thinking, content = self.extract_thinking_tokens(full_text)

        # For streaming, only return the delta
        if accumulated:
            # Calculate delta
            prev_thinking, prev_content = self.extract_thinking_tokens(accumulated)
            content_delta = (
                content[len(prev_content) :] if prev_content else content
            ) if content else None
            thinking_delta = (
                thinking[len(prev_thinking) :] if prev_thinking else thinking
            ) if thinking else None

            return ParsedOutput(
                content=content_delta if content_delta else None,
                thinking=thinking_delta if thinking_delta else None,
            )

        return ParsedOutput(content=content, thinking=thinking)

    def parse_complete(self, text: str) -> ParsedOutput:
        """Parse complete output with thinking tokens."""
        thinking, content = self.extract_thinking_tokens(text)
        return ParsedOutput(content=content, thinking=thinking)


class ToolCallOutputParser(BaseOutputParser):
    """Parser for models with native tool call support."""

    def __init__(self, model_id: ModelId):
        super().__init__(model_id)
        self.tool_call_buffer: dict[str, Any] = {}

    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """Parse a chunk that may contain tool call data."""
        # Tool calls are typically detected via tokenizer special tokens
        # This is handled by the model-specific parsers
        return ParsedOutput(content=text)

    def parse_complete(self, text: str) -> ParsedOutput:
        """Parse complete output that may contain tool calls."""
        # Try to detect and parse tool calls in text
        tool_calls = self._detect_tool_calls(text)

        if tool_calls:
            # Remove tool call text from content
            content = self._remove_tool_call_text(text)
            normalized_tool_calls = self.normalize_tool_calls(tool_calls)
            return ParsedOutput(
                content=content if content.strip() else None,
                tool_calls=normalized_tool_calls,
                finish_reason="tool_calls",
            )

        return ParsedOutput(content=text)


def get_output_parser(model_id: ModelId) -> BaseOutputParser:
    """
    Get the appropriate output parser for a model.

    Args:
        model_id: Model identifier

    Returns:
        Appropriate parser instance
    """
    capabilities = get_model_capabilities(model_id)

    # Check if model requires a custom parser
    if capabilities.requires_custom_parser:
        # Import model-specific parser
        model_family = capabilities.model_family
        try:
            if model_family == "deepseek":
                from exo.shared.adapters.parsers.deepseek_parser import (
                    DeepSeekOutputParser,
                )

                return DeepSeekOutputParser(model_id)
            elif model_family == "gpt-oss":
                from exo.shared.adapters.parsers.gpt_oss_parser import (
                    GptOssOutputParser,
                )

                return GptOssOutputParser(model_id)
            elif model_family == "qwen":
                from exo.shared.adapters.parsers.qwen_parser import QwenOutputParser

                return QwenOutputParser(model_id)
            elif model_family == "glm":
                from exo.shared.adapters.parsers.glm_parser import GlmOutputParser

                return GlmOutputParser(model_id)
            elif model_family == "kimi":
                from exo.shared.adapters.parsers.kimi_parser import KimiOutputParser

                return KimiOutputParser(model_id)
        except ImportError as e:
            logger.warning(
                f"Could not import custom parser for {model_family}: {e}. "
                "Falling back to standard parser."
            )

    # Fall back to standard parsers based on capabilities
    if capabilities.supports_thinking:
        return ThinkingOutputParser(model_id)
    elif capabilities.supports_tools:
        return ToolCallOutputParser(model_id)
    else:
        return StandardOutputParser(model_id)


def parse_output_chunk(
    model_id: ModelId, text: str, accumulated: str = ""
) -> ParsedOutput:
    """
    Convenience function to parse an output chunk.

    Args:
        model_id: Model identifier
        text: Current chunk text
        accumulated: Previously accumulated text

    Returns:
        ParsedOutput with extracted fields
    """
    parser = get_output_parser(model_id)
    return parser.parse_chunk(text, accumulated)


def parse_complete_output(model_id: ModelId, text: str) -> ParsedOutput:
    """
    Convenience function to parse complete output.

    Args:
        model_id: Model identifier
        text: Complete output text

    Returns:
        ParsedOutput with all extracted fields
    """
    parser = get_output_parser(model_id)
    return parser.parse_complete(text)
