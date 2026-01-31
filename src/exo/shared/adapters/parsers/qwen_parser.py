"""
Qwen-specific output parser.

Handles Qwen's output format including:
- Tool calls with arguments as dict (not JSON string)
- Thinking tokens for Qwen-Thinking variants
- Special formatting requirements
"""

import json
import re

from loguru import logger

from exo.shared.adapters.output_parser import ParsedOutput, ThinkingOutputParser
from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import ToolCall, ToolCallItem


class QwenOutputParser(ThinkingOutputParser):
    """
    Parser for Qwen model outputs.

    Handles three variants:
    - Standard Qwen3: No thinking, dict tool calls
    - Qwen3 Coder: No thinking, dict tool calls, code-optimized
    - Qwen3 Thinking: Has <think> tags, dict tool calls
    """

    def __init__(self, model_id: ModelId):
        super().__init__(model_id)
        model_id_lower = model_id.lower()
        self.is_thinking_model = "thinking" in model_id_lower
        self.is_coder_model = "coder" in model_id_lower

    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """Parse Qwen output chunk."""
        # Qwen-Thinking models use <think> tags
        if self.is_thinking_model and ("<think>" in text or "</think>" in text):
            return self._parse_thinking_chunk(text, accumulated)

        # Regular content
        return ParsedOutput(content=text)

    def _parse_thinking_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """Parse chunk with thinking tags."""
        full_text = accumulated + text
        thinking, content = self.extract_thinking_tokens(full_text)

        if accumulated:
            prev_thinking, prev_content = self.extract_thinking_tokens(accumulated)

            content_delta = (
                content[len(prev_content) :] if content and prev_content else content
            )
            thinking_delta = (
                thinking[len(prev_thinking) :]
                if thinking and prev_thinking
                else thinking
            )

            return ParsedOutput(
                content=content_delta if content_delta else None,
                thinking=thinking_delta if thinking_delta else None,
            )

        return ParsedOutput(content=content, thinking=thinking)

    def parse_complete(self, text: str) -> ParsedOutput:
        """Parse complete Qwen output."""
        # Extract thinking if present
        thinking = None
        content = text

        if self.is_thinking_model:
            thinking, content = self.extract_thinking_tokens(text)

        # Check for tool calls
        tool_calls = self._detect_qwen_tool_calls(content)
        if tool_calls:
            normalized_calls = self.normalize_qwen_tool_calls(tool_calls)
            # Remove tool call text from content
            content = self._remove_tool_call_text(content)

            return ParsedOutput(
                content=content if content and content.strip() else None,
                thinking=thinking,
                tool_calls=normalized_calls,
                finish_reason="tool_calls",
            )

        return ParsedOutput(content=content, thinking=thinking)

    def _detect_qwen_tool_calls(self, text: str) -> list[dict]:
        """
        Detect tool calls in Qwen output.

        Qwen may format tool calls differently than OpenAI.
        """
        tool_calls = []

        # Try to find function call markers
        # Qwen often uses specific markers like "✿FUNCTION✿" or similar
        function_pattern = r"✿FUNCTION✿:\s*(\{[^}]+\})"
        matches = re.finditer(function_pattern, text)

        for match in matches:
            try:
                tool_data = json.loads(match.group(1))
                tool_calls.append(tool_data)
            except json.JSONDecodeError:
                continue

        # Also try standard JSON format
        if not tool_calls:
            tool_calls = self._detect_tool_calls(text)

        return tool_calls

    def normalize_qwen_tool_calls(self, tool_calls: list[dict]) -> list[ToolCall]:
        """
        Normalize Qwen tool calls to OpenAI format.

        Qwen uses dict format for arguments, need to convert to JSON string.
        """
        normalized = []

        for idx, tc in enumerate(tool_calls):
            tool_id = tc.get("id", f"call_{idx}")
            function_data = tc.get("function", {})

            if isinstance(function_data, dict):
                name = function_data.get("name", "")
                arguments = function_data.get("arguments", {})

                # Qwen uses dict, convert to JSON string for OpenAI compatibility
                if isinstance(arguments, dict):
                    arguments_str = json.dumps(arguments)
                else:
                    arguments_str = str(arguments)

                normalized.append(
                    ToolCall(
                        id=tool_id,
                        index=idx,
                        type="function",
                        function=ToolCallItem(name=name, arguments=arguments_str),
                    )
                )

        return normalized
