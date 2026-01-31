"""
GLM-specific output parser.

Handles GLM (ChatGLM) output format including:
- Tool calls with dict-format arguments
- Special token handling
- GLM-specific formatting quirks
"""

import json
from typing import Any

from loguru import logger

from exo.shared.adapters.output_parser import ParsedOutput, ToolCallOutputParser
from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import ToolCall, ToolCallItem


class GlmOutputParser(ToolCallOutputParser):
    """Parser for GLM/ChatGLM model outputs."""

    def __init__(self, model_id: ModelId):
        super().__init__(model_id)

    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """Parse GLM output chunk."""
        # GLM typically outputs standard text
        # Tool calls are usually complete in one chunk
        return ParsedOutput(content=text)

    def parse_complete(self, text: str) -> ParsedOutput:
        """Parse complete GLM output."""
        # Check for tool calls
        tool_calls = self._detect_glm_tool_calls(text)

        if tool_calls:
            normalized_calls = self.normalize_glm_tool_calls(tool_calls)
            # Remove tool call JSON from content
            content = self._remove_tool_call_text(text)

            return ParsedOutput(
                content=content if content and content.strip() else None,
                tool_calls=normalized_calls,
                finish_reason="tool_calls",
            )

        return ParsedOutput(content=text)

    def _detect_glm_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """
        Detect tool calls in GLM output.

        GLM formats tool calls in a specific way that may differ from OpenAI.
        """
        import re

        tool_calls = []

        # GLM may use special formatting for tool calls
        # Try to find JSON structures that look like tool calls
        # Pattern for GLM tool call format
        tool_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}'

        matches = re.finditer(tool_pattern, text, re.DOTALL)

        for match in matches:
            try:
                potential_call = json.loads(match.group(0))
                # Verify it's a tool call structure
                if "name" in potential_call and "arguments" in potential_call:
                    tool_calls.append(
                        {
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": potential_call,
                        }
                    )
            except json.JSONDecodeError:
                continue

        return tool_calls

    def normalize_glm_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """
        Normalize GLM tool calls to OpenAI format.

        GLM uses dict format for arguments, need to convert to JSON string.
        """
        normalized = []

        for idx, tc in enumerate(tool_calls):
            tool_id = tc.get("id", f"call_{idx}")
            function_data = tc.get("function", {})

            if isinstance(function_data, dict):
                name = function_data.get("name", "")
                arguments = function_data.get("arguments", {})

                # GLM uses dict, convert to JSON string for OpenAI compatibility
                if isinstance(arguments, dict):
                    arguments_str = json.dumps(arguments, ensure_ascii=False)
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
