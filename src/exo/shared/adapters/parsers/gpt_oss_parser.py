"""
GPT-OSS specific output parser.

Handles GPT-OSS's Harmony format with thinking field and special token handling.
GPT-OSS uses the o200k_harmony tokenizer with <think> tags and structured thinking field.
"""

import json
import re

from loguru import logger

from exo.shared.adapters.output_parser import ParsedOutput, ThinkingOutputParser
from exo.shared.models.model_cards import ModelId


class GptOssOutputParser(ThinkingOutputParser):
    """Parser for GPT-OSS Harmony format outputs."""

    def __init__(self, model_id: ModelId):
        super().__init__(model_id)
        self.thinking_extracted = False

    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """
        Parse GPT-OSS chunk with Harmony format support.

        GPT-OSS may output:
        - Thinking in <think> tags
        - Regular assistant response
        - Structured thinking field (already separated by tokenizer)
        """
        # For streaming, handle <think> tags similar to DeepSeek
        if "<think>" in text or "</think>" in text:
            return self._parse_thinking_chunk(text, accumulated)

        # Regular content chunk
        return ParsedOutput(content=text)

    def _parse_thinking_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """Parse chunk containing thinking tags."""
        full_text = accumulated + text

        # Extract thinking and content
        thinking, content = self.extract_thinking_tokens(full_text)

        # For streaming, calculate delta
        if accumulated:
            prev_thinking, prev_content = self.extract_thinking_tokens(accumulated)

            content_delta = None
            if content and prev_content:
                if len(content) > len(prev_content):
                    content_delta = content[len(prev_content) :]
            elif content:
                content_delta = content

            thinking_delta = None
            if thinking and prev_thinking:
                if len(thinking) > len(prev_thinking):
                    thinking_delta = thinking[len(prev_thinking) :]
            elif thinking:
                thinking_delta = thinking

            return ParsedOutput(content=content_delta, thinking=thinking_delta)

        return ParsedOutput(content=content, thinking=thinking)

    def parse_complete(self, text: str) -> ParsedOutput:
        """Parse complete GPT-OSS output."""
        # Extract thinking from <think> tags
        thinking, content = self.extract_thinking_tokens(text)

        # GPT-OSS may also structure thinking as a separate field
        # This is typically handled by the tokenizer's chat template

        # Check for tool calls
        tool_calls = self._detect_tool_calls(content or "")
        if tool_calls:
            normalized_calls = self.normalize_tool_calls(tool_calls)
            if content:
                content = self._remove_tool_call_text(content)

            return ParsedOutput(
                content=content if content and content.strip() else None,
                thinking=thinking,
                tool_calls=normalized_calls,
                finish_reason="tool_calls",
            )

        return ParsedOutput(content=content, thinking=thinking)

    def _detect_harmony_structure(self, text: str) -> dict[str, str | None]:
        """
        Detect if output is in structured Harmony format.

        Harmony format separates thinking from response:
        {
          "thinking": "...",
          "response": "..."
        }
        """
        try:
            # Try to parse as JSON
            data = json.loads(text)
            if isinstance(data, dict) and ("thinking" in data or "response" in data):
                return {
                    "thinking": data.get("thinking"),
                    "content": data.get("response"),
                }
        except json.JSONDecodeError:
            pass

        return {"thinking": None, "content": text}
