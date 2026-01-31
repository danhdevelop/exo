"""
DeepSeek-specific output parser.

Handles DeepSeek's output format including thinking/reasoning tokens and tool calls.
DeepSeek models (especially DeepSeek-R1 and V3) use <think> tags for reasoning.
"""

import re

from loguru import logger

from exo.shared.adapters.output_parser import ParsedOutput, ThinkingOutputParser
from exo.shared.models.model_cards import ModelId


class DeepSeekOutputParser(ThinkingOutputParser):
    """Parser for DeepSeek model outputs."""

    def __init__(self, model_id: ModelId):
        super().__init__(model_id)
        self.in_thinking_block = False
        self.thinking_buffer = ""
        self.content_buffer = ""

    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """
        Parse DeepSeek output chunk with thinking token handling.

        DeepSeek models may output:
        - Standard text
        - <think>reasoning content</think>
        - Tool calls (in standard OpenAI format)
        """
        # Check for thinking tags
        if "<think>" in text:
            self.in_thinking_block = True
            # Split on opening tag
            parts = text.split("<think>", 1)
            if len(parts) == 2:
                content_before = parts[0]
                remaining = parts[1]

                # Check if closing tag is in this chunk
                if "</think>" in remaining:
                    think_parts = remaining.split("</think>", 1)
                    thinking_content = think_parts[0]
                    content_after = think_parts[1] if len(think_parts) > 1 else ""
                    self.in_thinking_block = False

                    return ParsedOutput(
                        content=content_before + content_after
                        if content_before or content_after
                        else None,
                        thinking=thinking_content if thinking_content.strip() else None,
                    )
                else:
                    # Thinking block continues
                    self.thinking_buffer = remaining
                    return ParsedOutput(
                        content=content_before if content_before.strip() else None
                    )

        elif "</think>" in text and self.in_thinking_block:
            # End of thinking block
            parts = text.split("</think>", 1)
            thinking_end = parts[0]
            content_after = parts[1] if len(parts) > 1 else ""

            full_thinking = self.thinking_buffer + thinking_end
            self.thinking_buffer = ""
            self.in_thinking_block = False

            return ParsedOutput(
                content=content_after if content_after.strip() else None,
                thinking=full_thinking if full_thinking.strip() else None,
            )

        elif self.in_thinking_block:
            # Continue accumulating thinking content
            self.thinking_buffer += text
            return ParsedOutput()  # Empty, still in thinking block

        else:
            # Regular content
            return ParsedOutput(content=text)

    def parse_complete(self, text: str) -> ParsedOutput:
        """Parse complete DeepSeek output."""
        # Extract thinking blocks
        thinking_parts = []
        content_parts = []

        # Find all thinking blocks
        remaining = text
        while "<think>" in remaining:
            before, after = remaining.split("<think>", 1)
            if before.strip():
                content_parts.append(before.strip())

            if "</think>" in after:
                thinking, remaining = after.split("</think>", 1)
                if thinking.strip():
                    thinking_parts.append(thinking.strip())
            else:
                # Unclosed thinking tag
                thinking_parts.append(after.strip())
                remaining = ""
                break

        # Add remaining content
        if remaining.strip():
            content_parts.append(remaining.strip())

        final_thinking = "\n\n".join(thinking_parts) if thinking_parts else None
        final_content = "\n".join(content_parts) if content_parts else None

        # Check for tool calls in content
        tool_calls = self._detect_tool_calls(final_content or "")
        if tool_calls:
            normalized_calls = self.normalize_tool_calls(tool_calls)
            # Remove tool call JSON from content
            if final_content:
                final_content = self._remove_tool_call_text(final_content)

            return ParsedOutput(
                content=final_content if final_content and final_content.strip() else None,
                thinking=final_thinking,
                tool_calls=normalized_calls,
                finish_reason="tool_calls",
            )

        return ParsedOutput(content=final_content, thinking=final_thinking)
