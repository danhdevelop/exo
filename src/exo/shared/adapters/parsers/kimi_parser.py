"""
Kimi-specific output parser.

Handles Kimi (Moonshot) output format including:
- Thinking tokens for Kimi-K2-Thinking model
- Tool calls in OpenAI format
- Special token handling
"""

from loguru import logger

from exo.shared.adapters.output_parser import ParsedOutput, ThinkingOutputParser
from exo.shared.models.model_cards import ModelId


class KimiOutputParser(ThinkingOutputParser):
    """Parser for Kimi model outputs."""

    def __init__(self, model_id: ModelId):
        super().__init__(model_id)
        self.is_thinking_model = "thinking" in model_id.lower()
        self.in_thinking_block = False
        self.thinking_buffer = ""

    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        """
        Parse Kimi output chunk.

        Kimi-K2-Thinking uses <thinking>...</thinking> tags.
        """
        if not self.is_thinking_model:
            return ParsedOutput(content=text)

        # Handle thinking tags
        if "<thinking>" in text:
            self.in_thinking_block = True
            parts = text.split("<thinking>", 1)
            content_before = parts[0]
            remaining = parts[1] if len(parts) > 1 else ""

            if "</thinking>" in remaining:
                # Complete thinking block in this chunk
                think_parts = remaining.split("</thinking>", 1)
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
                # Thinking continues
                self.thinking_buffer = remaining
                return ParsedOutput(
                    content=content_before if content_before.strip() else None
                )

        elif "</thinking>" in text and self.in_thinking_block:
            # End of thinking block
            parts = text.split("</thinking>", 1)
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
            # Continue accumulating thinking
            self.thinking_buffer += text
            return ParsedOutput()

        else:
            # Regular content
            return ParsedOutput(content=text)

    def parse_complete(self, text: str) -> ParsedOutput:
        """Parse complete Kimi output."""
        if not self.is_thinking_model:
            # Check for tool calls
            tool_calls = self._detect_tool_calls(text)
            if tool_calls:
                normalized_calls = self.normalize_tool_calls(tool_calls)
                content = self._remove_tool_call_text(text)
                return ParsedOutput(
                    content=content if content.strip() else None,
                    tool_calls=normalized_calls,
                    finish_reason="tool_calls",
                )
            return ParsedOutput(content=text)

        # Extract thinking blocks
        thinking_parts = []
        content_parts = []

        remaining = text
        while "<thinking>" in remaining:
            before, after = remaining.split("<thinking>", 1)
            if before.strip():
                content_parts.append(before.strip())

            if "</thinking>" in after:
                thinking, remaining = after.split("</thinking>", 1)
                if thinking.strip():
                    thinking_parts.append(thinking.strip())
            else:
                # Unclosed tag
                thinking_parts.append(after.strip())
                remaining = ""
                break

        if remaining.strip():
            content_parts.append(remaining.strip())

        final_thinking = "\n\n".join(thinking_parts) if thinking_parts else None
        final_content = "\n".join(content_parts) if content_parts else None

        # Check for tool calls
        if final_content:
            tool_calls = self._detect_tool_calls(final_content)
            if tool_calls:
                normalized_calls = self.normalize_tool_calls(tool_calls)
                final_content = self._remove_tool_call_text(final_content)
                return ParsedOutput(
                    content=final_content if final_content.strip() else None,
                    thinking=final_thinking,
                    tool_calls=normalized_calls,
                    finish_reason="tool_calls",
                )

        return ParsedOutput(content=final_content, thinking=final_thinking)
