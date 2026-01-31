"""Tests for output parser, especially XML with JSON content handling."""

import pytest

from exo.shared.adapters.output_parser import (
    BaseOutputParser,
    StandardOutputParser,
    ThinkingOutputParser,
    get_output_parser,
)
from exo.shared.models.model_cards import ModelId


class TestXmlWithJsonContent:
    """Test parsing XML tags containing JSON content."""

    def test_thinking_with_json_object(self):
        """Test extraction of JSON object inside <think> tags."""
        model_id = ModelId("deepseek-ai/DeepSeek-R1")
        parser = get_output_parser(model_id)

        text = """Here is my response.
<think>
{
  "reasoning": "First, I need to analyze the problem...",
  "conclusion": "The answer is 42"
}
</think>
The answer is 42."""

        result = parser.parse_complete(text)

        # Should extract thinking content
        assert result.thinking is not None
        assert "reasoning" in result.thinking or "First, I need to analyze" in result.thinking
        # Should have content without thinking tags
        assert result.content is not None
        assert "<think>" not in result.content
        assert "The answer is 42" in result.content

    def test_thinking_with_malformed_json(self):
        """Test that malformed JSON inside XML doesn't crash."""
        model_id = ModelId("deepseek-ai/DeepSeek-R1")
        parser = get_output_parser(model_id)

        text = """<think>
{
  "reasoning": "incomplete json...
</think>
Result content."""

        # Should not raise exception
        result = parser.parse_complete(text)

        # Should still extract thinking (as raw text since JSON is invalid)
        assert result.thinking is not None
        assert result.content is not None

    def test_thinking_with_plain_text(self):
        """Test that plain text inside XML works normally."""
        model_id = ModelId("deepseek-ai/DeepSeek-R1")
        parser = get_output_parser(model_id)

        text = """<think>
Let me think about this step by step.
First, I need to understand the problem.
Then, I'll formulate a solution.
</think>
Here's the answer."""

        result = parser.parse_complete(text)

        assert result.thinking is not None
        assert "step by step" in result.thinking
        assert result.content == "Here's the answer."

    def test_tool_call_in_xml_wrapper(self):
        """Test extraction of tool calls wrapped in XML tags."""
        model_id = ModelId("qwen/Qwen-3-8B")
        parser = get_output_parser(model_id)

        text = """I'll call a function for you.
<tool_call>
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\\"location\\": \\"San Francisco\\"}"
  }
}
</tool_call>"""

        result = parser.parse_complete(text)

        # Should extract tool call
        assert result.tool_calls is not None
        assert len(result.tool_calls) > 0
        assert result.tool_calls[0].function.name == "get_weather"
        # Content should not have the XML/JSON
        assert "<tool_call>" not in (result.content or "")

    def test_multiple_tool_calls_in_xml_array(self):
        """Test extraction of multiple tool calls in an XML-wrapped JSON array."""
        model_id = ModelId("qwen/Qwen-3-8B")
        parser = get_output_parser(model_id)

        text = """<function_calls>
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\\"location\\": \\"NYC\\"}"
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_time",
      "arguments": "{}"
    }
  }
]
</function_calls>"""

        result = parser.parse_complete(text)

        # Should extract both tool calls
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_mixed_xml_json_and_plain_content(self):
        """Test handling of mixed content with XML, JSON, and plain text."""
        model_id = ModelId("deepseek-ai/DeepSeek-R1")
        parser = get_output_parser(model_id)

        text = """Let me analyze this.
<think>
{
  "step": 1,
  "reasoning": "Breaking down the problem..."
}
</think>
Based on my analysis, here's the solution.
<think>
Further considerations: This approach is optimal.
</think>
Final answer: 42"""

        result = parser.parse_complete(text)

        # Should have both thinking and content
        assert result.thinking is not None
        assert result.content is not None
        # Content should not have thinking tags
        assert "<think>" not in result.content
        # Content should have the non-thinking parts
        assert "Let me analyze" in result.content
        assert "Final answer: 42" in result.content

    def test_empty_xml_tags(self):
        """Test handling of empty XML tags."""
        model_id = ModelId("deepseek-ai/DeepSeek-R1")
        parser = get_output_parser(model_id)

        text = """<think></think>
Content here."""

        result = parser.parse_complete(text)

        # Empty thinking should be None or empty
        assert not result.thinking or result.thinking == ""
        # Content should be preserved
        assert result.content == "Content here."

    def test_nested_json_in_xml(self):
        """Test deeply nested JSON inside XML tags."""
        model_id = ModelId("deepseek-ai/DeepSeek-R1")
        parser = get_output_parser(model_id)

        text = """<think>
{
  "analysis": {
    "level1": {
      "level2": {
        "reasoning": "Deep thought process here"
      }
    }
  },
  "conclusion": "Complex answer"
}
</think>
Result."""

        # Should not crash with deeply nested JSON
        result = parser.parse_complete(text)

        assert result.thinking is not None
        assert result.content is not None


class TestStandardParsing:
    """Test that standard parsing still works correctly."""

    def test_plain_text_no_special_formatting(self):
        """Test parsing of plain text without XML or special tokens."""
        model_id = ModelId("meta-llama/Llama-3.1-8B")
        parser = get_output_parser(model_id)

        text = "This is a simple response without any special formatting."

        result = parser.parse_complete(text)

        assert result.content == text
        assert result.thinking is None
        assert result.tool_calls is None

    def test_json_in_content_not_tool_call(self):
        """Test that JSON in content that's not a tool call is preserved."""
        model_id = ModelId("meta-llama/Llama-3.1-8B")
        parser = get_output_parser(model_id)

        text = """Here's a JSON example:
{
  "name": "example",
  "value": 123
}
This is not a tool call."""

        result = parser.parse_complete(text)

        # Should preserve JSON in content
        assert result.content is not None
        assert '"name": "example"' in result.content
        # Should not detect as tool call (no "type": "function")
        assert result.tool_calls is None or len(result.tool_calls) == 0
