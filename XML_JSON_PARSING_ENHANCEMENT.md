# XML with JSON Content Parsing Enhancement

## Summary

Enhanced the output parser to handle cases where models return XML tags containing JSON content. This is a common scenario where models might output structured reasoning or tool calls wrapped in XML tags.

## Changes Made

### 1. Enhanced `_extract_xml_thinking()` method
- Added `_process_xml_inner_content()` helper to intelligently process content inside XML tags
- If content is valid JSON, it extracts and formats it appropriately
- Handles malformed JSON gracefully by returning it as plain text

### 2. Enhanced `_detect_tool_calls()` method
- Added `_extract_json_from_xml_tags()` helper to extract JSON from common XML wrapper tags
- Supports tags like: `<tool_call>`, `<tool_calls>`, `<function_call>`, `<function_calls>`, `<response>`, `<result>`, `<data>`
- Detects tool calls both in XML-wrapped JSON and plain JSON format
- Avoids duplicate detection

### 3. Enhanced `_remove_tool_call_text()` method
- Removes XML-wrapped tool calls before removing plain JSON tool calls
- Ensures clean content extraction

### 4. Moved helper methods to `BaseOutputParser`
- Moved `_detect_tool_calls()`, `_extract_json_from_xml_tags()`, and `_remove_tool_call_text()` from `ToolCallOutputParser` to `BaseOutputParser`
- This allows all parser types (Standard, Thinking, ToolCall, and custom parsers) to use these methods

## Supported Scenarios

### Example 1: JSON in Thinking Tags
```xml
<think>
{
  "reasoning": "First, I need to analyze...",
  "conclusion": "The answer is 42"
}
</think>
```
The parser extracts and formats the JSON, presenting it as readable thinking content.

### Example 2: Tool Calls in XML Wrapper
```xml
<tool_call>
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"location\": \"SF\"}"
  }
}
</tool_call>
```
The parser extracts the tool call and processes it correctly.

### Example 3: Multiple Tool Calls in XML Array
```xml
<function_calls>
[
  {"type": "function", "function": {...}},
  {"type": "function", "function": {...}}
]
</function_calls>
```
The parser extracts all tool calls from the array.

### Example 4: Mixed Content
```text
Let me think about this.
<think>{"reasoning": "analyzing..."}</think>
Based on my analysis...
<think>Additional thoughts</think>
Final answer.
```
The parser handles both JSON and plain text thinking blocks correctly.

## Error Handling

- **Malformed JSON**: Returns as plain text without crashing
- **Empty XML tags**: Handles gracefully
- **Deeply nested JSON**: Processes correctly
- **Invalid XML**: Falls back to standard parsing

## Testing

Added comprehensive test suite in `src/exo/shared/tests/test_output_parser.py`:
- `test_thinking_with_json_object` - JSON inside thinking tags
- `test_thinking_with_malformed_json` - Graceful handling of invalid JSON
- `test_thinking_with_plain_text` - Plain text still works
- `test_tool_call_in_xml_wrapper` - Tool calls in XML
- `test_multiple_tool_calls_in_xml_array` - Array of tool calls
- `test_mixed_xml_json_and_plain_content` - Mixed scenarios
- `test_empty_xml_tags` - Edge case handling
- `test_nested_json_in_xml` - Deep nesting

All tests pass ✓
Type checking passes ✓

## Backward Compatibility

All changes are backward compatible:
- Existing parsing behavior unchanged
- New functionality only activates when XML with JSON is detected
- No breaking changes to API or existing parsers
