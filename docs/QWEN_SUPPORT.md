# Qwen Model Support in exo

This document describes the support for Qwen models in exo.

## Supported Qwen Models

exo includes model cards for various Qwen models from the mlx-community, including:
- Qwen3 0.6B, 30B, 80B, 235B (both 4-bit and 8-bit quantizations)
- Qwen3 MoE models (A3B, A22B, A35B)
- Qwen3 Thinking models (with reasoning capabilities)

These models are listed in `src/exo/shared/models/model_cards.py` and can be loaded via MLX-LM.

## Native API Endpoint

exo now provides a Qwen-native API endpoint at:

```
POST /v1/qwen/chat/completions
```

This endpoint accepts the same request format as the OpenAI-compatible endpoint (`ChatCompletionTaskParams`). In the future, this endpoint may support Qwen-specific parameters such as `enable_thinking` and `thinking_budget`.

Currently, the endpoint delegates to the standard chat completion handler.

## Thinking Model Parsing

For Qwen thinking models (e.g., `Qwen3-Next-80B-A3B-Thinking`), exo includes a parser skeleton in `src/exo/worker/runner/runner.py`. The function `parse_qwen_thinking` is currently a pass‑through; it should be extended to detect Qwen's special thinking tokens (`<think>` and `</think>`) and emit appropriate thinking markers in the stream.

When a model is identified as a Qwen thinking model (i.e., an instance of `Qwen3NextModel` or `Qwen3MoeModel`), the runner automatically applies `parse_qwen_thinking`.

## Tool Call Parsing (XML Format)

exo now supports parsing Qwen's XML-formatted tool calls. The parser supports **two different formats** used by different Qwen model variants:

### Format 1: JSON-in-XML (Qwen2.5 / Qwen3-Instruct)

```xml
<tool_call>
{"name": "get_weather", "arguments": {"location": "London, UK"}}
</tool_call>
```

### Format 2: Pure XML (Qwen3-Coder)

```xml
<tool_call>
 <function=calculator>
 <parameter=operation>multiply</parameter>
 <parameter=a>5</parameter>
 <parameter=b>3</parameter>
 </function>
</tool_call>
```

The `parse_qwen_tool_calls` function automatically:
1. Detects and extracts XML tool call tags
2. Attempts to parse as JSON first (Format 1)
3. Falls back to pure XML parsing if JSON parsing fails (Format 2)
4. Converts both formats to OpenAI-compatible tool call format
5. Sets the appropriate `finish_reason` to "tool_calls"
6. Removes the XML tags from the visible response text

The tool calls are returned in the response's `tool_calls` field in OpenAI-compatible format:

```json
{
  "id": "call_0",
  "type": "function",
  "function": {
    "name": "function_name",
    "arguments": "{\"arg1\": \"value1\", \"arg2\": \"value2\"}"
  }
}
```

This parser is automatically applied to all Qwen models (both `Qwen3NextModel` and `Qwen3MoeModel`).

## Implementation Details

### Model Detection
In `runner.py`, after the GPT‑OSS parser check, we added:

```python
elif isinstance(model, (Qwen3NextModel, Qwen3MoeModel)):
    mlx_generator = parse_qwen_thinking(mlx_generator)
```

### Parser Stub
`parse_qwen_thinking` is a generator that currently yields each `GenerationResponse` unchanged. It should be enhanced to:

1. Identify Qwen thinking tokens (likely token IDs for `<think>` and `</think>`).
2. Emit thinking‑start and thinking‑end markers (similar to the GPT‑OSS harmony parser).
3. Strip the thinking tokens from the final output if desired.

### Tool Call Parser
`parse_qwen_tool_calls` is a generator that processes the stream to detect and extract XML-formatted tool calls. It:

1. Detects `<tool_call>` and `</tool_call>` XML tags in the generated text
2. Attempts to parse content as JSON first (Qwen2.5/Qwen3-Instruct format)
3. Falls back to pure XML parsing if JSON fails (Qwen3-Coder format)
4. Converts both formats to OpenAI-compatible tool call format
5. Removes XML tags from the visible output
6. Sets `finish_reason` to "tool_calls" when tool calls are detected
7. Yields `GenerationResponse` objects with the `tool_calls` field populated

The parser uses the helper function `_parse_qwen_xml_tool_call` to handle the pure XML format, which extracts function names from `<function=name>` tags and parameters from `<parameter=name>value</parameter>` tags.

The parser is applied after `parse_qwen_thinking` in the processing pipeline for Qwen models.

### API Route
The new route is registered in `master/api.py`:

```python
self.app.post("/v1/qwen/chat/completions", response_model=None)(self.qwen_chat_completions)
```

The handler `qwen_chat_completions` simply calls the existing `chat_completions` method.

## Future Work

1. **Qwen‑specific request parameters**: Add fields like `enable_thinking`, `thinking_budget`, `reasoning_effort` to `ChatCompletionTaskParams` or create a separate `QwenChatCompletionTaskParams`.

2. **Thinking token detection**: Implement proper detection of Qwen thinking tokens using the tokenizer’s `special_tokens_map`.

3. **DashScope protocol support**: Fully emulate the DashScope native API request/response format (including error codes, response fields, and streaming format).

4. **Enhanced model cards**: Add metadata to distinguish thinking‑capable Qwen models and automatically enable thinking mode when appropriate.

## Testing

Run the existing test suite to ensure no regressions:

```bash
uv run pytest src/exo/worker/tests/unittests/test_runner/
uv run pytest src/exo/master/tests/
```

To test with a real Qwen thinking model, create an instance of a Qwen thinking model and send a chat completion request to the `/v1/qwen/chat/completions` endpoint.