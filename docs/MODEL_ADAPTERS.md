# Model Adapter System

The model adapter system provides comprehensive support for different LLM model formats while maintaining OpenAI API compatibility. It automatically handles parameter validation, format conversion, and output parsing for various model families.

## Overview

The adapter system consists of three main components:

1. **Model Capability Registry** - Defines what each model family supports
2. **Input Parameter Adapter** - Validates and transforms request parameters
3. **Output Format Parser** - Parses and normalizes model outputs

## Architecture

```
┌─────────────────┐
│  Client Request │
│  (OpenAI format)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Input Parameter Adapter    │
│  - Validate parameters      │
│  - Transform to model format│
│  - Remove unsupported params│
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Chat Template Application  │
│  - Apply model-specific     │
│    chat template            │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Model Generation           │
│  - Run inference            │
│  - Generate tokens/chunks   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Output Format Parser       │
│  - Extract thinking tokens  │
│  - Normalize tool calls     │
│  - Parse special formats    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  API Response               │
│  (OpenAI format with        │
│   thinking/tools extracted) │
└─────────────────────────────┘
```

## Supported Models and Features

### DeepSeek (deepseek-*)
- **Tool Calls**: ✅ OpenAI format
- **Thinking**: ✅ `<think>` tags
- **Streaming**: ✅
- **Parallel Tools**: ✅
- **Custom Parser**: Yes

### Kimi (kimi-k2-*)
- **Tool Calls**: ✅ OpenAI format
- **Thinking**: ✅ `<thinking>` tags (Thinking variant)
- **Streaming**: ✅
- **Parallel Tools**: ✅
- **Custom Parser**: Yes

### Llama (llama-3.*)
- **Tool Calls**: ✅ OpenAI format
- **Thinking**: ❌
- **Streaming**: ✅
- **Parallel Tools**: ✅
- **Custom Parser**: No

### Qwen3 Standard (qwen3-*)
- **Tool Calls**: ✅ Dict format (auto-converted)
- **Thinking**: ❌
- **Streaming**: ✅
- **Parallel Tools**: ✅
- **Custom Parser**: Yes

### Qwen3 Coder (qwen3-coder-*)
- **Tool Calls**: ✅ Dict format (auto-converted)
- **Thinking**: ❌
- **Streaming**: ✅
- **Parallel Tools**: ✅
- **Custom Parser**: Yes
- **Note**: Optimized for code generation

### Qwen3 Thinking (qwen3-*-thinking-*)
- **Tool Calls**: ✅ Dict format (auto-converted)
- **Thinking**: ✅ `<think>` tags
- **Streaming**: ✅
- **Parallel Tools**: ✅
- **Custom Parser**: Yes

### GPT-OSS (gpt-oss-*)
- **Tool Calls**: ✅ OpenAI format
- **Thinking**: ✅ Harmony format with `<think>` tags
- **Streaming**: ✅
- **Parallel Tools**: ❌
- **Custom Parser**: Yes

### GLM (glm-4.*)
- **Tool Calls**: ✅ Dict format (auto-converted)
- **Thinking**: ❌
- **Streaming**: ✅
- **Parallel Tools**: ✅
- **Custom Parser**: Yes

### MiniMax (minimax-*)
- **Tool Calls**: ✅ OpenAI format
- **Thinking**: ❌
- **Streaming**: ✅
- **Parallel Tools**: ✅
- **Custom Parser**: No

## Usage

### Basic Chat Completion

The adapter system is automatically applied to all chat completion requests:

```python
import requests

response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "deepseek-v3.1-4bit",
        "messages": [
            {"role": "user", "content": "Explain quantum computing"}
        ],
        "temperature": 0.7,
        "stream": False
    }
)

result = response.json()
# Response includes:
# - content: Main response text
# - thinking: Extracted reasoning (if model supports it)
# - tool_calls: Normalized tool calls (if used)
```

### Streaming with Thinking Tokens

For models that support thinking (DeepSeek, Qwen-Thinking, GPT-OSS, Kimi-Thinking):

```python
import requests

response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "gpt-oss-120b-MXFP4-Q8",
        "messages": [
            {"role": "user", "content": "Solve: What is 2+2?"}
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        data = json.loads(line[6:])
        delta = data["choices"][0]["delta"]

        # Thinking content (if present)
        if "thinking" in delta:
            print(f"[Thinking]: {delta['thinking']}")

        # Regular content
        if "content" in delta:
            print(f"[Response]: {delta['content']}")
```

### Tool Calling

Tool calls are automatically normalized across all models:

```python
response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "qwen3-80b-a3B-4bit",
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
)

result = response.json()
# tool_calls are in standard OpenAI format regardless of model:
# {
#   "id": "call_123",
#   "type": "function",
#   "function": {
#     "name": "get_weather",
#     "arguments": "{\"location\": \"Tokyo\"}"  # Always JSON string
#   }
# }
```

### Parameter Validation

The adapter automatically validates and adjusts parameters:

```python
response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "llama-3.1-8b",
        "messages": [...],
        "temperature": 3.0,  # Too high
        "top_k": 0,  # Too low
        "parallel_tool_calls": True,  # Check if supported
    }
)

# Adapter will:
# - Clamp temperature to max (2.0)
# - Clamp top_k to min (1)
# - Keep parallel_tool_calls (Llama supports it)
# - Log warnings for adjusted parameters
```

## API Integration

The adapter system is integrated into three key points:

### 1. Request Entry Point (api.py:chat_completions)

```python
async def chat_completions(
    self, payload: ChatCompletionTaskParams
) -> ChatCompletionResponse | StreamingResponse:
    # Resolve model card
    model_card = await resolve_model_card(ModelId(payload.model))
    payload.model = model_card.model_id

    # ADAPTER: Validate and adapt parameters
    payload = adapt_chat_completion_request(payload)

    # Create command and process...
```

### 2. Streaming Response (_generate_chat_stream)

```python
async def _generate_chat_stream(
    self, command_id: CommandId, model: str
) -> AsyncGenerator[str, None]:
    accumulated_text = ""

    async for chunk in self._chat_chunk_stream(command_id):
        # ADAPTER: Parse chunk with model-specific parser
        if isinstance(chunk, TokenChunk):
            parsed = parse_token_chunk_with_adapter(chunk, accumulated_text)
            accumulated_text += chunk.text

            # Create message with parsed content (thinking, tool_calls, etc.)
            delta_message = ChatCompletionMessage(
                role="assistant",
                content=parsed.content,
                thinking=parsed.thinking,
                tool_calls=parsed.tool_calls,
            )
```

### 3. Non-Streaming Response (_collect_chat_completion)

```python
async def _collect_chat_completion(
    self, command_id: CommandId, model: str
) -> ChatCompletionResponse:
    # Collect all chunks...
    combined_text = "".join(text_parts)

    # ADAPTER: Enrich message with parsed thinking/tools
    message = ChatCompletionMessage(
        role="assistant",
        content=combined_text,
        tool_calls=tool_calls,
    )
    message = enrich_chat_message_with_parsing(
        message, ModelId(model_id), combined_text
    )
```

## Extending the System

### Adding a New Model Family

1. **Define Capabilities** (src/exo/shared/models/model_capabilities.py):

```python
MODEL_CAPABILITIES["new-model"] = ModelCapabilities(
    model_family="new-model",
    supports_tools=True,
    supports_thinking=True,
    thinking_token_type=ThinkingTokenType.XML_TAGS,
    tool_call_format=ToolCallFormat.OPENAI,
    special_tokens=SpecialTokens(
        thinking_start="<reason>",
        thinking_end="</reason>",
    ),
    requires_custom_parser=True,
)
```

2. **Create Custom Parser** (src/exo/shared/adapters/parsers/new_model_parser.py):

```python
from exo.shared.adapters.output_parser import ThinkingOutputParser, ParsedOutput

class NewModelOutputParser(ThinkingOutputParser):
    def parse_chunk(self, text: str, accumulated: str = "") -> ParsedOutput:
        # Implement streaming chunk parsing
        pass

    def parse_complete(self, text: str) -> ParsedOutput:
        # Implement complete output parsing
        thinking, content = self.extract_thinking_tokens(text)
        tool_calls = self._detect_tool_calls(content or "")

        return ParsedOutput(
            content=content,
            thinking=thinking,
            tool_calls=self.normalize_tool_calls(tool_calls) if tool_calls else None,
        )
```

3. **Register Parser** (src/exo/shared/adapters/output_parser.py):

```python
def get_output_parser(model_id: ModelId) -> BaseOutputParser:
    if model_family == "new-model":
        from exo.shared.adapters.parsers.new_model_parser import NewModelOutputParser
        return NewModelOutputParser(model_id)
```

## Testing

### Test Parameter Adaptation

```python
from exo.shared.adapters.input_adapter import adapt_parameters
from exo.shared.models.model_cards import ModelId
from exo.shared.types.tasks import ChatCompletionTaskParams

params = ChatCompletionTaskParams(
    model="deepseek-v3.1-4bit",
    messages=[{"role": "user", "content": "test"}],
    temperature=3.0,  # Will be clamped
)

adapted = adapt_parameters(ModelId("deepseek-v3.1-4bit"), params)
assert adapted.temperature <= 2.0  # Clamped to max
```

### Test Output Parsing

```python
from exo.shared.adapters.output_parser import parse_complete_output
from exo.shared.models.model_cards import ModelId

output_text = "<think>Let me analyze this...</think>The answer is 42."
parsed = parse_complete_output(
    ModelId("deepseek-v3.1-4bit"),
    output_text
)

assert parsed.thinking == "Let me analyze this..."
assert parsed.content == "The answer is 42."
```

## Troubleshooting

### Issue: Thinking tokens not extracted

**Check:**
1. Model ID matches a supported model family
2. Model capabilities has `supports_thinking=True`
3. Special tokens are configured correctly
4. Custom parser is implemented (if required)

### Issue: Tool calls not working

**Check:**
1. Model supports tools (`supports_tools=True`)
2. Tool call format matches model expectations (OpenAI vs Dict)
3. Custom parser normalizes tool calls to OpenAI format
4. Tool definitions are valid

### Issue: Parameters ignored

**Check:**
1. Parameter is supported by model (check constraints)
2. Value is within valid range (will be clamped otherwise)
3. Check logs for adaptation warnings

## Performance Considerations

- **Streaming**: Parsers operate on deltas to minimize latency
- **Caching**: Capability lookups are fast (dict lookup)
- **Fallback**: Parsing errors fall back to raw output (no failures)
- **Logging**: Warnings only for unsupported features (not verbose)

## Future Enhancements

- [ ] JSON schema validation for tool parameters
- [ ] Multi-modal content parsing (images, audio)
- [ ] Custom stop sequences per model
- [ ] Automatic prompt optimization based on model
- [ ] Fine-grained streaming control (thinking-only mode)
- [ ] Model-specific tokenization hints
