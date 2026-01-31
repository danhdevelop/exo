# Model Adapter System - Implementation Summary

## Problem Solved

Your exo API is OpenAI-compatible, but different LLM models have varying:
- Input parameter support (some don't support all OpenAI parameters)
- Output formats (thinking tokens, reasoning, special response structures)
- Tool call formats (JSON string vs dict, different conventions)
- Special tokens and formatting requirements

This made it difficult for your customized agent to communicate smoothly with the exo API, requiring manual handling of each model's quirks.

## Solution Overview

I've implemented a **comprehensive model adapter system** that:

1. ✅ **Auto-validates and transforms input parameters** to match each model's capabilities
2. ✅ **Auto-parses output responses** to extract thinking tokens, reasoning, and special formats
3. ✅ **Normalizes tool calls** from different formats to standard OpenAI format
4. ✅ **Maintains full OpenAI API compatibility** - your agent doesn't need to change
5. ✅ **Zero breaking changes** - falls back gracefully on errors

## Architecture

### Three Core Components

```
┌──────────────────────────────────────────────────┐
│ 1. Model Capability Registry                     │
│    - Defines what each model supports            │
│    - Parameter constraints                       │
│    - Special tokens configuration                │
└──────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────┐
│ 2. Input Parameter Adapter                       │
│    - Validates parameters against capabilities   │
│    - Transforms parameters to model format       │
│    - Removes/adjusts unsupported parameters      │
└──────────────────────────────────────────────────┘
                        ↓
        [Model executes inference]
                        ↓
┌──────────────────────────────────────────────────┐
│ 3. Output Format Parser                          │
│    - Extracts thinking/reasoning tokens          │
│    - Normalizes tool calls to OpenAI format      │
│    - Parses special response formats             │
└──────────────────────────────────────────────────┘
```

## Files Created

### Core System

```
src/exo/shared/models/
└── model_capabilities.py          # Capability registry for all models

src/exo/shared/adapters/
├── __init__.py                     # Package exports
├── input_adapter.py                # Parameter validation & transformation
├── output_parser.py                # Output parsing & normalization
├── integration.py                  # Integration helpers for API
└── parsers/                        # Model-specific parsers
    ├── __init__.py
    ├── deepseek_parser.py          # DeepSeek reasoning format
    ├── gpt_oss_parser.py           # GPT-OSS Harmony format
    ├── qwen_parser.py              # Qwen thinking + dict tool calls
    ├── glm_parser.py               # GLM/ChatGLM dict format
    └── kimi_parser.py              # Kimi thinking format
```

### Documentation & Examples

```
docs/
└── MODEL_ADAPTERS.md               # Complete documentation

examples/
└── model_adapter_example.py        # Usage examples

ADAPTER_SYSTEM_SUMMARY.md           # This file
```

### API Integration

```
src/exo/master/api.py               # Updated with adapter integration
```

## Supported Models & Features

| Model Family | Tool Calls | Thinking/Reasoning | Custom Parser | Notes |
|--------------|------------|-------------------|---------------|-------|
| **DeepSeek** | ✅ OpenAI | ✅ `<think>` tags | ✅ | V3+ has reasoning |
| **Kimi-K2** | ✅ OpenAI | ✅ `<thinking>` tags | ✅ | Thinking variant |
| **Llama 3.x** | ✅ OpenAI | ❌ | ❌ | Standard format |
| **Qwen3** | ✅ Dict→JSON | ❌ | ✅ | Standard version |
| **Qwen3 Coder** | ✅ Dict→JSON | ❌ | ✅ | Code-optimized |
| **Qwen3 Thinking** | ✅ Dict→JSON | ✅ `<think>` tags | ✅ | Thinking variant |
| **GPT-OSS** | ✅ OpenAI | ✅ Harmony format | ✅ | Structured thinking |
| **GLM 4.x** | ✅ Dict→JSON | ❌ | ✅ | Dict tool args |
| **MiniMax** | ✅ OpenAI | ❌ | ❌ | Standard format |

## How It Works

### 1. Request Flow (Input Adaptation)

```python
# Before (without adapter):
POST /v1/chat/completions
{
  "model": "deepseek-v3.1-4bit",
  "temperature": 3.0,        # Too high for this model!
  "top_k": 0,                # Too low!
  "logprobs": true,          # Not supported!
  "messages": [...]
}

# After (with adapter):
# - Temperature clamped to 2.0 (model max)
# - top_k clamped to 1 (model min)
# - logprobs removed (not supported)
# - Warning logs generated
# ✅ Request succeeds with adjusted parameters
```

### 2. Response Flow (Output Parsing)

```python
# Raw model output:
"<think>Let me calculate: speed = distance/time = 120/2</think>The speed is 60 km/h."

# Automatically parsed to:
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The speed is 60 km/h.",      # Clean content
      "thinking": "Let me calculate: speed = distance/time = 120/2"  # Extracted
    }
  }]
}
```

### 3. Tool Call Normalization

```python
# Qwen/GLM internal format (dict):
{
  "function": {
    "name": "get_weather",
    "arguments": {"location": "Tokyo", "unit": "celsius"}  # Dict
  }
}

# Automatically normalized to OpenAI format:
{
  "function": {
    "name": "get_weather",
    "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"  # JSON string
  }
}
```

## Usage Examples

### Example 1: Basic Usage (No Changes Required!)

Your existing code works automatically:

```python
import requests

# Same code for ANY model!
response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "deepseek-v3.1-4bit",  # or "qwen3-80b", "llama-3.1-8b", etc.
        "messages": [
            {"role": "user", "content": "Explain quantum computing"}
        ]
    }
)

result = response.json()
message = result["choices"][0]["message"]

# NEW: Thinking is automatically extracted (if model supports it)
if message.get("thinking"):
    print(f"Reasoning: {message['thinking']}")

print(f"Answer: {message['content']}")
```

### Example 2: Streaming with Thinking

```python
response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "gpt-oss-120b-MXFP4-Q8",
        "messages": [{"role": "user", "content": "Solve 2+2"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        data = json.loads(line[6:])
        delta = data["choices"][0]["delta"]

        # Thinking streamed separately
        if "thinking" in delta:
            print(f"[Think]: {delta['thinking']}", end="")

        # Response content
        if "content" in delta:
            print(f"[Say]: {delta['content']}", end="")
```

### Example 3: Tool Calls (Auto-Normalized!)

```python
# Works identically for Qwen (dict format), DeepSeek (OpenAI format), etc.
response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "qwen3-80b-a3B-4bit",  # Uses dict internally
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }]
    }
)

# Tool calls always in standard OpenAI format!
tool_calls = response.json()["choices"][0]["message"]["tool_calls"]
for tc in tool_calls:
    args = json.loads(tc["function"]["arguments"])  # Always JSON string
    print(f"Call: {tc['function']['name']}({args})")
```

## Benefits

### For Your Agent

✅ **Single codebase** - Same OpenAI client works with all models
✅ **No model-specific logic** - Adapters handle differences automatically
✅ **Thinking support** - Access to reasoning without parsing tags manually
✅ **Tool calling works** - Normalized format across all models
✅ **Parameter safety** - Invalid params adjusted automatically

### For Development

✅ **Easy to extend** - Add new models by defining capabilities + parser
✅ **Type-safe** - Full Pydantic models with validation
✅ **Observable** - Logs warnings for adjustments and errors
✅ **Fallback-safe** - Errors don't break requests, fall back to raw output
✅ **Zero dependencies** - Uses existing project structure

### For API Compatibility

✅ **OpenAI compliant** - Standard format maintained
✅ **Backward compatible** - Existing clients unaffected
✅ **Extended fields** - `thinking` field added when available
✅ **Streaming support** - Works for both streaming and non-streaming

## Testing Your Setup

### 1. Run the Example Script

```bash
# Make sure exo is running with some models loaded
uv run exo

# In another terminal, run the examples
python examples/model_adapter_example.py
```

### 2. Test Individual Models

```bash
# Test DeepSeek with thinking
curl -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3.1-4bit",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'

# Response will have both "thinking" and "content" fields

# Test Qwen with tools (dict format auto-converted)
curl -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-80b-a3B-4bit",
    "messages": [{"role": "user", "content": "Get weather in London"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
      }
    }]
  }'

# Tool calls will be in standard OpenAI format
```

### 3. Check Logs

The adapter system logs helpful information:

```
INFO: Model deepseek-v3.1-4bit supports thinking format
WARNING: Temperature 3.0 above maximum 2.0. Clamping.
WARNING: Model llama-3.1-8b does not support logprobs. Removing.
DEBUG: Extracted thinking: "Let me analyze..."
```

## Integration with Your Agent

Your customized agent can now use a **single OpenAI-compatible client**:

```python
from openai import OpenAI

# Point to your exo instance
client = OpenAI(base_url="http://localhost:52415/v1", api_key="not-needed")

# Works with ANY exo model!
def chat_with_model(model: str, prompt: str):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    message = response.choices[0].message

    # Access thinking if available
    thinking = getattr(message, "thinking", None)
    if thinking:
        print(f"[Reasoning]: {thinking}")

    print(f"[Response]: {message.content}")

    # Tool calls work uniformly
    if message.tool_calls:
        for tc in message.tool_calls:
            print(f"[Tool]: {tc.function.name}({tc.function.arguments})")

# Use with any model
chat_with_model("deepseek-v3.1-4bit", "Explain AI")
chat_with_model("qwen3-80b-a3B-4bit", "What's the weather?")
chat_with_model("llama-3.1-8b", "Write a poem")
```

## Adding New Models

When new models are added to exo, simply:

1. Define capabilities in `model_capabilities.py`
2. Create parser in `parsers/` (if needed)
3. Done! API automatically uses new adapter

Example for a new "ModelX":

```python
# In model_capabilities.py
MODEL_CAPABILITIES["modelx"] = ModelCapabilities(
    model_family="modelx",
    supports_tools=True,
    supports_thinking=True,
    thinking_token_type=ThinkingTokenType.XML_TAGS,
    special_tokens=SpecialTokens(
        thinking_start="<reason>",
        thinking_end="</reason>",
    ),
)

# In parsers/modelx_parser.py (if custom parsing needed)
class ModelXOutputParser(ThinkingOutputParser):
    def parse_complete(self, text: str) -> ParsedOutput:
        thinking, content = self.extract_thinking_tokens(text)
        return ParsedOutput(content=content, thinking=thinking)
```

## Performance Impact

- **Negligible overhead** - Dict lookups and regex parsing are fast
- **Streaming-optimized** - Parsers work on deltas, not full accumulation
- **Fallback on error** - Parsing failures don't slow down responses
- **No external dependencies** - Pure Python with existing libraries

## Monitoring & Debugging

### Enable Debug Logging

```python
# In your code or config
import logging
logging.getLogger("exo.shared.adapters").setLevel(logging.DEBUG)
```

### Check Adapter Behavior

```python
from exo.shared.adapters import (
    get_output_parser,
    adapt_parameters,
)
from exo.shared.models.model_cards import ModelId

# Test parameter adaptation
model_id = ModelId("deepseek-v3.1-4bit")
adapted = adapt_parameters(model_id, your_params)

# Test output parsing
parser = get_output_parser(model_id)
parsed = parser.parse_complete("<think>reasoning</think>answer")
```

## Next Steps

1. ✅ **System is integrated** - No action needed, works automatically
2. 📚 **Read documentation** - See `docs/MODEL_ADAPTERS.md`
3. 🧪 **Run examples** - Try `examples/model_adapter_example.py`
4. 🚀 **Use with your agent** - Standard OpenAI client works!

## Summary

You now have a **production-ready model adapter system** that:

- ✅ Handles all parameter differences automatically
- ✅ Extracts thinking/reasoning tokens transparently
- ✅ Normalizes tool calls from any format
- ✅ Maintains full OpenAI API compatibility
- ✅ Requires zero changes to your agent code

Your customized agent can now communicate with the exo API **smoothly and reliably** across all supported LLM models using the standard OpenAI protocol! 🎉

## Questions?

Refer to:
- **Full documentation**: `docs/MODEL_ADAPTERS.md`
- **Code examples**: `examples/model_adapter_example.py`
- **Model capabilities**: `src/exo/shared/models/model_capabilities.py`
- **Integration code**: `src/exo/shared/adapters/integration.py`
