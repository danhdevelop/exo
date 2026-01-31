# Model Adapter System - Quick Start

## What Is This?

A **complete solution** for handling different LLM model formats through a unified OpenAI-compatible API. No more manual parsing of thinking tokens, tool call format differences, or parameter incompatibilities!

## The Problem You Had

```python
# Before: Different models required different handling
if model == "deepseek":
    # Parse <think> tags manually
    # Handle specific parameters
elif model == "qwen":
    # Parse tool calls from dict format
    # Convert to JSON strings
elif model == "gpt-oss":
    # Parse Harmony format
    # Extract thinking field
# ... and so on for each model
```

## The Solution Now

```python
# After: One codebase for all models
response = client.chat.completions.create(
    model="ANY_MODEL",  # deepseek, qwen, gpt-oss, llama, etc.
    messages=[{"role": "user", "content": "Hello"}]
)

# Automatically get:
# - Cleaned content
# - Extracted thinking (if model supports it)
# - Normalized tool calls (if used)
message = response.choices[0].message
```

## What Was Implemented

### 1. **Model Capability Registry**
Location: `src/exo/shared/models/model_capabilities.py`

Defines what each model supports:
- Tool calling format (OpenAI vs Dict)
- Thinking tokens (`<think>`, `<thinking>`, etc.)
- Parameter constraints (temp, top_k ranges)
- Special tokens and EOS ids

### 2. **Input Parameter Adapter**
Location: `src/exo/shared/adapters/input_adapter.py`

Automatically:
- Validates parameters against model capabilities
- Clamps out-of-range values (temp, top_k, etc.)
- Removes unsupported parameters (logprobs, etc.)
- Logs warnings for adjustments

### 3. **Output Format Parser**
Location: `src/exo/shared/adapters/output_parser.py`

Automatically:
- Extracts thinking/reasoning from text
- Removes thinking tags from content
- Normalizes tool calls to OpenAI format
- Handles streaming with partial chunks

### 4. **Model-Specific Parsers**
Location: `src/exo/shared/adapters/parsers/*.py`

Custom parsers for:
- `deepseek_parser.py` - `<think>` tag extraction
- `gpt_oss_parser.py` - Harmony format support
- `qwen_parser.py` - Dict→JSON tool call conversion + thinking
- `glm_parser.py` - Dict tool call format
- `kimi_parser.py` - `<thinking>` tag extraction

### 5. **API Integration**
Location: `src/exo/master/api.py` (modified)

Integrated at three key points:
- Request entry: Parameter validation/adaptation
- Streaming: Chunk-by-chunk parsing
- Non-streaming: Complete output parsing

## How to Use

### No Code Changes Required!

Your existing OpenAI client code **just works** with all models:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:52415/v1",
    api_key="not-needed"
)

# Works with ANY model
response = client.chat.completions.create(
    model="deepseek-v3.1-4bit",  # or qwen3-80b, llama-3.1-8b, etc.
    messages=[
        {"role": "user", "content": "Explain quantum entanglement"}
    ]
)

message = response.choices[0].message

# NEW: Thinking automatically extracted!
if hasattr(message, "thinking") and message.thinking:
    print(f"[Reasoning]: {message.thinking}")

print(f"[Answer]: {message.content}")
```

### Streaming with Thinking

```python
stream = client.chat.completions.create(
    model="gpt-oss-120b-MXFP4-Q8",
    messages=[{"role": "user", "content": "Solve 2+2"}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta

    # Thinking streamed separately
    if hasattr(delta, "thinking") and delta.thinking:
        print(f"💭 {delta.thinking}", end="")

    # Content streamed normally
    if delta.content:
        print(delta.content, end="")
```

### Tool Calls (Auto-Normalized!)

```python
# Works identically for all models
response = client.chat.completions.create(
    model="qwen3-80b-a3B-4bit",  # Uses dict format internally
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }]
)

# Always in standard OpenAI format!
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        # arguments is ALWAYS a JSON string (normalized)
        print(f"{tc.function.name}: {tc.function.arguments}")
```

## Supported Models

| Model | Thinking | Tools | Special Parsing |
|-------|----------|-------|-----------------|
| DeepSeek V3+ | ✅ `<think>` | ✅ | ✅ |
| Kimi-K2-Thinking | ✅ `<thinking>` | ✅ | ✅ |
| Llama 3.x | ❌ | ✅ | ❌ |
| Qwen3 | ❌ | ✅ (dict→JSON) | ✅ |
| Qwen3 Coder | ❌ | ✅ (dict→JSON) | ✅ |
| Qwen3 Thinking | ✅ `<think>` | ✅ (dict→JSON) | ✅ |
| GPT-OSS | ✅ Harmony | ✅ | ✅ |
| GLM 4.x | ❌ | ✅ (dict→JSON) | ✅ |
| MiniMax | ❌ | ✅ | ❌ |

## Testing

### 1. Run Example Script

```bash
# Start exo (in one terminal)
uv run exo

# Run examples (in another terminal)
python examples/model_adapter_example.py
```

### 2. Quick cURL Test

```bash
# Test with thinking model
curl -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3.1-4bit",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }' | jq '.'

# Response will have both "thinking" and "content"
```

### 3. Python Test

```python
import requests

response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "deepseek-v3.1-4bit",
        "messages": [{"role": "user", "content": "Explain AI"}]
    }
)

result = response.json()
msg = result["choices"][0]["message"]

print(f"Thinking: {msg.get('thinking', 'N/A')}")
print(f"Content: {msg['content']}")
```

## Key Benefits

✅ **Single API client** - Use standard OpenAI SDK for all models
✅ **Automatic thinking extraction** - No manual tag parsing
✅ **Normalized tool calls** - Works uniformly across models
✅ **Parameter safety** - Invalid params auto-adjusted
✅ **Zero agent changes** - Existing code works as-is
✅ **Streaming support** - Works for both streaming and non-streaming
✅ **Fallback-safe** - Errors don't break requests

## Common Use Cases

### 1. Multi-Model Agent

```python
class MultiModelAgent:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:52415/v1")

    def ask(self, model: str, question: str):
        """Works with ANY model!"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}]
        )

        message = response.choices[0].message

        return {
            "content": message.content,
            "thinking": getattr(message, "thinking", None),
            "model_used": model
        }

# Use with any model
agent = MultiModelAgent()
print(agent.ask("deepseek-v3.1-4bit", "Explain quantum physics"))
print(agent.ask("qwen3-80b-a3B-4bit", "Write a poem"))
print(agent.ask("llama-3.1-8b", "Summarize this article"))
```

### 2. Tool-Using Agent

```python
def weather_agent(location: str):
    """Agent that uses tools - works with any model!"""
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}}
            }
        }
    }]

    response = client.chat.completions.create(
        model="qwen3-80b-a3B-4bit",  # or deepseek, llama, etc.
        messages=[{"role": "user", "content": f"Weather in {location}?"}],
        tools=tools
    )

    # Tool calls automatically normalized!
    if response.choices[0].message.tool_calls:
        return "Tool called successfully"
```

### 3. Thinking-Aware Chain

```python
def reasoning_chain(steps: list[str]):
    """Chain that logs reasoning for each step."""
    results = []

    for step in steps:
        response = client.chat.completions.create(
            model="deepseek-v3.1-4bit",
            messages=[{"role": "user", "content": step}]
        )

        message = response.choices[0].message

        results.append({
            "step": step,
            "thinking": message.thinking,  # Automatically extracted!
            "answer": message.content
        })

    return results
```

## Files to Reference

- 📚 **Full Documentation**: `docs/MODEL_ADAPTERS.md`
- 🚀 **Examples**: `examples/model_adapter_example.py`
- 📝 **Summary**: `ADAPTER_SYSTEM_SUMMARY.md`
- ⚙️ **Capabilities**: `src/exo/shared/models/model_capabilities.py`
- 🔧 **Integration**: `src/exo/shared/adapters/integration.py`

## Troubleshooting

### Thinking not extracted?
- Check model ID matches a supported family
- Verify model has `supports_thinking=True` in capabilities
- Check logs for parsing warnings

### Tool calls not working?
- Verify model supports tools
- Check if format needs custom parser (Qwen, GLM)
- Look for normalization errors in logs

### Parameters ignored?
- Check if parameter is supported by model
- Values may be clamped to valid range
- Check logs for adjustment warnings

## Next Steps

1. ✅ System is ready - No setup required
2. 🧪 Run `examples/model_adapter_example.py`
3. 📖 Read `docs/MODEL_ADAPTERS.md` for details
4. 🚀 Use standard OpenAI client with any model!

---

**You're all set!** Your agent can now communicate smoothly with exo across all supported models using a single, unified OpenAI-compatible interface. 🎉
