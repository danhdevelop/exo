# Qwen Model Variants - Capability Reference

## Overview

Qwen models come in three distinct variants with different capabilities. The adapter system automatically detects and handles each variant appropriately.

## Variant Comparison

| Feature | Qwen3 Standard | Qwen3 Coder | Qwen3 Thinking |
|---------|----------------|-------------|----------------|
| **Model IDs** | `qwen3-0.6b`, `qwen3-30b`, `qwen3-80b-a3B`, `qwen3-235b-a22b` | `qwen3-coder-30b`, `qwen3-coder-480b-a35b` | `qwen3-80b-a3B-thinking` |
| **Primary Use** | General purpose | Code generation | Reasoning tasks |
| **Tool Calling** | ✅ Dict format | ✅ Dict format | ✅ Dict format |
| **Thinking Tokens** | ❌ | ❌ | ✅ `<think>...</think>` |
| **Streaming** | ✅ | ✅ | ✅ |
| **Parallel Tools** | ✅ | ✅ | ✅ |
| **Max Tokens** | 8192 | 8192 | 8192 |

## 1. Qwen3 Standard

**Model IDs:**
- `qwen3-0.6b`, `qwen3-0.6b-8bit`
- `qwen3-30b`, `qwen3-30b-8bit`
- `qwen3-80b-a3B-4bit`, `qwen3-80b-a3B-8bit`
- `qwen3-235b-a22b-4bit`, `qwen3-235b-a22b-8bit`

**Characteristics:**
- General-purpose chat model
- Strong multilingual support
- Tool calling with dict format (auto-converted to JSON)
- No explicit thinking output
- Suitable for most chat applications

**Usage:**
```python
response = client.chat.completions.create(
    model="qwen3-80b-a3B-4bit",
    messages=[{"role": "user", "content": "Translate to French: Hello"}]
)

# Response has content only (no thinking field)
print(response.choices[0].message.content)
```

## 2. Qwen3 Coder

**Model IDs:**
- `qwen3-coder-30b-4bit`, `qwen3-coder-30b-6bit`
- `qwen3-coder-480b-a35b-4bit`, `qwen3-coder-480b-a35b-8bit`

**Characteristics:**
- Optimized for code generation and understanding
- Enhanced code completion accuracy
- Better at following coding conventions
- Tool calling with dict format (auto-converted to JSON)
- No explicit thinking output
- Ideal for programming assistants

**Usage:**
```python
response = client.chat.completions.create(
    model="qwen3-coder-30b-4bit",
    messages=[{
        "role": "user",
        "content": "Write a Python function to sort a list"
    }]
)

# Optimized for code generation
print(response.choices[0].message.content)
```

**Best For:**
- Code completion
- Code explanation
- Code refactoring
- Programming Q&A
- Technical documentation

## 3. Qwen3 Thinking

**Model IDs:**
- `qwen3-80b-a3B-thinking-4bit`
- `qwen3-80b-a3B-thinking-8bit`

**Characteristics:**
- Includes explicit reasoning process
- Outputs `<think>` tags with reasoning
- Better for complex problem-solving
- Tool calling with dict format (auto-converted to JSON)
- Thinking automatically extracted by adapter
- Ideal for tasks requiring step-by-step reasoning

**Usage:**
```python
response = client.chat.completions.create(
    model="qwen3-80b-a3B-thinking-4bit",
    messages=[{
        "role": "user",
        "content": "Solve: If x + 5 = 12, what is x?"
    }]
)

message = response.choices[0].message

# Thinking is automatically extracted!
if message.thinking:
    print(f"Reasoning: {message.thinking}")
    # Output: "Let me solve this step by step: x + 5 = 12, so x = 12 - 5 = 7"

print(f"Answer: {message.content}")
# Output: "x = 7"
```

**Best For:**
- Math problems
- Logical reasoning
- Complex analysis
- Scientific questions
- Multi-step problem solving

## Tool Calling Difference

All Qwen variants use **dict format** for tool call arguments internally, which the adapter automatically converts to OpenAI's JSON string format:

### Internal Format (Qwen)
```python
{
  "function": {
    "name": "get_weather",
    "arguments": {"location": "Paris", "unit": "celsius"}  # Dict
  }
}
```

### Normalized Output (Your Agent Receives)
```python
{
  "function": {
    "name": "get_weather",
    "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"  # JSON string
  }
}
```

This conversion is **automatic** - you always receive standard OpenAI format regardless of Qwen variant!

## Detection Logic

The adapter detects variants using model ID patterns:

```python
model_id_lower = model_id.lower()

if "qwen" in model_id_lower and "coder" in model_id_lower:
    # Qwen Coder variant
    variant = "qwen-coder"
elif "qwen" in model_id_lower and "thinking" in model_id_lower:
    # Qwen Thinking variant
    variant = "qwen-thinking"
elif "qwen" in model_id_lower:
    # Standard Qwen
    variant = "qwen"
```

## Choosing the Right Variant

### Use Qwen3 Standard When:
- General chat and Q&A
- Multilingual tasks
- Content generation (non-code)
- Summarization
- Translation

### Use Qwen3 Coder When:
- Code generation or completion
- Code review and refactoring
- Technical documentation
- API integration
- Algorithm implementation

### Use Qwen3 Thinking When:
- Complex problem-solving
- Mathematical reasoning
- Logical puzzles
- Multi-step analysis
- You want to see the reasoning process

## Example: Same Task, Different Variants

### Task: "Implement bubble sort"

**Qwen3 Standard:**
```python
# Good general explanation + code
# No explicit reasoning shown
```

**Qwen3 Coder:**
```python
# Optimized code with best practices
# Better variable names and structure
# More efficient implementation
# No explicit reasoning shown
```

**Qwen3 Thinking:**
```python
# Thinking: "Bubble sort works by comparing adjacent elements..."
# Content: [Code with detailed explanation]
# Shows reasoning process
```

## Migration Between Variants

Switching between variants requires **no code changes**:

```python
# Same code works for all variants!
def ask_qwen(model: str, question: str):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}]
    )

    message = response.choices[0].message

    return {
        "content": message.content,
        "thinking": getattr(message, "thinking", None)  # Only present in Thinking
    }

# Works with any variant
ask_qwen("qwen3-80b-a3B-4bit", "Hello")
ask_qwen("qwen3-coder-30b-4bit", "Write a function")
ask_qwen("qwen3-80b-a3B-thinking-4bit", "Solve 2+2")
```

## Summary

The adapter system **automatically handles** all three Qwen variants:

✅ **Detection**: Identifies variant from model ID
✅ **Tool Calls**: Converts dict format to JSON strings
✅ **Thinking**: Extracts `<think>` tags (Thinking variant only)
✅ **Streaming**: Works correctly for all variants
✅ **Parameters**: Validates and adapts appropriately

**You don't need to do anything special** - just use the standard OpenAI client with the appropriate model ID!
