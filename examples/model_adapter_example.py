#!/usr/bin/env python3
"""
Example usage of the model adapter system with the exo API.

This demonstrates how the adapter automatically handles different model formats,
thinking tokens, and tool calls while maintaining OpenAI API compatibility.
"""

import json
import time

import requests


def example_basic_completion():
    """Basic chat completion with automatic parameter adaptation."""
    print("\n=== Basic Chat Completion ===\n")

    response = requests.post(
        "http://localhost:52415/v1/chat/completions",
        json={
            "model": "llama-3.1-8b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        },
    )

    result = response.json()
    print(f"Response: {result['choices'][0]['message']['content']}")


def example_thinking_model():
    """Chat completion with a model that supports thinking tokens."""
    print("\n=== Thinking Model (DeepSeek) ===\n")

    response = requests.post(
        "http://localhost:52415/v1/chat/completions",
        json={
            "model": "deepseek-v3.1-4bit",
            "messages": [
                {
                    "role": "user",
                    "content": "Solve this step by step: If a train travels 120km in 2 hours, what's its speed?",
                }
            ],
            "temperature": 0.5,
        },
    )

    result = response.json()
    message = result["choices"][0]["message"]

    # Thinking is automatically extracted and returned as a separate field
    if message.get("thinking"):
        print(f"[Thinking Process]:\n{message['thinking']}\n")

    print(f"[Final Answer]:\n{message['content']}")


def example_streaming_with_thinking():
    """Streaming completion that shows thinking and content separately."""
    print("\n=== Streaming with Thinking ===\n")

    response = requests.post(
        "http://localhost:52415/v1/chat/completions",
        json={
            "model": "gpt-oss-120b-MXFP4-Q8",
            "messages": [
                {"role": "user", "content": "Explain how recursion works in programming."}
            ],
            "stream": True,
        },
        stream=True,
    )

    for line in response.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue

        data_str = line[6:].decode("utf-8")
        if data_str == "[DONE]":
            break

        try:
            data = json.loads(data_str)
            delta = data["choices"][0]["delta"]

            # Thinking content is streamed separately
            if thinking := delta.get("thinking"):
                print(f"[Thinking]: {thinking}", end="", flush=True)

            # Regular content
            if content := delta.get("content"):
                print(f"[Response]: {content}", end="", flush=True)

        except json.JSONDecodeError:
            continue

    print()  # Newline at end


def example_tool_calling():
    """Tool calling with automatic format normalization."""
    print("\n=== Tool Calling ===\n")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = requests.post(
        "http://localhost:52415/v1/chat/completions",
        json={
            "model": "qwen3-80b-a3B-4bit",  # Qwen uses dict format internally
            "messages": [
                {"role": "user", "content": "What's the weather like in Tokyo?"}
            ],
            "tools": tools,
        },
    )

    result = response.json()
    message = result["choices"][0]["message"]

    # Tool calls are automatically normalized to OpenAI format
    # regardless of the model's internal format
    if tool_calls := message.get("tool_calls"):
        for tc in tool_calls:
            func = tc["function"]
            args = json.loads(func["arguments"])  # Always a JSON string
            print(f"Tool: {func['name']}")
            print(f"Arguments: {args}")


def example_parameter_validation():
    """Demonstrates automatic parameter validation and adjustment."""
    print("\n=== Parameter Validation ===\n")

    # Send parameters that may need adjustment
    response = requests.post(
        "http://localhost:52415/v1/chat/completions",
        json={
            "model": "llama-3.1-8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 3.0,  # Too high - will be clamped to 2.0
            "top_k": 0,  # Too low - will be clamped to 1
            "max_tokens": 50000,  # May exceed limit
            "logprobs": True,  # Not supported by all models
        },
    )

    result = response.json()
    print(
        f"Response received successfully (parameters auto-adjusted): {result['choices'][0]['message']['content'][:50]}..."
    )
    print("\nCheck server logs for parameter adjustment warnings.")


def example_multi_model_comparison():
    """Compare responses from different model families with the same interface."""
    print("\n=== Multi-Model Comparison ===\n")

    models = [
        "llama-3.1-8b",
        "qwen3-30b",
        "deepseek-coder-v2-lite-instruct-mlx-6bit",
    ]

    prompt = "Write a Python function to calculate fibonacci numbers."

    for model in models:
        print(f"\n--- {model} ---")
        try:
            response = requests.post(
                "http://localhost:52415/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                },
                timeout=30,
            )

            result = response.json()
            message = result["choices"][0]["message"]

            # All responses have the same structure regardless of model
            if message.get("thinking"):
                print(f"[Thinking]: {message['thinking'][:50]}...")

            print(f"[Response]: {message['content'][:100]}...")

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        except KeyError:
            print(f"Model {model} may not be available")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Model Adapter System Examples")
    print("=" * 60)

    examples = [
        ("Basic Completion", example_basic_completion),
        ("Thinking Model", example_thinking_model),
        ("Streaming with Thinking", example_streaming_with_thinking),
        ("Tool Calling", example_tool_calling),
        ("Parameter Validation", example_parameter_validation),
        ("Multi-Model Comparison", example_multi_model_comparison),
    ]

    for name, func in examples:
        try:
            func()
            time.sleep(1)  # Brief pause between examples
        except Exception as e:
            print(f"\n❌ Example '{name}' failed: {e}")
            print("Make sure exo is running with the appropriate models loaded.")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
