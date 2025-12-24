import asyncio
import numpy as np
from typing import Any, Callable, Generator
from collections import OrderedDict

from tinygrad import Tensor

from exo.shared.types.api import ChatCompletionMessage, FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.tinygrad.constants import TEMPERATURE, TOP_P, MAX_TOKENS
from exo.worker.engines.tinygrad.models.llama import sample_logits
from exo.worker.engines.tinygrad.stateful_model import make_prompt_state
from exo.worker.engines.tinygrad.utils_tinygrad import get_tinygrad_executor
from exo.worker.runner.bootstrap import logger


def apply_chat_template(tokenizer: Any, chat_task_data: ChatCompletionTaskParams) -> str:
    """Apply chat template to messages"""
    messages = chat_task_data.messages
    formatted_messages = []

    for message in messages:
        content = message.content
        if hasattr(content, 'text'):
            content = content.text
        elif isinstance(content, list) and len(content) > 0:
            content = content[0].text

        if content is None and message.thinking is None:
            continue

        formatted_messages.append({
            "role": message.role,
            "content": content or ""
        })

    # Try to use tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in formatted_messages])
        prompt += "\nassistant: "

    return prompt


def warmup_inference(
    model: Any,
    tokenizer: Any,
    sampler: Callable,
) -> int:
    """Warmup the inference engine"""
    logger.info("Starting warmup inference")

    content = "Prompt to warm up the inference engine. Repeat this."
    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=ChatCompletionTaskParams(
            model="",
            messages=[
                ChatCompletionMessage(
                    role="user",
                    content=content,
                )
            ],
        ),
    )

    tokens = tokenizer.encode(warmup_prompt)
    tokens_generated = 0
    max_warmup_tokens = 50

    # Create initial state
    state = make_prompt_state(Tensor(np.array([tokens])), model)

    # Generate warmup tokens
    for _ in range(max_warmup_tokens):
        x = Tensor(np.array([[state.next_token if hasattr(state, 'next_token') else tokens[-1]]]))
        h = model.embed(x)
        out = model.forward(h, start_pos=state.start, cache=state.cache)

        # Sample next token
        logits = out[:, -1, :]
        next_token = sample_logits(Tensor(logits).flatten(), TEMPERATURE, 0, 0.8, TOP_P, 0.0)
        next_token_id = int(next_token.realize().numpy())

        state.start += 1
        state.next_token = next_token_id
        tokens_generated += 1

        # Check for end of sequence
        if next_token_id == tokenizer.eos_token_id:
            break

    logger.info(f"Warmup complete: generated {tokens_generated} tokens")
    return tokens_generated


def tinygrad_generate(
    model: Any,
    tokenizer: Any,
    sampler: Callable,
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """Generate text using tinygrad model"""
    logger.info(f"Starting tinygrad generation with task: {task}")

    # Apply chat template to get prompt
    prompt = apply_chat_template(tokenizer=tokenizer, chat_task_data=task)
    logger.info(f"Prompt: {prompt[:100]}...")

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    logger.info(f"Encoded {len(tokens)} tokens")

    # Create initial state
    x = Tensor(np.array([tokens]))
    state = make_prompt_state(x, model)

    max_tokens = task.max_tokens or MAX_TOKENS
    tokens_generated = 0

    # Generate tokens
    for _ in range(max_tokens):
        # Get next token input
        if hasattr(state, 'next_token'):
            x = Tensor(np.array([[state.next_token]]))
        else:
            x = Tensor(np.array([tokens]))

        h = model.embed(x)
        out = model.forward(h, start_pos=state.start, cache=state.cache)

        # Sample next token
        logits = out[:, -1, :]
        next_token = sample_logits(
            Tensor(logits).flatten(),
            TEMPERATURE,
            0,
            0.8,
            TOP_P,
            0.0
        )
        next_token_id = int(next_token.realize().numpy())

        # Decode token
        text = tokenizer.decode([next_token_id])

        # Update state
        state.start += 1
        state.next_token = next_token_id
        tokens_generated += 1

        # Yield response
        finish_reason = None
        if next_token_id == tokenizer.eos_token_id:
            finish_reason = "stop"
        elif tokens_generated >= max_tokens:
            finish_reason = "length"

        yield GenerationResponse(
            text=text,
            token=next_token_id,
            finish_reason=finish_reason,
        )

        if finish_reason is not None:
            break

    logger.info(f"Generation complete: {tokens_generated} tokens generated")


async def async_tinygrad_generate(
    model: Any,
    tokenizer: Any,
    sampler: Callable,
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """Async wrapper for tinygrad generation"""
    executor = get_tinygrad_executor()
    loop = asyncio.get_running_loop()

    # Run generation in executor
    def gen_wrapper():
        return list(tinygrad_generate(model, tokenizer, sampler, task))

    results = await loop.run_in_executor(executor, gen_wrapper)

    for result in results:
        yield result
