"""
Input parameter adapter for transforming OpenAI API parameters to model-specific formats.

This module handles parameter validation, transformation, and normalization for different
LLM models. It ensures parameters are compatible with the target model's capabilities
and converts them to the appropriate format.
"""

from copy import deepcopy
from typing import Any

from loguru import logger

from exo.shared.models.model_capabilities import (
    OutputFormat,
    ToolCallFormat,
    get_model_capabilities,
    get_parameter_constraints,
)
from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import ChatCompletionTaskParams


class ParameterValidationError(Exception):
    """Raised when parameters are invalid for the target model."""

    pass


class ParameterAdapter:
    """Adapts OpenAI API parameters to model-specific formats."""

    def __init__(self, model_id: ModelId):
        self.model_id = model_id
        self.capabilities = get_model_capabilities(model_id)
        self.constraints = get_parameter_constraints(model_id)

    def validate_and_adapt(
        self, params: ChatCompletionTaskParams
    ) -> ChatCompletionTaskParams:
        """
        Validate and adapt parameters for the target model.

        Args:
            params: Original OpenAI-format parameters

        Returns:
            Adapted parameters suitable for the model

        Raises:
            ParameterValidationError: If parameters are invalid
        """
        # Create a copy to avoid modifying the original
        adapted = deepcopy(params)

        # Validate and adapt each parameter category
        self._validate_tools(adapted)
        self._validate_sampling_params(adapted)
        self._adapt_message_format(adapted)
        self._validate_response_format(adapted)
        self._remove_unsupported_params(adapted)

        return adapted

    def _validate_tools(self, params: ChatCompletionTaskParams) -> None:
        """Validate tool-related parameters."""
        if params.tools is not None and not self.capabilities.supports_tools:
            logger.warning(
                f"Model {self.model_id} does not support tool calling. "
                "Tools will be ignored."
            )
            params.tools = None
            params.tool_choice = None
            params.parallel_tool_calls = None
            return

        if (
            params.parallel_tool_calls
            and not self.capabilities.supports_parallel_tool_calls
        ):
            logger.warning(
                f"Model {self.model_id} does not support parallel tool calls. "
                "Setting parallel_tool_calls=False."
            )
            params.parallel_tool_calls = False

        # Adapt tool format based on model expectations
        if params.tools is not None:
            self._adapt_tool_format(params)

    def _adapt_tool_format(self, params: ChatCompletionTaskParams) -> None:
        """Adapt tool definitions to model-specific format."""
        if self.capabilities.tool_call_format == ToolCallFormat.DICT:
            # Some models (GLM, Qwen) expect arguments as dict in the template
            # This is handled during chat template application, but we can
            # add metadata here if needed
            logger.debug(
                f"Model {self.model_id} uses DICT format for tool calls. "
                "Arguments will be normalized during template application."
            )
        elif self.capabilities.tool_call_format == ToolCallFormat.CUSTOM:
            # Models with custom formats may need special handling
            logger.debug(
                f"Model {self.model_id} uses custom tool call format. "
                "Custom parser will be applied."
            )

    def _validate_sampling_params(self, params: ChatCompletionTaskParams) -> None:
        """Validate and constrain sampling parameters."""
        # Validate temperature
        if params.temperature is not None:
            if params.temperature < self.constraints.temperature_min:
                logger.warning(
                    f"Temperature {params.temperature} below minimum "
                    f"{self.constraints.temperature_min}. Clamping."
                )
                params.temperature = self.constraints.temperature_min
            elif params.temperature > self.constraints.temperature_max:
                logger.warning(
                    f"Temperature {params.temperature} above maximum "
                    f"{self.constraints.temperature_max}. Clamping."
                )
                params.temperature = self.constraints.temperature_max

        # Validate top_p
        if params.top_p is not None:
            if params.top_p < self.constraints.top_p_min:
                logger.warning(
                    f"top_p {params.top_p} below minimum {self.constraints.top_p_min}. Clamping."
                )
                params.top_p = self.constraints.top_p_min
            elif params.top_p > self.constraints.top_p_max:
                logger.warning(
                    f"top_p {params.top_p} above maximum {self.constraints.top_p_max}. Clamping."
                )
                params.top_p = self.constraints.top_p_max

        # Validate top_k
        if params.top_k is not None:
            if params.top_k < self.constraints.top_k_min:
                logger.warning(
                    f"top_k {params.top_k} below minimum {self.constraints.top_k_min}. Clamping."
                )
                params.top_k = self.constraints.top_k_min
            elif (
                self.constraints.top_k_max is not None
                and params.top_k > self.constraints.top_k_max
            ):
                logger.warning(
                    f"top_k {params.top_k} above maximum {self.constraints.top_k_max}. Clamping."
                )
                params.top_k = self.constraints.top_k_max

        # Validate max_tokens
        if (
            params.max_tokens is not None
            and self.constraints.max_tokens_limit is not None
        ):
            if params.max_tokens > self.constraints.max_tokens_limit:
                logger.warning(
                    f"max_tokens {params.max_tokens} exceeds limit "
                    f"{self.constraints.max_tokens_limit}. Clamping."
                )
                params.max_tokens = self.constraints.max_tokens_limit

    def _adapt_message_format(self, params: ChatCompletionTaskParams) -> None:
        """Adapt message format for model-specific requirements."""
        # Check for system message support
        if not self.capabilities.supports_system_message:
            system_messages = [m for m in params.messages if m.role == "system"]
            if system_messages:
                logger.warning(
                    f"Model {self.model_id} does not support system messages. "
                    "System messages will be converted to user messages."
                )
                # Convert system messages to user messages
                for msg in params.messages:
                    if msg.role == "system":
                        msg.role = "user"

        # Handle thinking field in messages (for GPT-OSS compatibility)
        if self.capabilities.supports_thinking:
            for msg in params.messages:
                if hasattr(msg, "thinking") and msg.thinking:
                    logger.debug(
                        f"Message contains thinking field. Model {self.model_id} "
                        "supports thinking format."
                    )

    def _validate_response_format(self, params: ChatCompletionTaskParams) -> None:
        """Validate response format parameters."""
        # Check logprobs support
        if params.logprobs and not self.constraints.supports_logprobs:
            logger.warning(
                f"Model {self.model_id} does not support logprobs. "
                "logprobs will be ignored."
            )
            params.logprobs = None
            params.top_logprobs = None

        # Check response_format (JSON mode)
        if params.response_format is not None:
            if OutputFormat.JSON_MODE not in self.capabilities.output_formats:
                logger.warning(
                    f"Model {self.model_id} may not support JSON mode. "
                    "Response format will be attempted but may not work reliably."
                )

    def _remove_unsupported_params(self, params: ChatCompletionTaskParams) -> None:
        """Remove parameters not supported by the model."""
        # Check frequency_penalty
        if (
            params.frequency_penalty is not None
            and not self.constraints.supports_frequency_penalty
        ):
            logger.debug(
                f"Model {self.model_id} does not support frequency_penalty. Removing."
            )
            params.frequency_penalty = None

        # Check presence_penalty
        if (
            params.presence_penalty is not None
            and not self.constraints.supports_presence_penalty
        ):
            logger.debug(
                f"Model {self.model_id} does not support presence_penalty. Removing."
            )
            params.presence_penalty = None

        # Check logit_bias
        if params.logit_bias is not None and not self.constraints.supports_logit_bias:
            logger.debug(
                f"Model {self.model_id} does not support logit_bias. Removing."
            )
            params.logit_bias = None

        # Check seed
        if params.seed is not None and not self.constraints.supports_seed:
            logger.debug(f"Model {self.model_id} does not support seed. Removing.")
            params.seed = None


def adapt_parameters(
    model_id: ModelId, params: ChatCompletionTaskParams
) -> ChatCompletionTaskParams:
    """
    Convenience function to adapt parameters for a model.

    Args:
        model_id: Target model identifier
        params: OpenAI-format parameters

    Returns:
        Adapted parameters
    """
    adapter = ParameterAdapter(model_id)
    return adapter.validate_and_adapt(params)


def validate_tools_compatible(model_id: ModelId, tools: list[dict[str, Any]]) -> bool:
    """
    Check if a model can handle the provided tool definitions.

    Args:
        model_id: Target model identifier
        tools: Tool definitions in OpenAI format

    Returns:
        True if compatible, False otherwise
    """
    capabilities = get_model_capabilities(model_id)

    if not capabilities.supports_tools:
        return False

    # Additional validation could be added here
    # e.g., checking tool schema complexity, parameter types, etc.

    return True
