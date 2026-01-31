"""
Model adapters for parameter validation and output parsing.

This package provides comprehensive support for different LLM model formats,
ensuring smooth interoperability through the OpenAI API protocol.
"""

from exo.shared.adapters.input_adapter import (
    ParameterAdapter,
    ParameterValidationError,
    adapt_parameters,
    validate_tools_compatible,
)
from exo.shared.adapters.output_parser import (
    BaseOutputParser,
    ParsedOutput,
    get_output_parser,
    parse_complete_output,
    parse_output_chunk,
)

__all__ = [
    "ParameterAdapter",
    "ParameterValidationError",
    "adapt_parameters",
    "validate_tools_compatible",
    "BaseOutputParser",
    "ParsedOutput",
    "get_output_parser",
    "parse_output_chunk",
    "parse_complete_output",
]
