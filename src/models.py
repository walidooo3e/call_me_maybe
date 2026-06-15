from pydantic import BaseModel
from enum import Enum
from typing import Protocol, Any


class LLMModel(Protocol):
    """Protocol defining the interface for the LLM SDK model."""
    def get_logits_from_input_ids(
        self, input_ids: list[int]
    ) -> list[float]: ...
    def get_path_to_vocab_file(self) -> str: ...
    def encode(self, text: str) -> Any: ...
    def decode(self, token_ids: list[int]) -> str: ...


class Types(str, Enum):
    """A class of possible argument and return types"""
    number = "number"
    string = "string"
    boolean = "boolean"


class Parameter(BaseModel):
    """A class of Parameters"""
    type: Types


class Returns(BaseModel):
    """A class of Returns"""
    type: Types


class FunctionDefinition(BaseModel):
    """A class of the function definition"""
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Returns
