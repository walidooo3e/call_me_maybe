from pydantic import BaseModel
from enum import Enum


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
