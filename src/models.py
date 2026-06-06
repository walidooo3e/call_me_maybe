from pydantic import BaseModel
from enum import Enum

class Types(str, Enum):
    number = "number"
    string = "string"
    boolean = "boolean"

class Parameter(BaseModel):
    type: Types

class Returns(BaseModel):
    type: Types

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Returns