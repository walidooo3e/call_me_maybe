from pydantic import ValidationError
from src.models import FunctionDefinition
import json

def load_function_definitions(path: str) -> list[FunctionDefinition]:
    try:
        function_list = []
        with open(path, "r") as f:
            file = json.load(f)
        for function in file:
            definition = FunctionDefinition(**function)
            function_list.append(definition)
        return function_list
    except FileNotFoundError:
        print(f"No file found at {path}")
        return []
    except ValidationError:
        print(f"The file at path: {path} is malstructured.")
        return []
    except json.JSONDecodeError:
        print("Please use a path to a valid json.")
        return []

def load_prompts(path: str) -> list[str]:
    try:
        prompt_list = []
        with open(path, "r") as f:
            file = json.load(f)
        for prompt in file:
            prompt_list.append(prompt["prompt"])
        return prompt_list
    except FileNotFoundError:
        print(f"No file found at {path}")
        return []
    except json.JSONDecodeError:
        print("Please use a path to a valid json.")
        return []