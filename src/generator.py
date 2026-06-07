from src.models import FunctionDefinition
import llm_sdk

def select_function(
    prompt: str,
    functions: list[FunctionDefinition],
    model: llm_sdk.Small_LLM_Model
) -> FunctionDefinition:
    general_prompt = "Given the following functions:\n"
    for function in functions:
        general_prompt += f" -{function.name}: {function.description}\n"
    general_prompt += f"User request: '{prompt}'\nFunction to call:\n"
    print(general_prompt)
    
    