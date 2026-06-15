import llm_sdk
from src.constraints import (
    get_valid_tokens_for_boolean,
    get_valid_tokens_for_function_name,
    get_valid_tokens_for_numbers,
    get_valid_tokens_for_string,
)
from src.models import FunctionDefinition, Types


def select_function(
    prompt: str,
    functions: list[FunctionDefinition],
    model: llm_sdk.Small_LLM_Model,
    vocab: dict[str, int]
) -> str:
    """A generator of the function name"""
    general_prompt = "Given the following functions:\n"
    function_names = []
    for function in functions:
        general_prompt += f" -{function.name}: {function.description}\n"
        function_names.append(function.name)
    general_prompt += f"User request: '{prompt}'\nFunction to call:\n"
    input_ids = model.encode(general_prompt)
    input_ids = input_ids[0].tolist()
    generated: list[int] = []
    while len(generated) < 50:
        logits = model.get_logits_from_input_ids(input_ids)
        generated_str = model.decode(generated)
        if generated_str in function_names:
            break
        valid_tokens = get_valid_tokens_for_function_name(
            generated_str, function_names, vocab
        )
        valid_set = set(valid_tokens)
        for i in range(len(logits)):
            if i not in valid_set:
                logits[i] = -float('inf')
        next_token = logits.index(max(logits))
        if next_token == 198:
            break
        generated.append(next_token)
        input_ids.append(next_token)
    result: str = model.decode(generated)
    return result


def extract_arguments(
    prompt: str,
    function: FunctionDefinition,
    model: llm_sdk.Small_LLM_Model,
    vocab: dict[str, int]
) -> dict[str, float | bool | str]:
    """A generator of the arguments"""
    arguments: dict[str, float | bool | str] = {}
    number_tokens = set(get_valid_tokens_for_numbers(vocab)) | {198}
    string_tokens = set(get_valid_tokens_for_string(vocab))
    for param_name, param in function.parameters.items():
        if param.type == Types.string:
            param_prompt = (
                f"Extract only the {param_name} "
                f"from this request: '{prompt}'\n"
                f"The {param_name} is: \""
            )
        else:
            if arguments:
                already = "\n".join(f"{k}={v}" for k, v in arguments.items())
                param_prompt = (
                    f"From: '{prompt}'\n"
                    f"{already}\n"
                    f"{param_name}="
                )
            else:
                param_prompt = (
                    f"From: '{prompt}'\n"
                    f"The value of {param_name} is:\n"
                    f"{param_name}="
                )
        input_ids = model.encode(param_prompt)
        input_ids = input_ids[0].tolist()
        generated: list[int] = []
        while len(generated) < 20:
            logits = model.get_logits_from_input_ids(input_ids)
            generated_str = model.decode(generated)
            if param.type == Types.number:
                valid_set = number_tokens
            elif param.type == Types.string:
                valid_set = string_tokens
            elif param.type == Types.boolean:
                valid_tokens = get_valid_tokens_for_boolean(
                    generated_str, vocab
                )
                valid_set = set(valid_tokens)
            for i in range(len(logits)):
                if i not in valid_set:
                    logits[i] = -float('inf')
            next_token = logits.index(max(logits))
            if next_token == 198:
                break
            if param.type == Types.string and next_token in (1, 698):
                break
            generated.append(next_token)
            input_ids.append(next_token)
            generated_str = model.decode(generated)
        if param.type == Types.number:
            decoded_val = model.decode(generated).rstrip('.')
            arguments[param_name] = float(decoded_val)
        elif param.type == Types.boolean:
            is_true = model.decode(generated) == "true"
            arguments[param_name] = True if is_true else False
        elif param.type == Types.string:
            arguments[param_name] = model.decode(generated)
    return arguments
