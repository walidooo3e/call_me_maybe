from src.models import FunctionDefinition, Types
from src.constraints import get_valid_tokens_for_function_name, get_valid_tokens_for_boolean, get_valid_tokens_for_numbers, get_valid_tokens_for_string
import llm_sdk

def select_function(
    prompt: str,
    functions: list[FunctionDefinition],
    model: llm_sdk.Small_LLM_Model,
    vocab: dict[str, int]
) -> FunctionDefinition:
    general_prompt = "Given the following functions:\n"
    function_names = []
    for function in functions:
        general_prompt += f" -{function.name}: {function.description}\n"
        function_names.append(function.name)
    general_prompt += f"User request: '{prompt}'\nFunction to call:\n"
    input_ids = model.encode(general_prompt)
    input_ids = input_ids[0].tolist()
    generated = []
    while len(generated) < 50:
        logits = model.get_logits_from_input_ids(input_ids)
        generated_str = model.decode(generated)
        if generated_str in function_names:
            break
        valid_tokens = get_valid_tokens_for_function_name(generated_str, function_names, vocab)
        for i in range(len(logits)):
            if i not in valid_tokens:
                logits[i] = -float('inf')
        next_token = logits.index(max(logits))
        if next_token == 198:
            break
        generated.append(next_token)
        input_ids.append(next_token)
    logits = model.get_logits_from_input_ids(input_ids)
    next_token = logits.index(max(logits))
    result = model.decode(generated)
    return result


def extract_arguments(
    prompt: str,
    function: FunctionDefinition,
    model: llm_sdk.Small_LLM_Model,
    vocab: dict[str, int]
) -> dict[str, any]:
    arguments = {}
    for param_name, param in function.parameters.items():
        param_prompt = f"User request: '{prompt}'\nFunction: {function.name} - {function.description}\nParameter '{param_name}' value:"
        print(param_prompt)
        input_ids = model.encode(param_prompt)
        input_ids = input_ids[0].tolist()
        generated = []
        while len(generated) < 20:
            logits = model.get_logits_from_input_ids(input_ids)
            generated_str = model.decode(generated)
            if param.type == Types.number:
                valid_tokens = get_valid_tokens_for_numbers(vocab)
            elif param.type == Types.string:
                valid_tokens = get_valid_tokens_for_string(vocab)
            elif param.type == Types.boolean:
                valid_tokens = get_valid_tokens_for_boolean(generated_str, vocab)
            for i in range(len(logits)):
                if i not in valid_tokens:
                    logits[i] = -float('inf')
            next_token = logits.index(max(logits))
            if next_token == 198:
                break
            if next_token == 1:  # closing quote
                break
            generated.append(next_token)
            input_ids.append(next_token)
            generated_str = model.decode(generated)
            if param.type == Types.number:
                try:
                    float(generated_str)
                    # check if next best token is still numeric
                    next_valid = get_valid_tokens_for_numbers(vocab)
                    # find best token among valid ones
                    best_next = max(next_valid, key=lambda t: logits[t])
                    next_str = model.decode([best_next])
                    if not next_str.strip().replace('.','').replace('-','').isdigit():
                        break
                except ValueError:
                    pass
        raw = model.decode(generated)
        print(f"Raw tokens: {generated}")
        print(f"Raw string: {repr(raw)}")
        if param.type == Types.number:
                arguments[param_name] = float(model.decode(generated))
        elif param.type == Types.boolean:
            arguments[param_name] = True if model.decode(generated) == "true" else False
        elif param.type == Types.string:
            arguments[param_name] = model.decode(generated)
    return arguments

