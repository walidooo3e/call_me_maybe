from src.constraints import (
    get_valid_tokens_for_boolean,
    get_valid_tokens_for_function_name,
    get_valid_tokens_for_numbers,
    get_valid_tokens_for_string,
)
from src.models import FunctionDefinition, Types, LLMModel


def select_function(
    prompt: str,
    functions: list[FunctionDefinition],
    model: LLMModel,
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
    model: LLMModel,
    vocab: dict[str, int]
) -> dict[str, float | bool | str]:
    """A generator of the arguments"""
    arguments: dict[str, float | bool | str] = {}
    number_tokens = set(get_valid_tokens_for_numbers(vocab)) | {198}
    string_tokens = set(get_valid_tokens_for_string(vocab))
    
    for param_name, param in function.parameters.items():
        if param.type == Types.string:
            if function.name == "fn_substitute_string_with_regex":
                if param_name == "regex":
                    param_prompt = (
                        "Instruction: Generate the raw regex pattern to fulfill the request. Do not use capturing groups or extra parentheses.\n\n"
                        "Request: Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS\n"
                        "Regex: \"\\d+\"\n\n"
                        "Request: Replace all vowels in 'Programming is fun' with asterisks\n"
                        "Regex: \"[aeiouAEIOU]\"\n\n"
                        "Request: Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'\n"
                        "Regex: \"\\bcat\\b\"\n\n"
                        f"Request: {prompt}\n"
                        "Regex: \""
                    )
                elif param_name == "source_string":
                    param_prompt = (
                        "Instruction: Extract the exact target sentence/string that needs modification.\n\n"
                        "Request: Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS\n"
                        "String: \"Hello 34 I'm 233 years old\"\n\n"
                        "Request: Replace all vowels in 'Programming is fun' with asterisks\n"
                        "String: \"Programming is fun\"\n\n"
                        "Request: Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'\n"
                        "String: \"The cat sat on the mat with another cat\"\n\n"
                        f"Request: {prompt}\n"
                        "String: \""
                    )
                elif param_name == "replacement":
                     param_prompt = (
                        "Instruction: Extract the literal replacement string. Convert descriptions like 'asterisks' to symbols.\n\n"
                        "Request: Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS\n"
                        "Replacement: \"NUMBERS\"\n\n"
                        "Request: Replace all vowels in 'Programming is fun' with asterisks\n"
                        "Replacement: \"*\"\n\n"
                        "Request: Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'\n"
                        "Replacement: \"dog\"\n\n"
                        f"Request: {prompt}\n"
                        "Replacement: \""
                    )
            else:
                param_prompt = (
                    f"Extract only the exact {param_name} "
                    f"from this request: '{prompt}'\n"
                    "in lower case"
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
            
            if next_token == 198:  # Newline tracking
                break
                
            # 1. Append the token first to keep any characters trailing right before the quote
            generated.append(next_token)
            input_ids.append(next_token)
            
            # 2. Check if the closing quote is caught inside our sequence now
            if param.type == Types.string:
                generated_str = model.decode(generated)
                if '"' in generated_str:
                    break
                    
        if param.type == Types.number:
            decoded_val = model.decode(generated).rstrip('.')
            arguments[param_name] = float(decoded_val)
        elif param.type == Types.boolean:
            is_true = model.decode(generated) == "true"
            arguments[param_name] = True if is_true else False
        elif param.type == Types.string:
            # Cleanly grab only the valid sequence before the quote boundary
            raw_decoded = model.decode(generated)
            arguments[param_name] = raw_decoded.split('"')[0].strip()
            
    return arguments