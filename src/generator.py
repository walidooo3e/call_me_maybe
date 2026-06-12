from src.models import FunctionDefinition
from src.constraints import get_valid_tokens_for_function_name
import llm_sdk

def select_function(
    prompt: str,
    functions: list[FunctionDefinition],
    model: llm_sdk.Small_LLM_Model,
    vocab
) -> FunctionDefinition:
    general_prompt = "Given the following functions:\n"
    function_names = []
    for function in functions:
        general_prompt += f" -{function.name}: {function.description}\n"
        function_names.append(function.name)
    general_prompt += f"User request: '{prompt}'\nFunction to call:\n"
    print(general_prompt)
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
    print(result)

# def extract_arguments(
#     prompt: str,
#     function: FunctionDefinition,
#     model: llm_sdk.Small_LLM_Model
# ) -> dict[str, any]:
#     general_prompt = f"Given the following user request: {prompt}, what are the arguments that should be given to the function: {function.name}:{function.description}"
#     input_ids = model.encode(general_prompt)
#     input_ids = input_ids[0].tolist()
#     args = {}