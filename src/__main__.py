import llm_sdk
import json
import argparse
from src.loader import load_function_definitions, load_prompts
from src.generator import select_function, extract_arguments

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--functions_definition", 
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions")
    parser.add_argument("--input",
        default="data/input/function_calling_tests.json", 
        help="Path to the JSON file containing natural language prompts")
    parser.add_argument("--output",
        default="data/output/function_calls.json",
        help="Path to the output JSON file where results will be written")
    return parser.parse_args()


def loader():
    sdk_model = llm_sdk.Small_LLM_Model()
    
    with open(sdk_model.get_path_to_vocab_file()) as f:
        vocab = json.load(f)

    for token_str, token_id in vocab.items():
        if token_str in ['Ċ', 'ĊĊ', '\n', '<0x0A>']:
            print(f"{repr(token_str)} → ID {token_id}")

if __name__ == "__main__":
    args = parse_args()
    functions = load_function_definitions(args.functions_definition)
    functions_map = {f.name: f for f in functions}
    prompts = load_prompts(args.input)
    sdk_model = llm_sdk.Small_LLM_Model()
    with open(sdk_model.get_path_to_vocab_file()) as f:
        vocab = json.load(f)
    for prompt in prompts:
        function = select_function(prompt, functions, sdk_model, vocab)
        arguments = extract_arguments(prompt, functions_map[function], sdk_model, vocab)
        print(f"function: {function}, arguments: {arguments}")
        