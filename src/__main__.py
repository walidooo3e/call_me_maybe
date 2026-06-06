import llm_sdk
import json
import argparse


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

    # Check multi-character tokens
    multi_char = [(s, i) for s, i in vocab.items() if len(s) > 1]
    print(f"Multi-character tokens: {len(multi_char)}")
    print("\nSome examples:")
    for s, i in multi_char[:20]:
        print(f"  {repr(s):>30} → ID {i}")

    # Specifically check if number-like tokens exist
    print("\nNumber-like tokens:")
    for s, i in vocab.items():
        if s.strip().replace('.','').replace('-','').isdigit() and len(s) > 1:
            print(f"  {repr(s):>30} → ID {i}")
        
    # Check what encoding "42" gives us
    result = sdk_model.encode("42")
    print(f"\nHow '42' is encoded: {result}")

if __name__ == "__main__":
    args = parse_args()
    print(args)
