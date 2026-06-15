import argparse


def parse_args() -> argparse.Namespace:
    """A parsing function that parses the required json files"""
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing natural language prompts"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file where results will be written"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model identifier to use for generation"
    )
    return parser.parse_args()
