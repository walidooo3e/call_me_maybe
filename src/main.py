import llm_sdk
import json

definitions = llm_sdk.load_json_data("functions_definition.json")

if definitions:
    print(f"Top-level keys: {list(definitions)}")
    print("\nStructure Preview:")
    print(json.dumps(definitions, indent=2)[:500]+"...")
