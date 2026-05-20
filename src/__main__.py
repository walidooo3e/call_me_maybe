import llm_sdk
import json

if __name__ == "__main__":
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