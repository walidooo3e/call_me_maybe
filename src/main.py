import llm_sdk
import json

definitions = llm_sdk.load_json_data("functions_definition.json")

if definitions:
    print(f"Top-level keys: {list(definitions)}")
    print("\nStructure Preview:")
    print(json.dumps(definitions, indent=2)[:500]+"...")

if __name__ == "__main__":
    # 1. Initialize the model 
    # (Defaulting to CPU if you don't have a GPU to ensure it runs immediately)
    print("Initializing model...")
    sdk_model = llm_sdk.Small_LLM_Model()

    # 2. Define a test string
    # We use a prompt where the next word is highly predictable to verify accuracy
    test_prompt = "The capital of France is"
    
    print(f"\nTest Prompt: '{test_prompt}'")

    # 3. Encode the string
    input_tensor = sdk_model.encode(test_prompt)
    input_ids = input_tensor[0].tolist() # Convert tensor to list for the logit method
    print(f"Encoded IDs: {input_ids}")

    # 4. Get Logits for the next token
    logits = sdk_model.get_logits_from_input_ids(input_ids)
    
    # 5. Simple Validation: Find the token with the highest logit
    top_token_id = logits.index(max(logits))
    predicted_word = sdk_model.decode([top_token_id])

    print(f"Logits vector length: {len(logits)}")
    print(f"Top predicted token ID: {top_token_id}")
    print(f"Model's top prediction: '{predicted_word}'")
    
    if "Paris" in predicted_word:
        print("\n✅ Success: The model is producing coherent logits.")
    else:
        print("\n⚠️ Note: Model output received, but prediction varies by model size.")
