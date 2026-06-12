import llm_sdk

def get_valid_tokens_for_function_name(
    generated: str,
    function_names: list[str],
    vocab: dict[str, int]
) -> list[int]:
    remaining = [name for name in function_names if name.startswith(generated)]
    next_chars = set()
    for name in remaining:
        if len(name) > len(generated):
            next_chars.add(name[len(generated)])
    valid_tokens = []
    for token_str, token_id in vocab.items():
        for char in next_chars:
            if token_str.startswith(char):
                valid_tokens.append(token_id)
                break
    return valid_tokens