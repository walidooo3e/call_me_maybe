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

def get_valid_tokens_for_string(vocab: dict[str, int]) -> list[int]:
    return [token_id for token_str, token_id in vocab.items() if token_id != 198]
def get_valid_tokens_for_numbers(vocab: dict[str, int]) -> list[int]:
    valid_tokens = []
    for token_str, token_id in vocab.items():
        if token_str.strip().replace('.', '').replace('-', '').isdigit() or token_str == '.' or token_str == '-':
            valid_tokens.append(token_id)
    return valid_tokens
def get_valid_tokens_for_boolean(generated: str, vocab: dict[str, int]) -> list[int]:
    allowed = ["true", "false"]
    remaining = [b for b in allowed if b.startswith(generated)]
    next_chars = set()
    valid_tokens = []
    for boolean in remaining:
        if len(boolean) > len(generated):
            next_chars.add(boolean[len(generated)])
    for token_str, token_id in vocab.items():
        for char in next_chars:
            if token_str.startswith(char):
                valid_tokens.append(token_id)
    return valid_tokens

    