import tiktoken as tk

def get_encoding(model_name: str) -> tk.Encoding:
    return tk.encoding_for_model(model_name=model_name)

def get_tokens(string: str, encoding_name: str) -> list[int]:

    encoding = tk.get_encoding(encoding_name=encoding_name)

    return encoding.encode(string)

def count_token(string: str, encoding_name: str) -> int:
    return len(get_tokens(string, encoding_name))

def get_string(tokens: list[int], encoding_name: str) -> str:
    encoding = tk.get_encoding(encoding_name)

    return encoding.decode(tokens=tokens)

def main() -> None:
    prompt = "I have a white dog named Champ"

    # cl100k_base is used for gpt-4o
    print("number of tokens ", count_token(prompt, encoding_name="cl100k_base"))

if __name__ == "__main__":
    main()
