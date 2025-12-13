import tiktoken as tk

def get_encoding(model_name: str) -> tk.Encoding:
    """
    Return the encoding type used by a gpt model

    Args:
        model_name (str): The model to get the encoding for

    Returns:
        tk.Encoding: The encoding used by the selected model
    """
    return tk.encoding_for_model(model_name=model_name)

def get_tokens(string: str, encoding_name: str) -> list[int]:
    """
    Transform a string into encoding for a gpt model

    Args:
        string (str): The string to transform
        encoding_name (str): The encoding type to transform to

    Returns:
        list[int]: The transformed string
    """
    encoding = tk.get_encoding(encoding_name=encoding_name)

    return encoding.encode(string)

def count_token(string: str, encoding_name: str) -> int:
    """
    Quick function to get the length of an encoded string

    Args:
        string (str): String to encode
        encoding_name (str): The encoding type

    Returns:
        int: The length of the encoded string
    """
    return len(get_tokens(string, encoding_name))

def get_string(tokens: list[int], encoding_name: str) -> str:
    """
    Return the original string after encoding

    Args:
        tokens (list[int]): The token list
        encoding_name (str): The encoding algorithm

    Returns:
        str: The original string
    """
    encoding = tk.get_encoding(encoding_name)

    return encoding.decode(tokens=tokens)

def main() -> None:
    prompt = "I have a white dog named Champ"

    # cl100k_base is used for gpt-4o
    print("number of tokens ", count_token(prompt, encoding_name="cl100k_base"))

if __name__ == "__main__":
    main()
