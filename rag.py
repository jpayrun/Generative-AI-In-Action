
import re

def split_sentence(string: str) -> list[str]:
    sentences = re.split('[!?\.]', string)
    # Remove whitespace and ensure sentence is a string
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences
