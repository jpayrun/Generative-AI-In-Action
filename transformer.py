
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for the huggingface llm course my whole life!")
print(result)
