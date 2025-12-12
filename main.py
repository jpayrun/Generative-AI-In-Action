
import os

from dotenv import dotenv_values
from openai import OpenAI

def run_model(gpt_model: str):
    config = dotenv_values()

    client = OpenAI(api_key=config["openai-key"])

    try:
        # Request a chat completion from a model (e.g., gpt-4)
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, World!'"}
            ]
        )

        # Print the model's response
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your OPENAI_API_KEY is set correctly and you have a payment method on file.")

def main():
    run_model("gpt-4.1-nano")


if __name__ == "__main__":
    main()
