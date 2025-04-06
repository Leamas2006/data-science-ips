# pylint: disable=missing-module-docstring, missing-function-docstring, invalid-name
from dotenv import load_dotenv

from ai_eng.service.openai_service import OpenAIService

load_dotenv()

# -----------------------------------------------------------------------------
# Define clients
# -----------------------------------------------------------------------------

openai_service = OpenAIService(provider="openai")
ollama_service = OpenAIService(provider="ollama")
groq_service = OpenAIService(provider="groq")

# -----------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------

def get_response_from_groq(user_message: str, model: str) -> str:
    response = groq_service.create_chat_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=1.99
    )
    return response.choices[0].message.content

def get_response_from_ollama(user_message: str) -> str:
    response = ollama_service.create_chat_completion(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    return response.choices[0].message.content

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    message = "Write a poem about PhD candidates learning about AI"
    models = ['llama-3.3-70b-versatile', "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

    for model in models:
        print(f"Response from Groq using model {model}:")
        print(get_response_from_groq(message, model))
        print()

if __name__ == "__main__":
    main()
