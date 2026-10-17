import os

import requests
from dotenv import load_dotenv
from google import genai

def ask_gemini(client, prompt):
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )
    return response.text

def ask_ollama(prompt, model):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()
    return response.json()["response"]

def main():
    load_dotenv()

    print("\nChoose your AI provider:")
    print("1. Qwen 3.5 4B (local, default)")
    print("2. Llama 3.2 3B (local)")
    print("3. Gemini Flash (cloud)")

    choice = input("Enter choice [1]: ").strip() or "1"

    if choice not in ("1", "2", "3"):
        print("Invalid choice. Please select 1, 2, or 3.")
        return

    prompt = input("\nEnter your prompt: ").strip()

    if not prompt:
        print("Prompt cannot be empty.")
        return

    try:
        if choice == "1":
            response_text = ask_ollama(prompt, "qwen3.5:4b")

        elif choice == "2":
            response_text = ask_ollama(prompt, "llama3.2:3b")

        else:
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is missing from .env"
                )

            client = genai.Client(api_key=api_key)
            response_text = ask_gemini(client, prompt)

        print("\nAI response:")
        print(response_text)

    except Exception as error:
        print(
            f"\nRequest failed: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()
