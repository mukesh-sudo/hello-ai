import os

from dotenv import load_dotenv
from google import genai


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    print("Creating Gemini client...", flush=True)

    client = genai.Client(api_key=api_key)

    print("Sending request to Gemini...", flush=True)

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Explain what an AI in one simple line.",
        )

        print("\nGemini response:")
        print(response.text)

    except Exception as error:
        print(
            f"\nGemini request failed: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()