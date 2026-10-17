import pytest


from unittest.mock import Mock, patch

from hello_ai.main import ask_gemini, ask_ollama

def test_ask_ollama_returns_response():
    prompt = "Capital of Germany?"
    model = "qwen3.5:4b"
    expected_output = "Berlin"

    fake_response = Mock()
    fake_response.json.return_value = {
        "response": expected_output
    }

    with patch(
        "hello_ai.main.requests.post",
        return_value=fake_response,
    ) as mock_post:
        actual_output = ask_ollama(prompt, model)

    print(f"\nTest input: {prompt}")
    print(f"Selected model: {model}")
    print(f"Expected output: {expected_output}")
    print(f"Actual output: {actual_output}")

    assert actual_output == expected_output

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )

def test_ask_ollama_raises_for_http_error():
    fake_response = Mock()
    fake_response.raise_for_status.side_effect = (
        RuntimeError("HTTP error")
    )

    with patch(
        "hello_ai.main.requests.post",
        return_value=fake_response,
    ):
        with pytest.raises(RuntimeError, match="HTTP error"):
            ask_ollama("Hello", "qwen3.5:4b")

def test_ask_gemini_returns_response_text():
    fake_client = Mock()
    fake_client.models.generate_content.return_value.text = "Berlin"

    result = ask_gemini(fake_client, "Capital of Germany?")

    assert result == "Berlin"
    fake_client.models.generate_content.assert_called_once_with(
        model="gemini-3.8-flash",
        contents="Capital of Germany?",
    )
