import pytest

from hello_ai.main import main
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

def test_main_defaults_to_qwen(monkeypatch):
    inputs = iter(["", "Capital of Germany?"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with patch(
        "hello_ai.main.ask_ollama",
        return_value="Berlin",
    ) as mock_ollama:
        main()

    mock_ollama.assert_called_once_with(
        "Capital of Germany?",
        "qwen3.5:4b",
    )

def test_main_selects_llama(monkeypatch):
    inputs = iter(["2", "Capital of Germany?"])
    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    with patch(
        "hello_ai.main.ask_ollama",
        return_value="Berlin",
    ) as mock_ollama:
        main()

    mock_ollama.assert_called_once_with(
        "Capital of Germany?",
        "llama3.2:3b",
    )

def test_main_selects_gemini(monkeypatch):
    inputs = iter(["3", "Capital of Germany?"])
    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    with patch(
        "hello_ai.main.ask_gemini",
        return_value="Berlin",
    ) as mock_gemini:
        main()

    mock_gemini.assert_called_once()
    args, _ = mock_gemini.call_args

    assert args[1] == "Capital of Germany?"

def test_main_rejects_invalid_choice(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "4")

    with (
        patch("hello_ai.main.ask_ollama") as mock_ollama,
        patch("hello_ai.main.ask_gemini") as mock_gemini,
    ):
        main()

    mock_ollama.assert_not_called()
    mock_gemini.assert_not_called()


def test_main_rejects_empty_prompt(monkeypatch):
    inputs = iter(["1", ""])
    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    with (
        patch("hello_ai.main.ask_ollama") as mock_ollama,
        patch("hello_ai.main.ask_gemini") as mock_gemini,
    ):
        main()

    mock_ollama.assert_not_called()
    mock_gemini.assert_not_called()