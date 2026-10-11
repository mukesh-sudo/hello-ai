# hello-ai

A hands-on learning project for building AI applications and agentic AI systems with Python.

The goal is to learn the foundations step by step and gradually progress from basic LLM applications to tool-using and autonomous agents.

## Current Status

### Milestone 1 — Python Foundation + Gemini API

- Python 3.12
- `uv` for Python project and dependency management
- Environment variables with `.env`
- `python-dotenv` for loading environment variables
- Google Gemini API using `google-genai`
- CLI entry point using `uv run hello-ai`
- Git/GitHub workflow

The current application sends a prompt to Gemini and prints the generated response.

## Project Structure

```text
hello-ai/
├── src/
│   └── hello_ai/
│       ├── __init__.py
│       └── main.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
