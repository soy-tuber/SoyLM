"""Unified streaming chat interface for Claude and Gemini.

Both providers expose `stream_chat()` which yields plain-text deltas,
and `chat()` which returns a single concatenated string. The router
swallows provider-specific SDK differences so the rest of the app can
treat them identically.
"""
from __future__ import annotations

from typing import Iterable, Iterator

CLAUDE_MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

PROVIDERS = ["claude", "gemini"]


def list_models(provider: str) -> list[str]:
    if provider == "claude":
        return CLAUDE_MODELS
    if provider == "gemini":
        return GEMINI_MODELS
    return []


def stream_chat(
    provider: str,
    model: str,
    api_key: str,
    system: str,
    messages: list[dict],
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Yield text deltas. `messages` is OpenAI-style: [{role, content}, ...]."""
    if not api_key:
        raise ValueError(f"API key for {provider} is not set")

    if provider == "claude":
        yield from _claude_stream(model, api_key, system, messages, max_tokens)
    elif provider == "gemini":
        yield from _gemini_stream(model, api_key, system, messages, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def chat(
    provider: str,
    model: str,
    api_key: str,
    system: str,
    messages: list[dict],
    max_tokens: int = 2048,
) -> str:
    return "".join(stream_chat(provider, model, api_key, system, messages, max_tokens))


# --- Claude ---

def _claude_stream(
    model: str, api_key: str, system: str, messages: list[dict], max_tokens: int
) -> Iterator[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system or "You are a helpful assistant.",
        messages=messages,
    ) as stream:
        for delta in stream.text_stream:
            if delta:
                yield delta


# --- Gemini ---

def _gemini_stream(
    model: str, api_key: str, system: str, messages: list[dict], max_tokens: int
) -> Iterator[str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Convert OpenAI-style messages → Gemini contents (role: user|model).
    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
        )

    config = types.GenerateContentConfig(
        system_instruction=system or None,
        max_output_tokens=max_tokens,
    )

    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=config
    ):
        text = getattr(chunk, "text", None)
        if text:
            yield text
