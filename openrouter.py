"""
openrouter.py – synchronous client for the OpenRouter chat-completions API.

All LLM-backed nodes delegate here so that auth and base-URL logic live in
one place.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
_DEFAULT_TIMEOUT = 120.0


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Get a free key at https://openrouter.ai/keys"
        )
    return key


def chat_completion(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Call the OpenRouter chat-completions endpoint synchronously.

    Parameters
    ----------
    messages:
        OpenAI-compatible message list.
    model:
        OpenRouter model identifier, e.g. ``"openai/gpt-4o-mini"``.
    temperature:
        Sampling temperature (0 – 2).
    max_tokens:
        Maximum tokens in the completion.
    tools:
        Optional list of tool/function definitions for tool-use.
    tool_choice:
        ``"auto"`` / ``"none"`` / a specific function dict.
    extra_headers:
        Additional HTTP headers (e.g. ``X-Title`` for OpenRouter leaderboard).
    timeout:
        Request timeout in seconds.

    Returns
    -------
    dict
        The raw JSON response from OpenRouter.

    Raises
    ------
    httpx.HTTPStatusError
        If the API returns a non-2xx status code.
    """
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            content=json.dumps(body),
        )
        response.raise_for_status()
        return response.json()


def extract_text(response: dict[str, Any]) -> str:
    """Return the first choice's message content as a string."""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected OpenRouter response shape: {response}") from exc


def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tool_calls from the first choice, or an empty list."""
    try:
        return response["choices"][0]["message"].get("tool_calls") or []
    except (KeyError, IndexError):
        return []
