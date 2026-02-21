"""
LLMNode – calls OpenRouter to generate a text completion.

Each ``LLMNode`` has its own system prompt, model, and temperature, making
every LLM block in the flow independently configurable.
"""

from __future__ import annotations

from typing import Any

from flow import Node
import openrouter


class LLMNode(Node):
    """
    Makes a single chat-completion call to OpenRouter.

    Config keys
    -----------
    model : str
        OpenRouter model ID (default: ``openai/gpt-4o-mini``).
    system_prompt : str
        System-role message prepended to every call.
    input_key : str
        Context key containing the user message (default: ``"user_input"``).
    output_key : str
        Context key where the LLM reply is stored (default: ``"response"``).
    next_edge : str
        Edge to follow on success (default: ``"default"``).
    temperature : float
        Sampling temperature (default: ``0.7``).
    max_tokens : int
        Maximum response tokens (default: ``1024``).
    extra_context_keys : list[str]
        Additional context keys whose values are appended to the user turn
        as plain text, useful for injecting memory or prior outputs.
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        model: str = self.config.get("model", openrouter.DEFAULT_MODEL)
        system_prompt: str = self.config.get("system_prompt", "You are a helpful assistant.")
        input_key: str = self.config.get("input_key", "user_input")
        output_key: str = self.config.get("output_key", "response")
        next_edge: str = self.config.get("next_edge", "default")
        temperature: float = float(self.config.get("temperature", 0.7))
        max_tokens: int = int(self.config.get("max_tokens", 1024))
        extra_keys: list[str] = self.config.get("extra_context_keys", [])

        user_text: str = str(context.get(input_key, ""))

        # Optionally prepend extra context (e.g. memory) to the user turn
        for key in extra_keys:
            value = context.get(key)
            if value:
                user_text = f"[{key}]\n{value}\n\n{user_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        raw = openrouter.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reply = openrouter.extract_text(raw)

        return {"next": next_edge, output_key: reply}
