"""
AgentNode – autonomous ReAct-style agent with tool use.

The agent runs a Thought → Action → Observation loop until it either
produces a final answer or exhausts its step budget.  Tools are plain
Python callables registered in the config.

This node uses OpenRouter's tool-calling API so that the model itself
decides when and how to invoke tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from flow import Node
import openrouter

logger = logging.getLogger(__name__)

# Sentinel string that the model returns in its content when it is done
_FINAL_ANSWER_PREFIX = "FINAL ANSWER:"


class AgentNode(Node):
    """
    Autonomous agent that loops until it produces a ``FINAL ANSWER``.

    Config keys
    -----------
    model : str
        OpenRouter model ID (default: ``openai/gpt-4o-mini``).
    system_prompt : str
        System instruction for the agent.  The agent is told to either call a
        tool or reply with ``FINAL ANSWER: <answer>``.
    input_key : str
        Context key with the task/question (default: ``"user_input"``).
    output_key : str
        Context key where the final answer is stored (default: ``"response"``).
    next_edge : str
        Edge to follow on success (default: ``"default"``).
    max_steps : int
        Maximum tool-call iterations before forcing a final answer (default: ``8``).
    temperature : float
        Sampling temperature (default: ``0.2``).
    tools : list[dict]
        OpenAI-style tool definitions::

            [
                {
                    "name": "search",
                    "description": "Search the web for information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            ]

    tool_implementations : dict[str, Callable]
        Mapping from tool name → Python callable.  The callable receives the
        parsed JSON arguments dict as keyword arguments and must return a string.
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        model: str = self.config.get("model", openrouter.DEFAULT_MODEL)
        system_prompt: str = self.config.get(
            "system_prompt",
            (
                "You are an autonomous agent. "
                "Use the available tools to answer the user's question. "
                f"When you have the final answer, reply with exactly: "
                f'"{_FINAL_ANSWER_PREFIX} <your answer>"'
            ),
        )
        input_key: str = self.config.get("input_key", "user_input")
        output_key: str = self.config.get("output_key", "response")
        next_edge: str = self.config.get("next_edge", "default")
        max_steps: int = int(self.config.get("max_steps", 8))
        temperature: float = float(self.config.get("temperature", 0.2))

        tool_defs: list[dict[str, Any]] = self.config.get("tools", [])
        tool_impls: dict[str, Callable[..., str]] = self.config.get(
            "tool_implementations", {}
        )

        task: str = str(context.get(input_key, ""))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        # Format tool defs for OpenRouter (OpenAI-compatible format)
        or_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in tool_defs
        ]

        final_answer: str = ""

        for step in range(max_steps):
            logger.debug("[AgentNode %s] step %d", self.node_id, step + 1)

            raw = openrouter.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                tools=or_tools if or_tools else None,
                tool_choice="auto" if or_tools else None,
            )

            tool_calls = openrouter.extract_tool_calls(raw)
            reply_text = openrouter.extract_text(raw)

            # ----------------------------------------------------------------
            # Case 1: model wants to call one or more tools
            # ----------------------------------------------------------------
            if tool_calls:
                # Add the assistant's tool-call message to history
                messages.append(raw["choices"][0]["message"])

                for tc in tool_calls:
                    fn_name: str = tc["function"]["name"]
                    try:
                        fn_args: dict[str, Any] = json.loads(
                            tc["function"].get("arguments", "{}")
                        )
                    except json.JSONDecodeError:
                        fn_args = {}

                    impl = tool_impls.get(fn_name)
                    if impl is None:
                        observation = f"ERROR: tool '{fn_name}' is not implemented."
                    else:
                        try:
                            observation = str(impl(**fn_args))
                        except Exception as exc:  # noqa: BLE001
                            observation = f"ERROR calling '{fn_name}': {exc}"

                    logger.debug(
                        "[AgentNode %s] tool=%s args=%s obs=%s",
                        self.node_id, fn_name, fn_args, observation[:120],
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": observation,
                        }
                    )
                continue  # next step

            # ----------------------------------------------------------------
            # Case 2: model produced a text reply
            # ----------------------------------------------------------------
            messages.append({"role": "assistant", "content": reply_text})

            if _FINAL_ANSWER_PREFIX in reply_text:
                idx = reply_text.index(_FINAL_ANSWER_PREFIX) + len(_FINAL_ANSWER_PREFIX)
                final_answer = reply_text[idx:].strip()
                logger.debug(
                    "[AgentNode %s] got final answer after %d steps", self.node_id, step + 1
                )
                break

            # Model replied without tool calls and without final answer –
            # treat the whole reply as the answer on the last step.
            if step == max_steps - 1:
                final_answer = reply_text
        else:
            # Exhausted max_steps; use whatever the last reply was.
            if not final_answer:
                last = messages[-1]
                final_answer = str(last.get("content", ""))

        return {"next": next_edge, output_key: final_answer}
