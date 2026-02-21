"""
MemoryNode – stores and retrieves a rolling conversation history.

The node maintains a ``memory`` list in context (a list of
``{"role": …, "content": …}`` dicts), trimming it to a configurable window
so that downstream LLM calls always have relevant context without blowing up
the token budget.
"""

from __future__ import annotations

from typing import Any

from flow import Node


class MemoryNode(Node):
    """
    Manages a rolling conversation memory stored in context under ``memory_key``.

    Config keys
    -----------
    mode : str
        ``"write"`` – append the current turn to memory (default).
        ``"read"``  – inject memory summary into a context key for the next LLM call.
        ``"clear"`` – wipe the memory list.
    memory_key : str
        Context key where the memory list lives (default: ``"memory"``).
    user_key : str
        Context key for the user message to record (default: ``"user_input"``).
    assistant_key : str
        Context key for the assistant reply to record (default: ``"response"``).
    summary_key : str
        Context key where the formatted memory string is written in ``read`` mode
        (default: ``"memory_summary"``).
    max_turns : int
        Maximum number of *turns* (user+assistant pairs) to keep (default: ``10``).
    next_edge : str
        Edge to follow (default: ``"default"``).
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        mode: str = self.config.get("mode", "write")
        memory_key: str = self.config.get("memory_key", "memory")
        user_key: str = self.config.get("user_key", "user_input")
        assistant_key: str = self.config.get("assistant_key", "response")
        summary_key: str = self.config.get("summary_key", "memory_summary")
        max_turns: int = int(self.config.get("max_turns", 10))
        next_edge: str = self.config.get("next_edge", "default")

        memory: list[dict[str, str]] = context.get(memory_key, [])

        if mode == "clear":
            return {"next": next_edge, memory_key: []}

        if mode == "write":
            user_msg = context.get(user_key, "")
            assistant_msg = context.get(assistant_key, "")
            if user_msg:
                memory.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                memory.append({"role": "assistant", "content": str(assistant_msg)})
            # Trim to max_turns (each turn = 2 messages)
            max_messages = max_turns * 2
            if len(memory) > max_messages:
                memory = memory[-max_messages:]
            return {"next": next_edge, memory_key: memory}

        if mode == "read":
            lines: list[str] = []
            for msg in memory:
                role = msg.get("role", "user").capitalize()
                lines.append(f"{role}: {msg.get('content', '')}")
            summary = "\n".join(lines)
            return {"next": next_edge, summary_key: summary}

        raise ValueError(
            f"MemoryNode '{self.node_id}': unknown mode '{mode}'. "
            "Use 'write', 'read', or 'clear'."
        )
