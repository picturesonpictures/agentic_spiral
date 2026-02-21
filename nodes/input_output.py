"""
InputNode / OutputNode – entry and exit points of a flow.

``InputNode``
    Passes ``user_input`` (or any other keys from context) straight through.
    Its sole job is to be a clearly-labelled starting block.

``OutputNode``
    Collects designated keys from context into ``output``, then ends the flow.
"""

from __future__ import annotations

from typing import Any

from flow import Node


class InputNode(Node):
    """
    Entry point of the flow.

    Config keys
    -----------
    prompt_key : str
        Key in *context* that holds the raw user input (default: ``"user_input"``).
    next_edge : str
        Edge name to follow after passing through (default: ``"default"``).
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        prompt_key: str = self.config.get("prompt_key", "user_input")
        next_edge: str = self.config.get("next_edge", "default")

        # Ensure the prompt key exists
        if prompt_key not in context:
            context[prompt_key] = ""

        return {"next": next_edge}


class OutputNode(Node):
    """
    Terminal node – collects result keys and ends the flow.

    Config keys
    -----------
    collect : list[str]
        Keys to pull from context into ``output`` (default: ``["response"]``).
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        collect: list[str] = self.config.get("collect", ["response"])
        output = {k: context.get(k) for k in collect}
        return {"next": None, "output": output}
