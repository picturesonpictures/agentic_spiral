"""
RouterNode – conditional branching based on context values.

The router evaluates a list of rules in order; the first matching rule
determines the next edge.  This enables dynamic flow paths without
hard-coding conditions inside other nodes.
"""

from __future__ import annotations

import operator
import re
from typing import Any

from flow import Node

# Supported comparison operators
_OPS: dict[str, Any] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "contains": lambda a, b: b in a,
    "not_contains": lambda a, b: b not in a,
    "startswith": lambda a, b: str(a).startswith(str(b)),
    "endswith": lambda a, b: str(a).endswith(str(b)),
    "matches": lambda a, b: bool(re.search(str(b), str(a))),
}


class RouterNode(Node):
    """
    Routes the flow based on context values.

    Config keys
    -----------
    rules : list[dict]
        Ordered list of routing rules.  Each rule is::

            {
                "key":      "<context key to inspect>",
                "op":       "<operator name>",   # see _OPS above
                "value":    <comparison value>,
                "edge":     "<edge name to follow if rule matches>"
            }

    default_edge : str
        Edge to follow when no rule matches (default: ``"default"``).

    Example
    -------
    ::

        RouterNode("router", config={
            "rules": [
                {"key": "sentiment", "op": "eq",       "value": "positive", "edge": "positive_path"},
                {"key": "word_count", "op": "gt",      "value": 100,        "edge": "long_response"},
                {"key": "response",   "op": "contains","value": "ERROR",    "edge": "error_handler"},
            ],
            "default_edge": "fallback",
        })
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        rules: list[dict[str, Any]] = self.config.get("rules", [])
        default_edge: str = self.config.get("default_edge", "default")

        for rule in rules:
            key = rule.get("key", "")
            op_name = rule.get("op", "eq")
            compare_value = rule.get("value")
            edge = rule.get("edge", default_edge)

            ctx_value = context.get(key)
            op_fn = _OPS.get(op_name)
            if op_fn is None:
                raise ValueError(
                    f"RouterNode '{self.node_id}': unknown operator '{op_name}'. "
                    f"Available: {list(_OPS.keys())}"
                )

            try:
                if op_fn(ctx_value, compare_value):
                    return {"next": edge}
            except (TypeError, AttributeError):
                # If comparison fails (e.g. None vs str), skip this rule
                continue

        return {"next": default_edge}
