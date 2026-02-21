"""
TransformNode – pure-Python data transformation.

Transforms are lightweight processing steps that reshape context values without
making any LLM calls.  Common uses: text cleaning, extraction, formatting,
counting, or injecting computed values.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from flow import Node

# -----------------------------------------------------------------------
# Built-in transform functions
# -----------------------------------------------------------------------

def _strip(value: Any, _: Any) -> str:
    return str(value).strip()


def _upper(value: Any, _: Any) -> str:
    return str(value).upper()


def _lower(value: Any, _: Any) -> str:
    return str(value).lower()


def _word_count(value: Any, _: Any) -> int:
    return len(str(value).split())


def _char_count(value: Any, _: Any) -> int:
    return len(str(value))


def _to_json(value: Any, _: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _from_json(value: Any, _: Any) -> Any:
    return json.loads(str(value))


def _extract_first_line(value: Any, _: Any) -> str:
    return str(value).split("\n")[0].strip()


def _truncate(value: Any, params: Any) -> str:
    max_len = int(params) if params else 200
    text = str(value)
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _regex_extract(value: Any, params: Any) -> str:
    """Extract first group from *params* regex applied to *value*."""
    pattern = str(params)
    match = re.search(pattern, str(value))
    return match.group(1) if match and match.lastindex else (match.group(0) if match else "")


def _template(value: Any, params: Any) -> str:
    """Fill *params* template with ``{value}`` placeholder."""
    return str(params).replace("{value}", str(value))


_BUILTINS: dict[str, Callable[[Any, Any], Any]] = {
    "strip": _strip,
    "upper": _upper,
    "lower": _lower,
    "word_count": _word_count,
    "char_count": _char_count,
    "to_json": _to_json,
    "from_json": _from_json,
    "first_line": _extract_first_line,
    "truncate": _truncate,
    "regex_extract": _regex_extract,
    "template": _template,
}


class TransformNode(Node):
    """
    Applies one or more named transforms to context values.

    Config keys
    -----------
    transforms : list[dict]
        Ordered list of transform operations.  Each entry is::

            {
                "input_key":  "<context key to read>",
                "output_key": "<context key to write>",
                "fn":         "<transform name>",   # see _BUILTINS above
                "params":     <optional argument passed to the transform>
            }

        If ``output_key`` is omitted it defaults to ``input_key`` (in-place).

    next_edge : str
        Edge to follow after all transforms (default: ``"default"``).

    Example
    -------
    ::

        TransformNode("prep", config={
            "transforms": [
                {"input_key": "response", "fn": "word_count", "output_key": "word_count"},
                {"input_key": "response", "fn": "truncate",   "output_key": "summary", "params": 120},
            ]
        })
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        transforms: list[dict[str, Any]] = self.config.get("transforms", [])
        next_edge: str = self.config.get("next_edge", "default")

        updates: dict[str, Any] = {}
        for spec in transforms:
            input_key: str = spec["input_key"]
            output_key: str = spec.get("output_key", input_key)
            fn_name: str = spec.get("fn", "strip")
            params: Any = spec.get("params")

            fn = _BUILTINS.get(fn_name)
            if fn is None:
                raise ValueError(
                    f"TransformNode '{self.node_id}': unknown transform '{fn_name}'. "
                    f"Available: {list(_BUILTINS.keys())}"
                )

            input_value = context.get(input_key, updates.get(input_key))
            updates[output_key] = fn(input_value, params)

        result: dict[str, Any] = {"next": next_edge}
        result.update(updates)
        return result
