"""
Core flow engine for agentic_spiral.

Each Flow is a directed graph of Nodes. Nodes are connected via named edges.
When a node finishes, it emits a result dict containing a ``next`` key that
names the edge to follow, plus any payload the downstream node needs.

Design goals
------------
* **Unique blocks** – every Node subclass lives in ``nodes/`` and has its own
  purpose (LLM call, routing, memory, transform, agent loop …).
* **Dynamic setup** – flows are assembled at runtime from plain Python dicts or
  direct Python calls; no YAML/JSON required but easy to add.
* **OpenRouter** – all LLM calls are routed through
  https://openrouter.ai/api/v1/chat/completions.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FlowError(Exception):
    """Raised when the flow encounters an unrecoverable error."""


class Node:
    """
    Base class for every node in a flow.

    Subclasses must implement :meth:`run`.

    Parameters
    ----------
    node_id:
        Unique identifier for this node within a flow.
    config:
        Arbitrary per-node configuration dict passed at construction time.
    """

    def __init__(self, node_id: str, config: dict[str, Any] | None = None) -> None:
        self.node_id = node_id
        self.config: dict[str, Any] = config or {}

    # ------------------------------------------------------------------
    # Subclasses implement this
    # ------------------------------------------------------------------

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute this node's logic.

        Parameters
        ----------
        context:
            Shared flow context passed between nodes.  Nodes may read from
            and write to it freely.

        Returns
        -------
        dict
            Must contain at minimum a ``"next"`` key whose value is either the
            name of the next edge to follow, or ``None`` to stop the flow.
            Any additional keys are merged into *context* for downstream nodes.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.node_id!r})"


class Flow:
    """
    Directed graph of :class:`Node` objects.

    Usage
    -----
    ::

        flow = Flow()
        flow.add_node("start", InputNode("start"))
        flow.add_node("llm",   LLMNode("llm", config={...}))
        flow.add_edge("start", "default", "llm")
        flow.set_entry("start")
        result = flow.run({"user_input": "Hello!"})

    Parameters
    ----------
    name:
        Optional human-readable name for the flow.
    """

    def __init__(self, name: str = "flow") -> None:
        self.name = name
        self._nodes: dict[str, Node] = {}
        # edges[source_id][edge_name] = target_id
        self._edges: dict[str, dict[str, str]] = {}
        self._entry: str | None = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, node: Node) -> "Flow":
        """Register a node.  Returns *self* for chaining."""
        if node_id in self._nodes:
            raise FlowError(f"Node '{node_id}' already registered")
        self._nodes[node_id] = node
        self._edges.setdefault(node_id, {})
        return self

    def add_edge(self, source_id: str, edge_name: str, target_id: str) -> "Flow":
        """Connect *source_id* → *target_id* via *edge_name*."""
        if source_id not in self._nodes:
            raise FlowError(f"Source node '{source_id}' not found")
        if target_id not in self._nodes:
            raise FlowError(f"Target node '{target_id}' not found")
        self._edges[source_id][edge_name] = target_id
        return self

    def set_entry(self, node_id: str) -> "Flow":
        """Set the entry-point node."""
        if node_id not in self._nodes:
            raise FlowError(f"Entry node '{node_id}' not found")
        self._entry = node_id
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, initial_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute the flow starting from the entry node.

        Parameters
        ----------
        initial_context:
            Seed values placed into the shared context before execution.

        Returns
        -------
        dict
            The final shared context after the flow completes.
        """
        if self._entry is None:
            raise FlowError("No entry node set.  Call set_entry() first.")

        context: dict[str, Any] = dict(initial_context or {})
        current_id: str | None = self._entry
        visited: list[str] = []

        while current_id is not None:
            node = self._nodes.get(current_id)
            if node is None:
                raise FlowError(f"Node '{current_id}' not found during execution")

            logger.info("[%s] running node %s", self.name, node)
            visited.append(current_id)

            result = node.run(context)
            if not isinstance(result, dict):
                raise FlowError(
                    f"Node '{current_id}' must return a dict, got {type(result)}"
                )

            # Merge result (except "next") into shared context
            next_edge: str | None = result.pop("next", None)
            context.update(result)

            if next_edge is None:
                logger.info("[%s] flow ended at node %s", self.name, current_id)
                break

            target_id = self._edges.get(current_id, {}).get(next_edge)
            if target_id is None:
                raise FlowError(
                    f"Node '{current_id}' returned edge '{next_edge}' "
                    f"which is not wired up.  Available edges: "
                    f"{list(self._edges.get(current_id, {}).keys())}"
                )
            current_id = target_id

        context["_visited"] = visited
        return context
