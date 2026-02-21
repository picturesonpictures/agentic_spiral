"""
Tests for the core flow engine (flow.py).
"""

from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flow import Flow, FlowError, Node


class PassThroughNode(Node):
    """Node that just passes through to the configured next edge."""
    def run(self, context):
        context["visited_" + self.node_id] = True
        return {"next": self.config.get("next_edge", "default")}


class StopNode(Node):
    """Node that ends the flow."""
    def run(self, context):
        context["stopped"] = True
        return {"next": None}


class WriterNode(Node):
    """Node that writes a value to context."""
    def run(self, context):
        context[self.config["key"]] = self.config["value"]
        return {"next": self.config.get("next_edge", "default")}


# ---------------------------------------------------------------------------
# Flow construction
# ---------------------------------------------------------------------------

def test_add_and_run_simple_flow():
    flow = Flow("test")
    flow.add_node("a", PassThroughNode("a"))
    flow.add_node("b", StopNode("b"))
    flow.add_edge("a", "default", "b")
    flow.set_entry("a")

    ctx = flow.run({})
    assert ctx["visited_a"] is True
    assert ctx["stopped"] is True
    assert ctx["_visited"] == ["a", "b"]


def test_flow_chaining_api():
    flow = (
        Flow("chain")
        .add_node("a", PassThroughNode("a"))
        .add_node("b", StopNode("b"))
        .add_edge("a", "default", "b")
        .set_entry("a")
    )
    ctx = flow.run()
    assert ctx["_visited"] == ["a", "b"]


def test_initial_context_is_available():
    flow = Flow("ctx")
    flow.add_node("w", WriterNode("w", config={"key": "x", "value": 99}))
    flow.add_node("s", StopNode("s"))
    flow.add_edge("w", "default", "s")
    flow.set_entry("w")
    ctx = flow.run({"pre": "existing"})
    assert ctx["pre"] == "existing"
    assert ctx["x"] == 99


def test_duplicate_node_raises():
    flow = Flow()
    flow.add_node("a", PassThroughNode("a"))
    with pytest.raises(FlowError, match="already registered"):
        flow.add_node("a", PassThroughNode("a"))


def test_missing_entry_raises():
    flow = Flow()
    with pytest.raises(FlowError, match="No entry node"):
        flow.run()


def test_bad_edge_raises():
    flow = Flow()
    flow.add_node("a", PassThroughNode("a", config={"next_edge": "nonexistent"}))
    flow.add_node("b", StopNode("b"))
    flow.add_edge("a", "default", "b")
    flow.set_entry("a")
    with pytest.raises(FlowError, match="not wired up"):
        flow.run()


def test_node_must_return_dict():
    class BadNode(Node):
        def run(self, context):
            return "oops"  # not a dict

    flow = Flow()
    flow.add_node("bad", BadNode("bad"))
    flow.set_entry("bad")
    with pytest.raises(FlowError, match="must return a dict"):
        flow.run()


def test_edge_to_unknown_source_raises():
    flow = Flow()
    flow.add_node("a", PassThroughNode("a"))
    with pytest.raises(FlowError, match="Source node"):
        flow.add_edge("nonexistent", "default", "a")


def test_edge_to_unknown_target_raises():
    flow = Flow()
    flow.add_node("a", PassThroughNode("a"))
    with pytest.raises(FlowError, match="Target node"):
        flow.add_edge("a", "default", "nonexistent")
