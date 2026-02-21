"""
Tests for all node types (no network calls – LLM/Agent use monkeypatching).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# InputNode / OutputNode
# ---------------------------------------------------------------------------

from nodes.input_output import InputNode, OutputNode


def test_input_node_passes_through():
    node = InputNode("in")
    ctx = {"user_input": "hello"}
    result = node.run(ctx)
    assert result == {"next": "default"}
    assert ctx["user_input"] == "hello"


def test_input_node_creates_missing_key():
    node = InputNode("in")
    ctx = {}
    node.run(ctx)
    assert "user_input" in ctx


def test_input_node_custom_edge():
    node = InputNode("in", config={"next_edge": "start"})
    result = node.run({})
    assert result["next"] == "start"


def test_output_node_collects_keys():
    node = OutputNode("out", config={"collect": ["response", "score"]})
    ctx = {"response": "hello", "score": 42, "other": "ignored"}
    result = node.run(ctx)
    assert result["next"] is None
    assert result["output"] == {"response": "hello", "score": 42}


def test_output_node_missing_key_is_none():
    node = OutputNode("out", config={"collect": ["missing"]})
    result = node.run({})
    assert result["output"]["missing"] is None


# ---------------------------------------------------------------------------
# RouterNode
# ---------------------------------------------------------------------------

from nodes.router import RouterNode


def test_router_matches_eq():
    node = RouterNode("r", config={
        "rules": [{"key": "sentiment", "op": "eq", "value": "positive", "edge": "pos"}],
        "default_edge": "neg",
    })
    result = node.run({"sentiment": "positive"})
    assert result["next"] == "pos"


def test_router_falls_through_to_default():
    node = RouterNode("r", config={
        "rules": [{"key": "x", "op": "eq", "value": 99, "edge": "hit"}],
        "default_edge": "miss",
    })
    result = node.run({"x": 0})
    assert result["next"] == "miss"


def test_router_contains():
    node = RouterNode("r", config={
        "rules": [{"key": "text", "op": "contains", "value": "ERROR", "edge": "err"}],
        "default_edge": "ok",
    })
    assert node.run({"text": "An ERROR occurred"})["next"] == "err"
    assert node.run({"text": "All good"})["next"] == "ok"


def test_router_gt():
    node = RouterNode("r", config={
        "rules": [{"key": "n", "op": "gt", "value": 5, "edge": "big"}],
        "default_edge": "small",
    })
    assert node.run({"n": 10})["next"] == "big"
    assert node.run({"n": 3})["next"] == "small"


def test_router_unknown_op_raises():
    node = RouterNode("r", config={
        "rules": [{"key": "x", "op": "INVALID", "value": 1, "edge": "e"}],
    })
    with pytest.raises(ValueError, match="unknown operator"):
        node.run({"x": 1})


def test_router_matches_regex():
    node = RouterNode("r", config={
        "rules": [{"key": "text", "op": "matches", "value": r"\d{3}", "edge": "has_digits"}],
        "default_edge": "no_digits",
    })
    assert node.run({"text": "abc 123 def"})["next"] == "has_digits"
    assert node.run({"text": "no numbers"})["next"] == "no_digits"


def test_router_skips_none_comparisons():
    """A rule whose context key is None should be skipped, not crash."""
    node = RouterNode("r", config={
        "rules": [{"key": "missing", "op": "gt", "value": 5, "edge": "big"}],
        "default_edge": "other",
    })
    result = node.run({})
    assert result["next"] == "other"


# ---------------------------------------------------------------------------
# TransformNode
# ---------------------------------------------------------------------------

from nodes.transform import TransformNode


def test_transform_word_count():
    node = TransformNode("t", config={
        "transforms": [{"input_key": "text", "fn": "word_count", "output_key": "wc"}]
    })
    result = node.run({"text": "one two three"})
    assert result["wc"] == 3


def test_transform_truncate():
    node = TransformNode("t", config={
        "transforms": [{"input_key": "text", "fn": "truncate", "output_key": "short", "params": 5}]
    })
    result = node.run({"text": "Hello world!"})
    assert result["short"] == "Hello..."


def test_transform_upper_lower():
    node = TransformNode("t", config={
        "transforms": [
            {"input_key": "text", "fn": "upper", "output_key": "up"},
            {"input_key": "text", "fn": "lower", "output_key": "dn"},
        ]
    })
    result = node.run({"text": "MiXeD"})
    assert result["up"] == "MIXED"
    assert result["dn"] == "mixed"


def test_transform_template():
    node = TransformNode("t", config={
        "transforms": [
            {"input_key": "name", "fn": "template", "output_key": "msg", "params": "Hello, {value}!"}
        ]
    })
    result = node.run({"name": "World"})
    assert result["msg"] == "Hello, World!"


def test_transform_regex_extract():
    node = TransformNode("t", config={
        "transforms": [
            {"input_key": "text", "fn": "regex_extract", "output_key": "num", "params": r"(\d+)"}
        ]
    })
    result = node.run({"text": "Order #42 received"})
    assert result["num"] == "42"


def test_transform_unknown_fn_raises():
    node = TransformNode("t", config={
        "transforms": [{"input_key": "x", "fn": "nonexistent"}]
    })
    with pytest.raises(ValueError, match="unknown transform"):
        node.run({"x": "v"})


def test_transform_next_edge_default():
    node = TransformNode("t", config={
        "transforms": [{"input_key": "x", "fn": "strip"}]
    })
    result = node.run({"x": " hi "})
    assert result["next"] == "default"
    assert result["x"] == "hi"


# ---------------------------------------------------------------------------
# MemoryNode
# ---------------------------------------------------------------------------

from nodes.memory import MemoryNode


def test_memory_write_adds_messages():
    node = MemoryNode("m", config={"mode": "write"})
    ctx = {"user_input": "hi", "response": "hello"}
    result = node.run(ctx)
    mem = result["memory"]
    assert len(mem) == 2
    assert mem[0] == {"role": "user", "content": "hi"}
    assert mem[1] == {"role": "assistant", "content": "hello"}


def test_memory_write_trims_old_turns():
    node = MemoryNode("m", config={"mode": "write", "max_turns": 1})
    existing = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
    ]
    ctx = {"user_input": "new", "response": "reply", "memory": existing}
    result = node.run(ctx)
    mem = result["memory"]
    # max_turns=1 → keep last 2 messages
    assert len(mem) == 2
    assert mem[0]["content"] == "new"
    assert mem[1]["content"] == "reply"


def test_memory_read_formats_summary():
    node = MemoryNode("m", config={"mode": "read"})
    memory = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    result = node.run({"memory": memory})
    assert "User: What is 2+2?" in result["memory_summary"]
    assert "Assistant: 4" in result["memory_summary"]


def test_memory_clear():
    node = MemoryNode("m", config={"mode": "clear"})
    ctx = {"memory": [{"role": "user", "content": "old"}]}
    result = node.run(ctx)
    assert result["memory"] == []


def test_memory_unknown_mode_raises():
    node = MemoryNode("m", config={"mode": "unknown"})
    with pytest.raises(ValueError, match="unknown mode"):
        node.run({})


# ---------------------------------------------------------------------------
# LLMNode (mocked)
# ---------------------------------------------------------------------------

from nodes.llm import LLMNode
import nodes.llm as llm_module


def _mock_response(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text, "tool_calls": None}}]
    }


def test_llm_node_writes_output(monkeypatch):
    monkeypatch.setattr(
        llm_module.openrouter, "chat_completion",
        lambda **kwargs: _mock_response("mocked reply"),
    )
    node = LLMNode("llm", config={"output_key": "response"})
    result = node.run({"user_input": "hello"})
    assert result["response"] == "mocked reply"
    assert result["next"] == "default"


def test_llm_node_prepends_extra_context(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _mock_response("ok")

    monkeypatch.setattr(llm_module.openrouter, "chat_completion", fake_completion)
    node = LLMNode("llm", config={"extra_context_keys": ["memory_summary"]})
    node.run({"user_input": "q", "memory_summary": "prev context"})
    user_msg = captured["messages"][1]["content"]
    assert "memory_summary" in user_msg
    assert "prev context" in user_msg


# ---------------------------------------------------------------------------
# AgentNode (mocked)
# ---------------------------------------------------------------------------

from nodes.agent import AgentNode
import nodes.agent as agent_module


def test_agent_node_final_answer(monkeypatch):
    call_count = [0]

    def fake_completion(**kwargs):
        call_count[0] += 1
        return _mock_response("FINAL ANSWER: 42")

    monkeypatch.setattr(agent_module.openrouter, "chat_completion", fake_completion)
    monkeypatch.setattr(agent_module.openrouter, "extract_tool_calls", lambda r: [])
    monkeypatch.setattr(agent_module.openrouter, "extract_text",
                        lambda r: r["choices"][0]["message"]["content"])

    node = AgentNode("agent")
    result = node.run({"user_input": "What is 6*7?"})
    assert result["response"] == "42"
    assert result["next"] == "default"
    assert call_count[0] == 1


def test_agent_node_tool_call_then_answer(monkeypatch):
    """Agent first calls a tool, then gives a final answer."""
    responses = iter([
        # First call: tool invocation
        {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'},
                    }],
                }
            }]
        },
        # Second call: final answer
        _mock_response("FINAL ANSWER: 7"),
    ])

    def fake_completion(**kwargs):
        return next(responses)

    def fake_tool_calls(r):
        return r["choices"][0]["message"].get("tool_calls") or []

    def fake_extract_text(r):
        return r["choices"][0]["message"].get("content") or ""

    monkeypatch.setattr(agent_module.openrouter, "chat_completion", fake_completion)
    monkeypatch.setattr(agent_module.openrouter, "extract_tool_calls", fake_tool_calls)
    monkeypatch.setattr(agent_module.openrouter, "extract_text", fake_extract_text)

    def add(a, b):
        return str(a + b)

    node = AgentNode("agent", config={
        "tools": [{
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        }],
        "tool_implementations": {"add": add},
    })
    result = node.run({"user_input": "3+4?"})
    assert result["response"] == "7"
