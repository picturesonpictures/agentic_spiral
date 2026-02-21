"""
example.py – demonstrates an agentic spiral flow.

The flow:

    [input]
       │ default
    [memory_read]         ← inject prior conversation context
       │ default
    [llm_draft]           ← generate a first-draft answer
       │ default
    [transform_prep]      ← count words and truncate for routing
       │ default
    [router]              ← choose path based on word count
       │ long                         │ short / default
    [llm_refine]          [output]    [output]
       │ default
    [memory_write]
       │ default
    [output]

Run with:
    OPENROUTER_API_KEY=<your_key> python example.py
"""

from __future__ import annotations

import os
import logging

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from flow import Flow
from nodes import (
    AgentNode,
    InputNode,
    LLMNode,
    MemoryNode,
    OutputNode,
    RouterNode,
    TransformNode,
)


def build_spiral_flow() -> Flow:
    """Assemble and return the agentic spiral flow."""
    flow = Flow(name="agentic_spiral")

    # ── Nodes ──────────────────────────────────────────────────────────────
    flow.add_node("input", InputNode("input"))

    flow.add_node("memory_read", MemoryNode("memory_read", config={
        "mode": "read",
        "summary_key": "memory_summary",
    }))

    flow.add_node("llm_draft", LLMNode("llm_draft", config={
        "model": "openai/gpt-4o-mini",
        "system_prompt": (
            "You are a thoughtful assistant.  Answer clearly and concisely. "
            "If previous conversation context is provided, use it."
        ),
        "input_key": "user_input",
        "output_key": "draft",
        "extra_context_keys": ["memory_summary"],
        "temperature": 0.7,
    }))

    flow.add_node("transform_prep", TransformNode("transform_prep", config={
        "transforms": [
            {"input_key": "draft", "fn": "word_count", "output_key": "draft_word_count"},
            {"input_key": "draft", "fn": "strip",      "output_key": "draft"},
        ],
    }))

    flow.add_node("router", RouterNode("router", config={
        "rules": [
            {"key": "draft_word_count", "op": "gt", "value": 80, "edge": "long"},
        ],
        "default_edge": "short",
    }))

    flow.add_node("llm_refine", LLMNode("llm_refine", config={
        "model": "openai/gpt-4o-mini",
        "system_prompt": (
            "You are an expert editor.  Condense the following draft to fewer "
            "than 80 words while keeping all key information."
        ),
        "input_key": "draft",
        "output_key": "response",
        "temperature": 0.3,
    }))

    flow.add_node("memory_write", MemoryNode("memory_write", config={
        "mode": "write",
        "user_key": "user_input",
        "assistant_key": "response",
        "max_turns": 10,
    }))

    flow.add_node("output", OutputNode("output", config={
        "collect": ["response", "draft_word_count", "memory"],
    }))

    # ── Copy draft → response for the short path ──────────────────────────
    flow.add_node("copy_draft", TransformNode("copy_draft", config={
        "transforms": [
            {"input_key": "draft", "fn": "strip", "output_key": "response"},
        ],
    }))

    # ── Edges ──────────────────────────────────────────────────────────────
    flow.add_edge("input",        "default", "memory_read")
    flow.add_edge("memory_read",  "default", "llm_draft")
    flow.add_edge("llm_draft",    "default", "transform_prep")
    flow.add_edge("transform_prep", "default", "router")
    flow.add_edge("router",       "long",    "llm_refine")
    flow.add_edge("router",       "short",   "copy_draft")
    flow.add_edge("llm_refine",   "default", "memory_write")
    flow.add_edge("copy_draft",   "default", "memory_write")
    flow.add_edge("memory_write", "default", "output")

    flow.set_entry("input")
    return flow


def build_agent_flow() -> Flow:
    """
    An alternative flow that uses an AgentNode with a simple calculator tool.
    """

    def add(a: float, b: float) -> str:
        return str(a + b)

    def multiply(a: float, b: float) -> str:
        return str(a * b)

    flow = Flow(name="agent_flow")

    flow.add_node("input", InputNode("input"))
    flow.add_node("agent", AgentNode("agent", config={
        "model": "openai/gpt-4o-mini",
        "system_prompt": (
            "You are a maths assistant.  Use the provided tools to compute results. "
            f"When done, reply with '{AgentNode.__module__}' – just kidding, reply with "
            "'FINAL ANSWER: <result>'."
        ),
        "tools": [
            {
                "name": "add",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
            {
                "name": "multiply",
                "description": "Multiply two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        ],
        "tool_implementations": {"add": add, "multiply": multiply},
        "max_steps": 5,
    }))
    flow.add_node("output", OutputNode("output", config={"collect": ["response"]}))

    flow.add_edge("input", "default", "agent")
    flow.add_edge("agent", "default", "output")
    flow.set_entry("input")
    return flow


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY to run the live demo.")
    else:
        print("=== Agentic Spiral Flow ===")
        spiral = build_spiral_flow()
        ctx = spiral.run({"user_input": "Explain the concept of entropy in thermodynamics."})
        print("\nResponse:", ctx.get("response"))
        print("Word count of draft:", ctx.get("draft_word_count"))
        print("Visited nodes:", ctx.get("_visited"))

        print("\n=== Agent Flow ===")
        agent_flow = build_agent_flow()
        ctx2 = agent_flow.run({"user_input": "What is (3 + 7) * 12?"})
        print("Agent answer:", ctx2.get("response"))
