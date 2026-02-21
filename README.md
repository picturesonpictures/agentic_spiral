# agentic_spiral

An **OpenRouter-powered agentic node-based flow** system.  Build dynamic AI
pipelines by wiring together unique, self-contained blocks that each serve a
distinct purpose: LLM generation, conditional routing, data transformation,
conversational memory, and autonomous tool-using agents.

---

## Architecture

```
flow.py           – Core Flow/Node engine (graph execution, context passing)
openrouter.py     – Thin client for the OpenRouter chat-completions API
nodes/
  input_output.py – InputNode  (entry point)  |  OutputNode (terminal)
  llm.py          – LLMNode    (single OpenRouter chat-completion call)
  router.py       – RouterNode (conditional branching via rule engine)
  transform.py    – TransformNode (pure-Python data transforms, no LLM)
  memory.py       – MemoryNode (rolling conversation history)
  agent.py        – AgentNode  (ReAct-style tool-use loop)
```

### How it works

Each **Node** receives a shared **context** dict, does its work, and returns a
`{"next": "<edge_name>", ...}` dict.  The **Flow** follows the named edge to
the next node, merging returned values into context.  Returning
`{"next": None}` ends the flow.

---

## Node types

| Node | Purpose |
|------|---------|
| `InputNode` | Labels the entry point; initialises the prompt key |
| `OutputNode` | Collects designated keys and terminates the flow |
| `LLMNode` | Calls OpenRouter with a configurable system prompt & model |
| `RouterNode` | Evaluates ordered rules (`eq`, `gt`, `contains`, `matches`, …) and branches |
| `TransformNode` | Applies transforms (`word_count`, `truncate`, `template`, `regex_extract`, …) |
| `MemoryNode` | Writes / reads / clears a rolling conversation history |
| `AgentNode` | Autonomous loop: Thought → Tool call → Observation → Final answer |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenRouter API key

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Get a free key at <https://openrouter.ai/keys>.

### 3. Run the example flow

```bash
python example.py
```

This runs two flows:
- **Agentic Spiral** – generates a draft answer, counts words, refines if too
  long, stores the turn in memory.
- **Agent Flow** – uses a calculator tool (add / multiply) to answer a maths
  question.

---

## Build your own flow

```python
from flow import Flow
from nodes import InputNode, LLMNode, RouterNode, TransformNode, OutputNode

flow = Flow("my_flow")

flow.add_node("in",     InputNode("in"))
flow.add_node("llm",    LLMNode("llm", config={
    "model": "openai/gpt-4o-mini",
    "system_prompt": "You are a pirate.  Answer in pirate speak.",
}))
flow.add_node("count",  TransformNode("count", config={
    "transforms": [{"input_key": "response", "fn": "word_count", "output_key": "wc"}]
}))
flow.add_node("router", RouterNode("router", config={
    "rules": [{"key": "wc", "op": "gt", "value": 50, "edge": "long"}],
    "default_edge": "done",
}))
flow.add_node("done",   OutputNode("done"))

flow.add_edge("in",     "default", "llm")
flow.add_edge("llm",    "default", "count")
flow.add_edge("count",  "default", "router")
flow.add_edge("router", "long",    "done")
flow.add_edge("router", "done",    "done")
flow.set_entry("in")

result = flow.run({"user_input": "Tell me about treasure islands."})
print(result["response"])
```

---

## Tests

```bash
python -m pytest tests/ -v
```

All node types and the flow engine are covered without network calls (LLM/Agent
nodes are monkeypatched in tests).

