# Real-Agent Cross-Session E2E Test

Final verification layer: an agent client drives the real MCP server across
**two separate server processes** (fresh conversation = fresh process), sharing
one SQLite database — the same launch command registered in
`cline_mcp_settings.json`.

> Scope note (honest): the test below executes the full MCP protocol against
> the real server with an agent-equivalent MCP stdio client
> (`scripts/real_agent_e2e.py`). The interactive VS Code Cline extension
> conversation itself uses the identical registered command/protocol; run the
> checklist in `mcp-agent-testing.md` to observe it in the UI.

## Environment

- OS: Windows 10, Intel i7-11700, 16 GB RAM
- Python: 3.11 (`mcp>=1.0,<2.0`, FastMCP stdio transport)
- Artificial Memory commit: `50eb164` (pre-E2E), serialize fix applied during this test
- Server: `python -m artificial_memory.mcp` (stdio), one process per session
- Evidence: `benchmark_results/real_agent_e2e.json` (16 tool calls logged with
  latencies, response sizes, and a session log)

## Scenario & Observed Behavior

### Session 1 — Remember (process A)

Stored 3 semantic facts (project purpose / architecture / benchmarks, no
personal data) plus 1 timeline note via `memory_remember` → all returned
`memory_id` 1–4. Then the server **process was terminated**.

### Session 2 — Recall (process B, fresh process, same DB)

| Question asked | Tool | Observed |
|---|---|---|
| "What is the AM project and what does it do?" | `memory_recall` | 4 memories retrieved (176 tokens of the 600 budget); purpose fact present |
| "What is the architecture of the AM project?" | `memory_recall` | SQLite / FAISS / compression / MCP all present in recalled content |
| "What benchmarks or tests does AM have?" | `memory_recall` | deterministic + LLM benchmarks + MCP smoke test retrieved |
| "Where did this memory come from?" | `memory_trace` | `provenance_chain`, `source_conversation`, `exact_source_message` returned for `memory_id=3` (id still valid across processes) |
| "When was this remembered?" | `memory_timeline` | 1 dated event (the timeline note) |
| "Why was this memory retrieved?" | `memory_explain` | "Recalled 4 memories in immediate mode. Used adaptive resolution: RAW. Avg relevance: 0.82" |
| (expansion check) | `memory_expand` | raw-resolution content returned for id 3 |
| (inspection check) | `memory_inspect` | full IR + `provenance_chain` + source keys |

All 8 content markers (open-source, persistent memory, AI agents, SQLite,
FAISS, compression, MCP, benchmark) were found in cross-session recall output.

## Result

| Test | Expected | Result |
|---|---|---|
| MCP connection | Server available | **PASS** |
| Remember | Memory stored | **PASS** |
| Cross-session recall | Correct memory retrieved | **PASS** (8/8 markers) |
| Trace | Provenance available | **PASS** |
| Timeline | Temporal information available | **PASS** |
| Explain | Retrieval explanation available | **PASS** |
| Expand | Memory details available | **PASS** |
| Inspect | Memory inspection works | **PASS** |

**Overall: 8/8 PASS.**

## Bug Found & Fixed During This Test

- **MCP serialization bug**: `RecallResult` (dataclass) contains pydantic
  `MemoryIR` objects. The old `_serialize()` used `asdict()` + `json.dumps(default=str)`,
  which rendered nested `MemoryIR` as Python **repr strings** — agents could
  not read `memory_id` from recall responses, breaking `memory_trace` /
  `memory_expand` / `memory_inspect` workflows. Fixed with a recursive
  JSON-safe converter (`_jsonable`) in `src/artificial_memory/mcp/server.py`.

## Agent Tool-Selection Layer (local LLM as the agent brain)

`scripts/agent_tool_selection_test.py` goes one step further than scripted
calls: a local LLM (qwen3:4b, temperature 0) receives the real tool schemas
from the running MCP server and **decides itself** which tool to call; the
script only executes whatever the LLM chooses.

| Test | Expected | Result |
|---|---|---|
| Session 1: "Remember this project profile" | LLM calls `memory_remember` | **PASS** (memory_id=1 stored) |
| Session 2: "Do you remember what we said about the project?" | LLM calls `memory_recall` | **PASS** (tool chosen autonomously) |
| Session 2: "Where did that information come from?" | LLM calls `memory_trace` | **FAIL** — no memory to trace (see below) |
| Session 2: "When did we save it?" | LLM calls `memory_timeline` | **FAIL** — same cause |

**Overall: 2/4 PASS** — and the failures are the most instructive part:

- The MCP server was healthy throughout (every call that reached it returned
  a valid response). The failures are entirely in the **agent's tool-usage
  layer**: the 4B model stored the fact under topic `"Artificial Memory"` but
  queried `"Artificial Memory project"` → 0 hits → it correctly answered
  "I don't have that memory" instead of hallucinating, but had nothing to
  trace or timeline.
- A small local model could not maintain topic-name consistency even with an
  explicit instruction, and occasionally emitted malformed tool calls
  (`tool: null`). Larger agent models (what Cline actually drives) are
  expected to fare better; this remains unverified in the Cline UI.
- Practical takeaway for agent integrations: either pin the topic explicitly
  in prompts, or have the runtime fall back to fuzzy/recency-based topic
  matching when recall returns 0 (candidate for a future AM improvement, not
  implemented in this release).

## Limitations

- "Session" = separate server process + separate MCP connection sharing one
  DB. In VS Code Cline, a session is a conversation; the persistence
  mechanism (same `memory.db`) is identical, but UI-side behavior (tool
  auto-approval prompts, agent tool choice) is only covered by the manual
  checklist in `mcp-agent-testing.md`.
- Provenance `source_conversation` is null for memories stored via
  `memory_remember` (no conversation context); provenance is richest for
  memories compiled from conversations.
- Timeline only lists `MemoryType.TIMELINE` entries; semantic facts are
  found via recall, not timeline.
