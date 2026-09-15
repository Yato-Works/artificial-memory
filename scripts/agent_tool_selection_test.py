"""Agent tool-selection test: let a real LLM choose Artificial Memory MCP tools.

Unlike ``real_agent_e2e.py`` (scripted calls), this test gives a local LLM
the real tool schemas from the running MCP server and lets it *decide* which
tool to call — the same decision layer a Cline-class agent uses. Only the
tools the LLM names are executed; the script never forces a tool.

Session 1: "remember this OSS project"          -> expect memory_remember
Session 2 (fresh LLM chat, same server/DB):
  - "do you remember ... ?"                     -> expect memory_recall
  - "where did that info come from?"            -> expect memory_trace
  - "when did we save it?"                      -> expect memory_timeline

Usage:
    python scripts/agent_tool_selection_test.py [--model qwen3:4b]

Output: benchmark_results/agent_tool_selection.json + console table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from artificial_memory.llm.manager import OllamaProvider  # noqa: E402

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

SYSTEM_PROMPT = """You are an AI coding agent with access to memory tools over MCP.
Do NOT write your reasoning. Do NOT think out loud. Respond with ONLY one
JSON object and nothing else:
{{"action": "tool", "tool": "<tool name>", "arguments": {{...}}}}
to call a tool, or
{{"action": "answer", "text": "<short answer to the user>"}}
when the user's request is fully handled.

Topic rule: for remember/recall/timeline, use the project name exactly as
the user stated it, consistently across all calls. If recall returns 0
memories, retry once with a shorter topic (drop words like "project").

Available tools:
{tools}"""


def _strip(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_json(reply: str) -> dict:
    """Extract the first JSON object from an LLM reply, tolerating extra text."""
    reply = _strip(reply)
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        pass
    idx = reply.find("{")
    if idx != -1:
        try:
            obj, _end = json.JSONDecoder().raw_decode(reply[idx:])
            return obj
        except json.JSONDecodeError:
            pass
    return {"action": "answer", "text": reply}


def _load_registered_server() -> tuple[str, list[str]]:
    """Read the actual Cline registration so we test the real config."""
    settings = Path(
        Path.home() / "AppData/Roaming/Code/User/globalStorage/"
        "saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
    )
    cfg = json.loads(settings.read_text(encoding="utf-8"))
    entry = cfg["mcpServers"]["artificial-memory"]
    return entry["command"], list(entry["args"])


async def _agent_turn(session: ClientSession, provider: OllamaProvider,
                      tools_block: str, user_message: str, memory: list[dict],
                      transcript: list[dict], max_steps: int = 6) -> tuple[str, list[str]]:
    """One conversation turn: LLM loops choosing tools until it answers."""
    memory.append({"role": "user", "content": user_message})
    used_tools: list[str] = []
    answer = ""
    for _ in range(max_steps):
        raw = await provider.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools_block)},
                *memory,
            ],
            temperature=0.0,
            max_tokens=800,
        )
        reply = _strip(raw)
        decision = _extract_json(reply)
        if decision.get("action") == "answer":
            answer = decision.get("text", "")
            memory.append({"role": "assistant", "content": answer})
            transcript.append({"llm_answer": answer})
            break
        tool_name = decision.get("tool")
        tool_args = decision.get("arguments") or {}
        used_tools.append(tool_name)
        transcript.append({"tool_chosen": tool_name, "arguments": tool_args})
        try:
            result = await session.call_tool(tool_name, tool_args)
            tool_out = result.content[0].text if result.content else ""
            if result.isError:
                tool_out = f"tool error: {tool_out}"
        except Exception as tool_exc:  # noqa: BLE001 - feed errors back to the LLM
            tool_out = f"tool call failed: {type(tool_exc).__name__}: {tool_exc}"
        transcript.append({"tool_result_head": tool_out[:200]})
        # Summarize tool output for the LLM to keep the prompt small.
        head = tool_out[:600]
        memory.append({
            "role": "assistant",
            "content": f"[tool {tool_name} executed, result head]: {head}",
        })
    return answer, used_tools


def _tools_block(tools) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
        if t.inputSchema:
            lines.append(f"  args: {json.dumps(t.inputSchema)}")
    return "\n".join(lines)


async def run(model: str) -> int:
    command, args = _load_registered_server()
    tmp = Path(tempfile.mkdtemp(prefix="am_agent_loop_"))
    provider = OllamaProvider(model=model, think=False)
    transcript: list[dict] = []
    checks: dict[str, bool] = {}

    try:
        # Server cwd = temp dir (isolated db); command/args = Cline's registration.
        params = StdioServerParameters(command=command, args=args, cwd=str(tmp))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                tools_block = _tools_block(tools)

                # ---- Session 1: remember (fresh LLM chat) ----
                chat1: list[dict] = []
                t1 = await _agent_turn(
                    session, provider, tools_block,
                    "We are building an open-source project called Artificial Memory. "
                    "It uses SQLite for storage, FAISS for vector retrieval, MCP for "
                    "agent integration, and it ships LLM benchmarks. "
                    "Remember this project profile for future conversations.",
                    chat1, transcript,
                )
                checks["s1_autonomous_remember"] = "memory_remember" in t1[1]

                # ---- Session 2: fresh LLM chat, same server/db ----
                chat2: list[dict] = []
                await _agent_turn(
                    session, provider, tools_block,
                    "Do you remember what we said about the Artificial Memory "
                    "project? What does it use?",
                    chat2, transcript,
                )
                checks["s2_autonomous_recall"] = "memory_recall" in " ".join(
                    str(t) for t in transcript
                )

                await _agent_turn(
                    session, provider, tools_block,
                    "Where did that information come from? Show the source.",
                    chat2, transcript,
                )
                checks["s2_autonomous_trace"] = any(
                    t.get("tool_chosen") == "memory_trace" for t in transcript
                )

                await _agent_turn(
                    session, provider, tools_block,
                    "When did we save that memory?",
                    chat2, transcript,
                )
                checks["s2_autonomous_timeline"] = any(
                    t.get("tool_chosen") == "memory_timeline" for t in transcript
                )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out_dir = REPO_ROOT / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    evidence = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "note": "tool selection decided by the LLM; script executes only chosen tools",
            "registered_command": [command, *args],
        },
        "checks": checks,
        "transcript": transcript,
    }
    (out_dir / "agent_tool_selection.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Agent Tool-Selection Result ===")
    passed = 0
    for label, key in [
        ("Session1: LLM chose memory_remember", "s1_autonomous_remember"),
        ("Session2: LLM chose memory_recall", "s2_autonomous_recall"),
        ("Session2: LLM chose memory_trace", "s2_autonomous_trace"),
        ("Session2: LLM chose memory_timeline", "s2_autonomous_timeline"),
    ]:
        status = "PASS" if checks.get(key) else "FAIL"
        if checks.get(key):
            passed += 1
        print(f"  {label:38s} {status}")
    print(f"\nOverall: {passed}/4 PASS")
    print(f"Evidence: {out_dir / 'agent_tool_selection.json'}")
    return 0 if passed == 4 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:4b")
    raise SystemExit(asyncio.run(run(parser.parse_args().model)))
