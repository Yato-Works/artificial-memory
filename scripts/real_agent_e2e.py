"""Real-agent cross-session E2E test for the Artificial Memory MCP server.

Simulates the exact scenario a Cline-style agent performs, using the same
server launch command registered in ``cline_mcp_settings.json``:

  Session 1 (process A): remember project facts          -> process EXITS
  Session 2 (process B): fresh process, SAME database
      - memory_recall  x3  (purpose / architecture / benchmarks)
      - memory_trace      (provenance of an architecture memory)
      - memory_timeline   (when was this recorded)
      - memory_explain    (why was this retrieved)
      - memory_expand     (raw-resolution detail)
      - memory_inspect    (full IR + provenance state)

PASS/FAIL is computed from actual tool responses only. Evidence is written to
``benchmark_results/real_agent_e2e.json``.

Usage:
    python scripts/real_agent_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

TOPIC = "E2E/real-agent"

FACTS = [
    ("Artificial Memory is an open-source project that provides persistent "
     "memory for AI agents across sessions."),
    ("Artificial Memory's current architecture uses a Python runtime facade "
     "backed by SQLite storage, FAISS vector retrieval, progressive "
     "compression of memories, and MCP integration for agent access."),
    ("The Artificial Memory project ships deterministic system benchmarks, "
     "Ollama LLM benchmarks, and an MCP smoke test so results stay "
     "reproducible."),
]
TIMELINE_NOTE = (
    "Real-agent E2E session: the Artificial Memory architecture facts were "
    "recorded on this date."
)

SESSION_LOG: list[dict] = []


def _log(event: dict) -> None:
    event = dict(event)
    event["at"] = datetime.now().isoformat(timespec="seconds")
    SESSION_LOG.append(event)


def _server_params(cwd: Path) -> StdioServerParameters:
    """Same command/args the Cline registration uses (isolated cwd)."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "artificial_memory.mcp"],
        cwd=str(cwd),
    )


async def _call(session: ClientSession, name: str, args: dict) -> tuple[str, dict | None]:
    t0 = time.perf_counter()
    result = await session.call_tool(name, args)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    text = result.content[0].text if result.content else ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    _log({
        "event": "tool_call", "tool": name, "args": args,
        "is_error": result.isError, "response_bytes": len(text),
        "latency_ms": elapsed_ms,
        "response_head": text[:300],
    })
    if result.isError:
        raise RuntimeError(f"{name} returned an error: {text[:300]}")
    return text, data


async def run() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="am_real_agent_e2e_"))
    params = _server_params(tmp)
    checks: dict[str, bool] = {}
    observations: dict[str, object] = {}
    first_error: str | None = None

    try:
        # ================= Session 1: Remember =================
        _log({"event": "session_start", "session": 1, "note": "remember facts, then process exits"})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                checks["mcp_connection"] = True
                stored_ids = []
                for content in FACTS:
                    _, data = await _call(session, "memory_remember", {
                        "content": content, "topic": TOPIC, "memory_type": "semantic",
                    })
                    mid = ((data or {}).get("identity") or {}).get("memory_id")
                    if mid:
                        stored_ids.append(mid)
                _, data = await _call(session, "memory_remember", {
                    "content": TIMELINE_NOTE, "topic": TOPIC, "memory_type": "timeline",
                })
                mid = ((data or {}).get("identity") or {}).get("memory_id")
                if mid:
                    stored_ids.append(mid)
                checks["remember"] = len(stored_ids) == len(FACTS) + 1
                observations["stored_memory_ids"] = stored_ids
        _log({"event": "session_end", "session": 1, "note": "server process terminated"})

        time.sleep(1.5)  # ensure the OS process is fully gone before session 2

        # ================= Session 2: fresh process, same DB =================
        _log({"event": "session_start", "session": 2, "note": "fresh process, persisted database"})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # --- Cross-session recall (agent asks several targeted queries)
                combined_text = ""
                arch_id: int | None = None
                for query in [
                    "What is the Artificial Memory project and what does it do?",
                    "What is the architecture of the Artificial Memory project?",
                    "What benchmarks or tests does the Artificial Memory project have?",
                ]:
                    text, data = await _call(session, "memory_recall", {
                        "query": query, "topic": TOPIC, "level": 2, "max_tokens": 600,
                    })
                    combined_text += " " + text
                    if data and arch_id is None:
                        mems = data.get("memories") or []
                        if mems:
                            ident = (mems[0].get("identity") or {})
                            arch_id = ident.get("memory_id")

                missing = [m for m in
                           ["open-source", "persistent memory", "AI agents",
                            "SQLite", "FAISS", "compression", "MCP", "benchmark"]
                           if m.lower() not in combined_text.lower()]
                checks["cross_session_recall"] = not missing
                observations["recall_missing_markers"] = missing

                # --- Trace
                if arch_id is not None:
                    _, tdata = await _call(session, "memory_trace", {"memory_id": arch_id})
                    prov = tdata or {}
                    has_chain = bool(prov.get("provenance_chain")) or bool(
                        prov.get("source_conversation")
                    )
                    checks["trace"] = bool(has_chain)
                    observations["trace_keys"] = sorted(prov.keys())
                    observations["traced_memory_id"] = arch_id
                else:
                    checks["trace"] = False
                    observations["trace_keys"] = "no memory id returned by recall"

                # --- Timeline
                text, data = await _call(session, "memory_timeline", {"topic": TOPIC, "limit": 20})
                events = (data or {}).get("events") or []
                checks["timeline"] = len(events) >= 1
                observations["timeline_events"] = [
                    {"date": e.get("date"), "head": str(e.get("content"))[:80]}
                    for e in events[:5]
                ]

                # --- Explain
                text, data = await _call(session, "memory_explain", {
                    "query": "What is the architecture of the Artificial Memory project?",
                    "topic": TOPIC,
                })
                checks["explain"] = bool((data or {}).get("reasoning") or text.strip())
                observations["explain_reasoning_head"] = str((data or {}).get("reasoning"))[:200]

                # --- Expand
                if arch_id is not None:
                    text, data = await _call(session, "memory_expand", {
                        "memory_id": arch_id, "target_resolution": 0,
                    })
                    expanded = ((data or {}).get("semantic_content") or {}).get("content", "")
                    checks["expand"] = bool(expanded.strip())
                    observations["expand_content_head"] = expanded[:160]
                else:
                    checks["expand"] = False

                # --- Inspect
                if arch_id is not None:
                    text, data = await _call(session, "memory_inspect", {"memory_id": arch_id})
                    checks["inspect"] = bool(
                        (data or {}).get("memory_ir")
                    ) and "provenance_chain" in (data or {})
                    observations["inspect_keys"] = sorted((data or {}).keys())
                else:
                    checks["inspect"] = False
        _log({"event": "session_end", "session": 2})
    except Exception as exc:  # noqa: BLE001 - record and report, never crash silently
        parts = [f"{type(exc).__name__}: {exc}"]
        # Unwrap ExceptionGroups so sub-exception details are visible.
        subs = getattr(exc, "exceptions", None)
        depth = 0
        while subs and depth < 3:
            parts.extend(f"  sub: {type(s).__name__}: {s}" for s in subs)
            subs = next((getattr(s, "exceptions", None) for s in subs), None)
            depth += 1
        first_error = " | ".join(parts)
        _log({"event": "error", "error": first_error})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



    out_dir = REPO_ROOT / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    evidence = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "server_command": [sys.executable, "-m", "artificial_memory.mcp"],
            "session_model": "two separate server processes sharing one SQLite db",
        },
        "checks": checks,
        "observations": observations,
        "session_log": SESSION_LOG,
        "first_error": first_error,
    }
    (out_dir / "real_agent_e2e.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    labels = [
        ("MCP connection", "mcp_connection"),
        ("Remember", "remember"),
        ("Cross-session recall", "cross_session_recall"),
        ("Trace", "trace"),
        ("Timeline", "timeline"),
        ("Explain", "explain"),
        ("Expand", "expand"),
        ("Inspect", "inspect"),
    ]
    print("\n=== Real-Agent E2E Result ===")
    passed = 0
    for label, key in labels:
        status = ("PASS" if checks.get(key) else "FAIL") if key in checks else "NOT RUN"
        if checks.get(key):
            passed += 1
        print(f"  {label:22s} {status}")
    total = len(labels)
    print(f"\nOverall: {passed}/{total} PASS")
    if first_error:
        print(f"First error: {first_error}")
    print(f"Evidence: {out_dir / 'real_agent_e2e.json'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
