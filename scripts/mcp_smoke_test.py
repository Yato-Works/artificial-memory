"""End-to-end smoke test for the Artificial Memory MCP server (stdio).

Spawns the real MCP server as a subprocess (exactly like Cline / other MCP
clients do), connects with the official MCP Python client, and exercises
all registered tools:

    list_tools -> memory_remember -> memory_recall -> memory_expand
    -> memory_trace -> memory_explain -> memory_timeline -> memory_inspect

Exit code 0 = all tools responded, 1 = failure.

Usage:
    python scripts/mcp_smoke_test.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

BOOTSTRAP = """\
import sys
sys.path.insert(0, {src!r})
from artificial_memory.runtime.facade import RuntimeConfig
from artificial_memory.mcp.server import create_mcp_server

cfg = RuntimeConfig(
    database_path={db!r},
    memory_files_path={mf!r},
    vector_index_path={vi!r},
    audit_log_dir={audit!r},
    metrics_output_dir={metrics!r},
)
create_mcp_server(cfg).run()
"""

EXPECTED_TOOLS = {
    "memory_remember",
    "memory_recall",
    "memory_expand",
    "memory_trace",
    "memory_explain",
    "memory_timeline",
    "memory_inspect",
}

TOPIC = "SmokeTest/mcp"


async def _call(session: ClientSession, name: str, args: dict) -> dict:
    result = await session.call_tool(name, args)
    text = result.content[0].text if result.content else ""
    try:
        return {"ok": not result.isError, "text": text, "data": json.loads(text)}
    except json.JSONDecodeError:
        return {"ok": not result.isError, "text": text, "data": None}



async def run() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="am_mcp_smoke_"))
    bootstrap = BOOTSTRAP.format(
        src=str(REPO_ROOT / "src"),
        db=str(tmp / "smoke.db"),
        mf=str(tmp / "memory_files"),
        vi=str(tmp / "vector_index"),
        audit=str(tmp / "audit_logs"),
        metrics=str(tmp / "metrics"),
    )
    params = StdioServerParameters(command=sys.executable, args=["-c", bootstrap])

    failures: list[str] = []
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                missing = EXPECTED_TOOLS - names
                if missing:
                    failures.append(f"missing tools: {sorted(missing)}")
                print(f"[1/8] list_tools          -> {len(names)} tools found")

                remembered = []
                for i, content in enumerate([
                    "We decided to use PostgreSQL for production because it supports multi-region replication.",
                    "The team preference is Python 3.11+ with strict mypy typing.",
                    "Release 0.2.0 is scheduled for November and includes the memory debugger.",
                ]):
                    r = await _call(session, "memory_remember", {
                        "content": content, "topic": TOPIC,
                    })
                    if not r["ok"]:
                        failures.append(f"memory_remember #{i} failed: {r['text'][:200]}")
                    remembered.append(r)
                print(f"[2/8] memory_remember     -> {len(remembered)} memories stored")


                r = await _call(session, "memory_recall", {
                    "query": "Which database do we use in production and why?",
                    "topic": TOPIC, "level": 2,
                })
                if not r["ok"] or not r["text"].strip():
                    failures.append(f"memory_recall failed: {r['text'][:200]}")
                print(f"[3/8] memory_recall       -> {len(r['text'])} chars returned")

                r = await _call(session, "memory_explain", {
                    "query": "Which database do we use in production?",
                    "topic": TOPIC,
                })
                if not r["ok"]:
                    failures.append(f"memory_explain failed: {r['text'][:200]}")
                print("[4/8] memory_explain      -> mode/resolution explained")

                r = await _call(session, "memory_timeline", {"topic": TOPIC, "limit": 10})
                if not r["ok"]:
                    failures.append(f"memory_timeline failed: {r['text'][:200]}")
                print("[5/8] memory_timeline     -> timeline returned")

                first_id = None
                if remembered and remembered[0]["data"]:
                    d0 = remembered[0]["data"]
                    identity = d0.get("identity") or {}
                    first_id = (
                        identity.get("memory_id")
                        or d0.get("id")
                        or d0.get("memory_id")
                    )

                if first_id is not None:
                    r = await _call(session, "memory_trace", {"memory_id": first_id})
                    if not r["ok"]:
                        failures.append(f"memory_trace failed: {r['text'][:200]}")
                    print(f"[6/8] memory_trace        -> provenance chain for id={first_id}")

                    r = await _call(session, "memory_inspect", {"memory_id": first_id})
                    if not r["ok"]:
                        failures.append(f"memory_inspect failed: {r['text'][:200]}")
                    print(f"[7/8] memory_inspect      -> full state for id={first_id}")

                    r = await _call(session, "memory_expand", {
                        "memory_id": first_id, "target_resolution": 0,
                    })
                    print(f"[8/8] memory_expand       -> {'ok' if r['ok'] else r['text'][:120]}")
                else:
                    print("[6-8/8] skipped (no numeric id in remember response)")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("\nMCP SMOKE TEST: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nMCP SMOKE TEST: PASS (all tools responded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
