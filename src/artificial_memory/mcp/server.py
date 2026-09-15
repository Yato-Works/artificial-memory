"""MCP server exposing Artificial Memory tools (Interface Priority #1: MCP).

Conceptual tools (see V1 plan section 34):
    memory.remember  memory.recall   memory.expand   memory.trace
    memory.explain   memory.timeline memory.inspect

Every tool delegates to ArtificialMemoryRuntime - the MCP layer is purely an
interface adapter, never duplicating runtime logic.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without [mcp] extra
    FastMCP = None
    _MCP_AVAILABLE = False

from artificial_memory.core.models import MemoryType, ResolutionLevel
from artificial_memory.runtime.facade import (
    ArtificialMemoryRuntime,
    RuntimeConfig,
)


def _jsonable(obj: Any) -> Any:
    """Recursively convert arbitrary runtime objects to JSON-safe values.

    Tool responses mix dataclasses (facade results), pydantic models
    (MemoryIR / ContextIR), enums, and datetimes. ``json.dumps(default=str)``
    alone falls back to repr() strings for the pydantic objects nested inside
    dataclass fields, which makes ids unreadable for agents - so we walk the
    structure explicitly.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump(mode="json"))
        except Exception:  # pragma: no cover - model_dump edge cases
            pass
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _jsonable(vars(obj))
    return str(obj)


def _serialize(result: Any) -> str:
    """Best-effort JSON serialization for MCP tool responses."""
    if result is None:
        return "null"
    try:
        return json.dumps(_jsonable(result), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


def create_mcp_server(
    config: RuntimeConfig | None = None,
    runtime: ArtificialMemoryRuntime | None = None,
) -> FastMCP:
    """Create the MCP server instance with all memory tools registered."""
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "The 'mcp' package is required for the MCP server. "
            "Install it with: pip install -e '.[mcp]'"
        )

    am = runtime or ArtificialMemoryRuntime(config)
    mcp = FastMCP(
        "artificial-memory",
        instructions=(
            "Cognitive Memory Runtime: forgetting is resolution reduction, "
            "not deletion. Remember durable facts/decisions, recall with "
            "adaptive resolution, and inspect provenance/time-travel state."
        ),
    )

    @mcp.tool()
    async def memory_remember(
        content: str,
        topic: str,
        memory_type: str = "semantic",
        importance: float = 0.7,
    ) -> str:
        """Store durable information (facts, decisions, preferences) as memory.

        Use for information worth persisting across sessions: decisions with
        rationale, project facts, user preferences. Evidence is immutable;
        beliefs update separately.
        """
        result = await am.remember(
            content,
            topic=topic,
            memory_type=MemoryType(memory_type),
            importance=importance,
        )
        return _serialize(result)

    @mcp.tool()
    async def memory_recall(
        query: str,
        topic: str,
        level: int = 2,
        max_tokens: int = 4000,
    ) -> str:
        """Recall memories relevant to a query with adaptive resolution.

        Returns memories with provenance (source conversation, confidence,
        resolution used). Only expands resolution when tokens provide utility.
        Levels: 0=current, 1=long_term_summary, 2=episode, 3=light, 4=raw.
        """
        result = await am.recall(query, topic=topic, level=level, max_tokens=max_tokens)
        return _serialize(result)

    @mcp.tool()
    async def memory_expand(memory_id: int, target_resolution: int = 0) -> str:
        """Expand a compressed memory to a higher-resolution (more detailed,
        lower-numbered) version. 0=RAW ... 5=DEEP_LONG_TERM."""
        result = await am.expand(memory_id, target_resolution=ResolutionLevel(target_resolution))
        return _serialize(result)

    @mcp.tool()
    async def memory_trace(memory_id: int) -> str:
        """Trace a memory back to its source: conversation, messages, versions."""
        result = await am.trace(memory_id)
        return _serialize(result)

    @mcp.tool()
    async def memory_explain(query: str, topic: str) -> str:
        """Explain how recall would select memories for a query and why."""
        result = await am.explain(query, topic)
        return _serialize(result)

    @mcp.tool()
    async def memory_timeline(topic: str, limit: int = 100) -> str:
        """Get the temporal timeline of a topic (chronological memory events)."""
        result = await am.timeline(topic, limit=limit)
        return _serialize(result)

    @mcp.tool()
    async def memory_inspect(memory_id: int) -> str:
        """Inspect a memory's full state: IR, provenance chain, versions."""
        # NOTE: use the async trace() directly. facade.inspect() calls
        # asyncio.run() internally, which raises "cannot be called from a
        # running event loop" inside the MCP server.
        result = await am.trace(memory_id)
        return _serialize(result)

    return mcp


def main() -> None:
    """Run the MCP server over stdio."""
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    main()
