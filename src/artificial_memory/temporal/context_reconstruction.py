from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from artificial_memory.context.builder import EnhancedContextBuilder
from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.ir import ContextIR
from artificial_memory.core.models import (
    Memory,
    MemoryStatus,
)
from artificial_memory.recall.engine import RecallEngine


@dataclass
class ReconstructedContext:
    """A reconstructed context from a historical point in time."""
    timestamp: datetime
    query: str
    context_ir: ContextIR
    text: str
    stats: dict[str, Any]
    provenance: dict[str, Any]
    source_memories: list[int] = field(default_factory=list)


@dataclass
class ContextReconstructionConfig:
    """Configuration for context reconstruction."""
    include_rejected: bool = False
    include_truncated: bool = True
    show_selection_reasons: bool = True
    max_tokens: int = 8000


class ContextReconstructor:
    """Reconstructs LLM contexts as they would have been built at historical timestamps.

    This allows understanding what context would have been provided to an LLM
    at any point in the system's history.
    """

    def __init__(
        self,
        store: MemoryStore,
        recall_engine: RecallEngine,
        context_builder: EnhancedContextBuilder,
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.context_builder = context_builder

    def reconstruct_context_at(
        self,
        timestamp: datetime,
        query: str,
        topic_id: int | None = None,
        config: ContextReconstructionConfig | None = None,
    ) -> ReconstructedContext:
        """Reconstruct the context that would have been built at a timestamp."""
        config = config or ContextReconstructionConfig()

        # Get memory state at timestamp
        memories_at_time = self._get_memories_at(timestamp, topic_id)

        # Filter to current memories at that time
        current_memories = [
            m for m in memories_at_time
            if self._was_current_at(m, timestamp)
        ]

        # Build context using the builder with historical memories
        context_ir = self.context_builder.build_context_ir(
            query=query,
            topic_id=topic_id,
            max_tokens=config.max_tokens,
            current_memories=current_memories,
        )

        # Generate text representation
        text = self._context_ir_to_text(context_ir, config)

        # Collect provenance
        provenance = self._build_provenance(context_ir, timestamp)

        return ReconstructedContext(
            timestamp=timestamp,
            query=query,
            context_ir=context_ir,
            text=text,
            stats=self._extract_stats(context_ir),
            provenance=provenance,
            source_memories=[p.memory_id for p in context_ir.parts if p.memory_id],
        )

    def reconstruct_context_evolution(
        self,
        query: str,
        topic_id: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        interval_days: int = 7,
    ) -> list[ReconstructedContext]:
        """Reconstruct context evolution over a time period."""
        contexts = []

        if not end:
            end = datetime.now()
        if not start:
            # Find earliest memory
            memories = self.store.get_memories(limit=1)
            start = memories[0].created_at if memories else datetime.now() - timedelta(days=30)

        current = start
        while current <= end:
            try:
                self.reconstruct_context_at(
                    timestamp=current,
                    query="current state",  # Generic query for state reconstruction
                    topic_id=None,  # Would need topic_id parameter
                )
                contexts.append(ReconstructedContext(
                    timestamp=current,
                    query="current state",
                    context_ir=contexts[-1].context_ir if contexts else None,
                    text="",
                    stats={},
                    provenance={},
                ))
            except Exception:
                pass  # Skip timestamps with errors

            current += timedelta(days=interval_days)

        return contexts

    def compare_contexts(
        self,
        timestamp_a: datetime,
        timestamp_b: datetime,
        query: str,
        topic_id: int | None = None,
    ) -> dict[str, Any]:
        """Compare contexts at two different timestamps."""
        ctx_a = self.reconstruct_context_at(timestamp_a, query, None)
        ctx_b = self.reconstruct_context_at(timestamp_b, query, None)

        # Compare parts
        parts_a = {p.source: p for p in ctx_a.context_ir.parts}
        parts_b = {p.source: p for p in ctx_b.context_ir.parts}

        all_sources = set(parts_a.keys()) | set(parts_b.keys())

        changes = []
        for source in all_sources:
            part_a = parts_a.get(source)
            part_b = parts_b.get(source)

            if part_a and not part_b:
                changes.append({
                    "type": "part_removed",
                    "source": source,
                    "tokens_removed": part_a.tokens,
                })
            elif not part_a and part_b:
                changes.append({
                    "type": "part_added",
                    "source": part_b.source,
                    "tokens_added": part_b.tokens,
                })
            elif part_a.tokens != part_b.tokens:
                changes.append({
                    "type": "tokens_changed",
                    "source": source,
                    "tokens_a": part_a.tokens,
                    "tokens_b": part_b.tokens,
                    "delta": part_b.tokens - part_a.tokens,
                })

        return {
            "timestamp_a": timestamp_a.isoformat(),
            "timestamp_b": timestamp_b.isoformat(),
            "tokens_a": ctx_a.context_ir.stats.effective_tokens,
            "tokens_b": ctx_b.context_ir.stats.effective_tokens,
            "token_delta": ctx_b.context_ir.stats.effective_tokens - ctx_a.context_ir.stats.effective_tokens,
            "changes": changes,
        }

    def get_context_at_decision_point(
        self,
        decision_memory_id: int,
        query: str = "decision context",
    ) -> ReconstructedContext | None:
        """Get the context that existed when a decision was made."""
        decision_mem = self.store.get_memory(decision_memory_id)
        if not decision_mem or decision_mem.memory_type.value != "decision":
            return None

        # Use the decision's timestamp
        timestamp = decision_mem.valid_from or decision_mem.created_at
        topic_id = decision_mem.topic_id

        return self.reconstruct_context_at(timestamp, query, topic_id)

    # ==================== Helper Methods ====================

    def _get_memories_at(
        self,
        timestamp: datetime,
        topic_id: int | None = None,
    ) -> list[Memory]:
        """Get all memories that existed at a timestamp."""
        if topic_id:
            memories = self.store.get_memories(topic_id=topic_id, limit=1000)
        else:
            memories = self.store.get_memories(limit=1000)

        # Filter to memories that existed at timestamp
        return [
            m for m in memories
            if m.created_at <= timestamp and not self._was_archived_before(m, timestamp)
        ]

    def _was_current_at(self, memory: Memory, timestamp: datetime) -> bool:
        """Check if a memory was 'current' at a timestamp."""
        if not memory.is_current:
            return False

        # Check if it was already superseded/archived
        if memory.valid_until and memory.valid_until <= timestamp:
            return False

        # Check if it was archived before timestamp
        if memory.status in [MemoryStatus.ARCHIVED, MemoryStatus.DEEP_ARCHIVED]:
            if memory.updated_at and memory.updated_at <= timestamp:
                return False

        return True

    def _was_archived_before(self, memory: Memory, timestamp: datetime) -> bool:
        """Check if memory was archived before timestamp."""
        if memory.status in [MemoryStatus.ARCHIVED, MemoryStatus.DEEP_ARCHIVED]:
            if memory.updated_at and memory.updated_at <= timestamp:
                return True
        return False

    def _build_provenance(self, context_ir: ContextIR, timestamp: datetime) -> dict[str, Any]:
        """Build provenance information for reconstructed context."""
        return {
            "reconstructed_at": timestamp.isoformat(),
            "parts": [
                {
                    "memory_id": p.memory_id,
                    "source": p.source,
                    "tier": p.tier.value,
                    "priority": p.priority,
                    "tokens": p.tokens,
                    "selected": True,
                    "selection_reason": p.metadata.get("selection_reason", ""),
                }
                for p in context_ir.parts
            ],
        }

    def _extract_stats(self, context_ir: ContextIR) -> dict[str, Any]:
        """Extract statistics from ContextIR."""
        return {
            "effective_tokens": context_ir.stats.effective_tokens,
            "raw_tokens": context_ir.stats.raw_tokens,
            "compression_ratio": context_ir.stats.compression_ratio,
            "parts_count": context_ir.stats.parts_count,
            "selected_parts": context_ir.stats.selected_parts,
            "tier_distribution": context_ir.stats.tier_distribution,
        }

    def _context_ir_to_text(
        self,
        context_ir: ContextIR,
        config: ContextReconstructionConfig,
    ) -> str:
        """Convert ContextIR to human-readable text."""
        lines = [
            "Context Reconstruction",
            f"Query: {context_ir.query}",
            f"Timestamp: {context_ir.timestamp.isoformat()}",
            f"Tokens: {context_ir.stats.effective_tokens}/{context_ir.stats.raw_tokens}",
            f"Compression: {context_ir.stats.compression_ratio:.2f}x",
            "",
            "Parts:",
        ]

        for part in context_ir.parts:
            marker = "✓" if part.content in context_ir.stats.get("selected_content", []) else "✗"
            lines.append(
                f"  {marker} [{part.tier.value}] {part.source} "
                f"(priority={part.priority:.2f}, {part.tokens} tokens)"
            )
            if config.show_selection_reasons and part.metadata.get("selection_reason"):
                lines.append(f"      Reason: {part.metadata['selection_reason']}")
            if config.include_truncated and len(part.content) > 200:
                lines.append(f"      Content: {part.content[:200]}... [truncated]")
            elif not config.include_truncated:
                lines.append(f"      Content: {part.content[:200]}")
            else:
                lines.append(f"      Content: {part.content}")

        return "\n".join(lines)


def create_context_reconstructor(
    store: MemoryStore,
    recall_engine: RecallEngine,
    context_builder: EnhancedContextBuilder,
) -> ContextReconstructor:
    return ContextReconstructor(store, recall_engine, context_builder)
