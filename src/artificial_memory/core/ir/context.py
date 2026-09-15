from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from .common import (
    MemoryIdentity,
    PriorityTier,
    ResolutionLevel,
)


@dataclass
class BudgetAllocation:
    """Token budget allocation for each priority tier."""
    high: int = 0
    medium: int = 0
    low: int = 0

    def total(self) -> int:
        return self.high + self.medium + self.low

    def get_for_tier(self, tier: PriorityTier) -> int:
        return getattr(self, tier.value)


class ContextPart(BaseModel):
    """A single context part with metadata."""
    model_config = ConfigDict(strict=True, extra="forbid")

    content: str
    tokens: int
    priority: float
    source: str
    tier: PriorityTier = PriorityTier.MEDIUM
    memory_id: int | None = None
    memory_identity: MemoryIdentity | None = None
    resolution_used: ResolutionLevel | None = None
    selection_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextProvenance(BaseModel):
    """Provenance tracking for context parts."""
    model_config = ConfigDict(strict=True, extra="forbid")

    parts_provenance: list[dict[str, Any]] = field(default_factory=list)
    total_memories_considered: int = 0
    total_memories_selected: int = 0
    expansion_decisions: list[dict[str, Any]] = field(default_factory=list)


class ContextStats(BaseModel):
    """Detailed context building statistics."""
    model_config = ConfigDict(strict=True, extra="forbid")

    raw_tokens: int = 0
    effective_tokens: int = 0
    compression_ratio: float = 0.0
    parts_count: int = 0
    selected_parts: int = 0
    tier_distribution: dict[str, int] = field(default_factory=dict)
    priority_distribution: dict[str, int] = field(default_factory=dict)
    truncated_parts: int = 0
    budget_allocation: BudgetAllocation | None = None


class ContextIR(BaseModel):
    """Optimized context package for LLM consumption."""
    model_config = ConfigDict(strict=True, extra="forbid")

    query: str
    budget: BudgetAllocation
    parts: list[ContextPart] = field(default_factory=list)
    stats: ContextStats = field(default_factory=ContextStats)
    provenance: ContextProvenance = field(default_factory=ContextProvenance)
    temporal_snapshot: datetime = field(default_factory=datetime.now)
    system_prompt: str = ""

    def to_compact_string(self) -> str:
        """Compact representation for logging/debugging."""
        parts_str = " | ".join(
            f"[{p.tier.value}] {p.source}: {p.content[:50]}..."
            for p in self.parts
        )
        return f"ContextIR(query='{self.query[:30]}...', parts={len(self.parts)}, tokens={self.stats.effective_tokens}, [{parts_str}])"

    def to_json(self) -> str:
        """JSON serialization for caching/debugging."""
        return self.model_dump_json(indent=2)

    def get_parts_by_tier(self, tier: PriorityTier) -> list[ContextPart]:
        return [p for p in self.parts if p.tier == tier]

    def get_parts_by_memory_id(self, memory_id: int) -> list[ContextPart]:
        return [p for p in self.parts if p.memory_id == memory_id]


def create_context_ir(
    query: str,
    budget: BudgetAllocation,
    system_prompt: str = "",
) -> ContextIR:
    """Factory function to create ContextIR."""
    return ContextIR(
        query=query,
        budget=budget,
        system_prompt=system_prompt,
    )
