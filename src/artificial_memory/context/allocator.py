"""Context allocator (AM v0.2.0 Phase 4).

Plan #18: the context window is a constrained resource. Instead of stuffing
top-k memories into it, every candidate receives one of five context
representations::

    FULL | COMPRESSED | SUMMARY | TAG | OMIT

chosen by priority. Allocation therefore optimizes not "how much context" but
**useful information per token**.

Inputs to the allocation decision (plan #18):

* query relevance, importance, confidence, temporal relevance,
* contradiction risk, resolution, token cost, expected utility.

Design notes
------------
* The allocator is deterministic and LLM-free: priorities come from the
  Phase 1 state vector plus lexical query coverage, and COMPRESSED / SUMMARY
  texts come from stored ``MemoryVersion``s produced by the compression
  pipeline (never generated on the fly). When the preferred resolution is not
  stored, the allocator degrades one representation further.
* TAGs are synthesized deterministically from the memory's own tokens plus
  its id, so even the minimal representation is traceable.
* Output preserves priority order and records the reason for every OMIT, so
  the allocation is auditable (plan #29).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from artificial_memory.context.builder import TokenCounter
from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory, MemoryVersion, ResolutionLevel

if TYPE_CHECKING:  # imported lazily at runtime to avoid an import cycle:
    # context -> memory.__init__ -> memory.compiler -> compiler -> context
    from artificial_memory.memory.state_signals import StateSignalCalculator


class RepresentationLevel(StrEnum):
    """How much of a memory enters the context window (plan #18)."""

    FULL = "full"
    COMPRESSED = "compressed"
    SUMMARY = "summary"
    TAG = "tag"
    OMIT = "omit"


@dataclass
class AllocatorConfig:
    """Thresholds and weights for the allocation decision."""

    # Priority thresholds selecting the *desired* representation.
    full_threshold: float = 0.70
    compressed_threshold: float = 0.50
    summary_threshold: float = 0.30
    # Candidates below the relevance floor are OMITed regardless of budget.
    relevance_floor: float = 0.05
    # Max tokens for a synthesized TAG.
    tag_max_tokens: int = 14
    # How much information each representation retains (plan: "useful
    # information per token"). Used for the utility-per-token statistic.
    retention: dict[str, float] = field(
        default_factory=lambda: {"full": 1.0, "compressed": 0.6, "summary": 0.3, "tag": 0.1}
    )
    # Priority weights (each must be within [0, 1]).
    relevance_weight: float = 0.40
    importance_weight: float = 0.20
    confidence_weight: float = 0.15
    temporal_weight: float = 0.10
    resolution_weight: float = 0.05
    contradiction_penalty: float = 0.30

    def __post_init__(self) -> None:
        weights = [
            "full_threshold",
            "compressed_threshold",
            "summary_threshold",
            "relevance_floor",
            "relevance_weight",
            "importance_weight",
            "confidence_weight",
            "temporal_weight",
            "resolution_weight",
            "contradiction_penalty",
        ]
        for name in weights:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if not (self.full_threshold >= self.compressed_threshold >= self.summary_threshold):
            raise ValueError("representation thresholds must be ordered full >= compressed >= summary")
        if self.tag_max_tokens <= 0:
            raise ValueError("tag_max_tokens must be positive")
        for name, value in self.retention.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"retention[{name!r}] must be within [0, 1], got {value}")

    def retention_for(self, level: RepresentationLevel) -> float:
        return self.retention.get(level.value, 0.0)

@dataclass
class AllocatedPart:
    """One candidate's final representation in the context window."""

    memory: Memory
    level: RepresentationLevel
    content: str
    tokens: int
    priority: float
    relevance: float
    desired_level: RepresentationLevel
    retention: float = 0.0
    memory_id: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        self.memory_id = self.memory.id or 0

    @property
    def included(self) -> bool:
        return self.level is not RepresentationLevel.OMIT

    @property
    def useful_information(self) -> float:
        """Priority-weighted retained information this part contributes."""
        if not self.included:
            return 0.0
        return self.priority * self.retention

    @property
    def utility_per_token(self) -> float:
        if self.tokens <= 0:
            return 0.0
        return self.useful_information / self.tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "level": self.level.value,
            "tokens": self.tokens,
            "priority": self.priority,
            "relevance": self.relevance,
            "desired_level": self.desired_level.value,
            "degraded": self.level is not self.desired_level,
            "reason": self.reason,
            "resolution": self.memory.resolution.name,
        }

@dataclass
class ContextAllocation:
    """The result of one allocation pass over the token budget."""

    query: str
    max_tokens: int
    parts: list[AllocatedPart] = field(default_factory=list)
    omitted: list[AllocatedPart] = field(default_factory=list)
    considered: int = 0

    @property
    def included(self) -> list[AllocatedPart]:
        return [part for part in self.parts if part.included]

    @property
    def used_tokens(self) -> int:
        return sum(part.tokens for part in self.included)

    @property
    def distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for part in self.parts:
            counts[part.level.value] = counts.get(part.level.value, 0) + 1
        return counts

    @property
    def avg_utility_per_token(self) -> float:
        tokens = self.used_tokens
        if tokens <= 0:
            return 0.0
        return sum(part.useful_information for part in self.included) / tokens

    def memory_ids(self) -> list[int]:
        return [part.memory_id for part in self.included]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "max_tokens": self.max_tokens,
            "considered": self.considered,
            "used_tokens": self.used_tokens,
            "distribution": self.distribution,
            "avg_utility_per_token": self.avg_utility_per_token,
            "parts": [part.to_dict() for part in self.parts],
            "omitted": [part.to_dict() for part in self.omitted],
        }

    def to_text(self) -> str:
        """Render the allocation as the actual context text for the model.

        Representation levels are marked so the answering model knows how
        much trust to place in each part; omitted memories are not rendered.
        """
        blocks: list[str] = []
        for part in self.included:
            blocks.append(f"[{part.level.value.upper()}] {part.content}")
        return "\n\n".join(blocks)

class ContextAllocator:
    """Allocates context-window tokens across candidate memories (plan #18).

    The allocator never invents content: COMPRESSED / SUMMARY texts come from
    stored ``MemoryVersion``s, TAGs are synthesized deterministically from the
    memory's own tokens, and FULL is the memory as stored.
    """

    # Downgrade path applied when the desired representation does not fit.
    DOWNGRADE: dict[RepresentationLevel, list[RepresentationLevel]] = {
        RepresentationLevel.FULL: [
            RepresentationLevel.FULL,
            RepresentationLevel.COMPRESSED,
            RepresentationLevel.SUMMARY,
            RepresentationLevel.TAG,
        ],
        RepresentationLevel.COMPRESSED: [
            RepresentationLevel.COMPRESSED,
            RepresentationLevel.SUMMARY,
            RepresentationLevel.TAG,
        ],
        RepresentationLevel.SUMMARY: [RepresentationLevel.SUMMARY, RepresentationLevel.TAG],
        RepresentationLevel.TAG: [RepresentationLevel.TAG],
    }

    def __init__(
        self,
        store: MemoryStore,
        config: AllocatorConfig | None = None,
        token_counter: TokenCounter | None = None,
        calculator: StateSignalCalculator | None = None,
    ):
        self.store = store
        self.config = config or AllocatorConfig()
        self.token_counter = token_counter or TokenCounter()
        # Lazy import: artificial_memory.memory's package init pulls the
        # compiler, which imports this package (import-cycle guard).
        from artificial_memory.memory.state_signals import StateSignalCalculator

        self.calculator = calculator or StateSignalCalculator(store)

    # ==================== Public API ====================

    def allocate(
        self,
        query: str,
        candidates: list[Memory],
        max_tokens: int,
        now: datetime | None = None,
    ) -> ContextAllocation:
        """Allocate ``max_tokens`` across ``candidates`` for ``query``."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        now = now or datetime.now()
        # Lazy import (import-cycle guard): see ContextAllocator.__init__.
        from artificial_memory.memory.evidence import tokenize

        query_tokens = tokenize(query)

        scored = [(memory, self._score(memory, query_tokens, now)) for memory in candidates]
        scored.sort(key=lambda entry: (-entry[1][0], entry[0].id or 0))

        allocation = ContextAllocation(query=query, max_tokens=max_tokens, considered=len(candidates))
        remaining = max_tokens

        for memory, (priority, relevance, signals) in scored:
            if not memory.is_current:
                allocation.omitted.append(
                    self._omitted(memory, priority, relevance, "superseded / not current")
                )
                continue
            if relevance < self.config.relevance_floor:
                allocation.omitted.append(
                    self._omitted(memory, priority, relevance, "below relevance floor")
                )
                continue

            desired = self._desired_level(priority)
            part = self._place(memory, desired, priority, relevance, remaining)
            if part.level is RepresentationLevel.OMIT:
                allocation.omitted.append(part)
                continue
            remaining -= part.tokens
            allocation.parts.append(part)

        return allocation

    def allocate_from_reconstruction(
        self,
        package: Any,
        max_tokens: int,
        now: datetime | None = None,
    ) -> ContextAllocation:
        """Allocate the budget over a Phase 3 reconstructed evidence package.

        Uses the package's evidence items (already ranked, temporally filtered
        and conflict-checked) as the candidate pool.
        """
        now = now or package.timestamp
        memories = [item.memory for item in package.evidence]
        return self.allocate(package.query, memories, max_tokens, now=now)

    # ==================== Scoring ====================

    def _score(
        self,
        memory: Memory,
        query_tokens: set[str],
        now: datetime,
    ) -> tuple[float, float, dict[str, float]]:
        """Return ``(priority, relevance, signals)`` for one candidate.

        Priority covers every factor from plan #18: query relevance,
        importance, confidence, temporal relevance, contradiction risk and
        resolution (token cost enters via the representation choice below).
        """
        relevance = self._relevance(memory, query_tokens)
        state = self.calculator.compute_state_vector(memory, now=now)
        signals = state.signals

        priority = (
            self.config.relevance_weight * relevance
            + self.config.importance_weight * memory.importance
            + self.config.confidence_weight * state.confidence
            + self.config.temporal_weight * signals.get("temporal", 1.0)
            + self.config.resolution_weight * signals.get("resolution_detail", 0.0)
            - self.config.contradiction_penalty * state.contradiction_risk
        )
        priority = max(0.0, min(1.0, priority))
        return priority, relevance, signals

    @staticmethod
    def _relevance(memory: Memory, query_tokens: set[str]) -> float:
        # Lazy import (import-cycle guard): see ContextAllocator.__init__.
        from artificial_memory.memory.evidence import tokenize

        if not query_tokens:
            return 0.0
        shared = query_tokens & tokenize(memory.content)
        if not shared:
            return 0.0
        return len(shared) / len(query_tokens)

    def _desired_level(self, priority: float) -> RepresentationLevel:
        if priority >= self.config.full_threshold:
            return RepresentationLevel.FULL
        if priority >= self.config.compressed_threshold:
            return RepresentationLevel.COMPRESSED
        if priority >= self.config.summary_threshold:
            return RepresentationLevel.SUMMARY
        return RepresentationLevel.TAG

    # ==================== Rendering ====================

    def _place(
        self,
        memory: Memory,
        desired: RepresentationLevel,
        priority: float,
        relevance: float,
        remaining: int,
    ) -> AllocatedPart:
        """Pick the highest representation that fits ``remaining`` tokens."""
        for level in self.DOWNGRADE[desired]:
            effective, content = self._render(memory, level)
            tokens = self.token_counter.count(content)
            if tokens <= remaining:
                reason = (
                    "desired representation"
                    if effective is desired
                    else f"degraded from {desired.value} to {effective.value} to fit the budget"
                )
                return AllocatedPart(
                    memory=memory,
                    level=effective,
                    content=content,
                    tokens=tokens,
                    priority=priority,
                    relevance=relevance,
                    desired_level=desired,
                    retention=self.config.retention_for(effective),
                    reason=reason,
                )
        return self._omitted(memory, priority, relevance, "budget exhausted")

    def _render(self, memory: Memory, level: RepresentationLevel) -> tuple[RepresentationLevel, str]:
        """Content for one representation level, degrading if unstored.

        Returns the *effective* level actually rendered together with its
        content, so callers never label TAG content as COMPRESSED.
        """
        if level is RepresentationLevel.FULL:
            return RepresentationLevel.FULL, memory.content
        if level is RepresentationLevel.COMPRESSED:
            version = self._version(memory, steps=1)
            if version is not None:
                return RepresentationLevel.COMPRESSED, version.content
            return self._render(memory, RepresentationLevel.SUMMARY)
        if level is RepresentationLevel.SUMMARY:
            version = self._version(memory, steps=2)
            if version is not None:
                return RepresentationLevel.SUMMARY, version.content
            return self._render(memory, RepresentationLevel.TAG)
        if level is RepresentationLevel.TAG:
            return RepresentationLevel.TAG, self._tag(memory)
        return RepresentationLevel.OMIT, ""

    def _version(self, memory: Memory, steps: int) -> MemoryVersion | None:
        """The stored compression ``steps`` levels below the current resolution."""
        if memory.id is None:
            return None
        resolutions = [level.value for level in ResolutionLevel]
        index = resolutions.index(memory.resolution.value)
        target = min(index + steps, len(resolutions) - 1)
        if target == index:
            return None
        for level_value in resolutions[index + 1 : target + 1]:
            version = self.store.get_memory_version(memory.id, ResolutionLevel(level_value))
            if version is not None:
                return version
        return None

    def _tag(self, memory: Memory) -> str:
        """Deterministic minimal representation: id + leading content tokens."""
        from artificial_memory.memory.evidence import tokenize  # cycle guard

        tokens = sorted(tokenize(memory.content))
        if not tokens:
            stripped = memory.content.strip()
            tokens = [stripped[:12]] if stripped else ["empty"]
        words = tokens[:4]
        return f"#{memory.id or 0} " + " · ".join(words)

    # ==================== Helpers ====================

    def _omitted(
        self,
        memory: Memory,
        priority: float,
        relevance: float,
        reason: str,
    ) -> AllocatedPart:
        return AllocatedPart(
            memory=memory,
            level=RepresentationLevel.OMIT,
            content="",
            tokens=0,
            priority=priority,
            relevance=relevance,
            desired_level=self._desired_level(priority),
            retention=0.0,
            reason=reason,
        )


def create_context_allocator(
    store: MemoryStore,
    config: AllocatorConfig | None = None,
    token_counter: TokenCounter | None = None,
) -> ContextAllocator:
    return ContextAllocator(store, config, token_counter)


__all__ = [
    "AllocatedPart",
    "AllocatorConfig",
    "ContextAllocation",
    "ContextAllocator",
    "RepresentationLevel",
    "create_context_allocator",
]





