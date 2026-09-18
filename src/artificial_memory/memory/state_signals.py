"""Memory state vector computation (AM v0.2.0 Phase 1).

A memory is modeled as *state* with a five-dimensional state vector:

    importance, confidence, currentness, future_utility, contradiction_risk

All signals are computed **deterministically** from runtime evidence
(temporal scope, access history, associations, status). No LLM is involved;
LLM-driven reflection arrives in Phase 5 and only ever *proposes* mutations
(see ``artificial_memory.memory.evolution_policy.EvolutionGovernor``).

Design notes
------------
* The state vector is derived on demand (pure function of the memory plus
  store evidence), not denormalized into the schema. No migration is needed
  and the vector can never drift out of sync with underlying evidence.
* "Forgetting = resolution down" means utility is *not* detail alone: a
  DEEP_LONG_TERM summary may be highly useful. Resolution detail is only a
  small term in ``future_utility``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import AssociationType, Memory, MemoryStatus, ResolutionLevel


@dataclass
class StateSignalConfig:
    """Tunable weights and decay parameters for state signal computation."""

    # Exponential recency decay half-life, in days.
    recency_half_life_days: float = 30.0
    # Access-count saturation constant: usage = count / (count + k).
    access_saturation_k: float = 10.0
    # Temporal relevance assigned when a memory has no validity window.
    no_window_relevance: float = 1.0

    # Weights for currentness (each must be within [0, 1]).
    currentness_temporal_weight: float = 0.40
    currentness_recency_weight: float = 0.35
    currentness_contradiction_weight: float = 0.25

    # Weights for future utility (each must be within [0, 1]).
    utility_importance_weight: float = 0.35
    utility_confidence_weight: float = 0.25
    utility_usage_weight: float = 0.20
    utility_resolution_weight: float = 0.10
    utility_currentness_weight: float = 0.10

    def __post_init__(self) -> None:
        weight_names = [
            "currentness_temporal_weight",
            "currentness_recency_weight",
            "currentness_contradiction_weight",
            "utility_importance_weight",
            "utility_confidence_weight",
            "utility_usage_weight",
            "utility_resolution_weight",
            "utility_currentness_weight",
        ]
        for name in weight_names:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")
        if self.access_saturation_k <= 0:
            raise ValueError("access_saturation_k must be positive")
        if not 0.0 <= self.no_window_relevance <= 1.0:
            raise ValueError("no_window_relevance must be within [0, 1]")


@dataclass
class MemoryStateVector:
    """The v0.2.0 memory state vector plus the raw signals behind it."""

    memory_id: int
    importance: float
    confidence: float
    currentness: float
    future_utility: float
    contradiction_risk: float
    signals: dict[str, float] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "importance": self.importance,
            "confidence": self.confidence,
            "currentness": self.currentness,
            "future_utility": self.future_utility,
            "contradiction_risk": self.contradiction_risk,
            "signals": dict(self.signals),
            "computed_at": self.computed_at.isoformat(),
        }


class StateSignalCalculator:
    """Computes the memory state vector from deterministic runtime signals.

    Signals used:

    * ``recency``            - exponential decay from last access (or creation)
    * ``usage``              - saturating function of access_count
    * ``temporal``           - is ``now`` inside the memory's validity window
    * ``contradiction_risk`` - strength of CONTRADICTS associations
    * ``resolution_detail``  - how much detail the representation holds
    """

    def __init__(self, store: MemoryStore, config: StateSignalConfig | None = None):
        self.store = store
        self.config = config or StateSignalConfig()

    # ==================== Public API ====================

    def compute_state_vector(
        self, memory: Memory, now: datetime | None = None
    ) -> MemoryStateVector:
        """Compute the state vector for a single memory."""
        now = now or datetime.now()
        signals = self._compute_signals(memory, now)

        currentness = (
            self.config.currentness_temporal_weight * signals["temporal"]
            + self.config.currentness_recency_weight * signals["recency"]
            + self.config.currentness_contradiction_weight * (1.0 - signals["contradiction_risk"])
        )

        future_utility = (
            self.config.utility_importance_weight * memory.importance
            + self.config.utility_confidence_weight * signals["confidence"]
            + self.config.utility_usage_weight * signals["usage"]
            + self.config.utility_resolution_weight * signals["resolution_detail"]
            + self.config.utility_currentness_weight * currentness
        )

        return MemoryStateVector(
            memory_id=memory.id or 0,
            importance=memory.importance,
            confidence=signals["confidence"],
            currentness=_clamp01(currentness),
            future_utility=_clamp01(future_utility),
            contradiction_risk=signals["contradiction_risk"],
            signals=signals,
            computed_at=now,
        )


    def compute_for_memories(
        self, memories: list[Memory], now: datetime | None = None
    ) -> list[MemoryStateVector]:
        """Compute state vectors for many memories."""
        return [self.compute_state_vector(memory, now=now) for memory in memories]

    def rank_by_future_utility(
        self,
        memories: list[Memory],
        now: datetime | None = None,
        reverse: bool = True,
    ) -> list[tuple[Memory, MemoryStateVector]]:
        """Rank memories by future utility (for context allocation / reflection)."""
        pairs = [(memory, self.compute_state_vector(memory, now=now)) for memory in memories]
        pairs.sort(key=lambda pair: pair[1].future_utility, reverse=reverse)
        return pairs

    # ==================== Signal computation ====================

    def _compute_signals(self, memory: Memory, now: datetime) -> dict[str, float]:
        return {
            "recency": self._signal_recency(memory, now),
            "usage": self._signal_usage(memory),
            "temporal": self._signal_temporal(memory, now),
            "contradiction_risk": self._signal_contradiction_risk(memory),
            "resolution_detail": self._signal_resolution_detail(memory),
            "confidence": _clamp01(memory.confidence),
        }

    def _signal_recency(self, memory: Memory, now: datetime) -> float:
        """Exponential recency decay with a configurable half-life."""
        anchor = memory.last_accessed or memory.created_at
        days_inactive = max(0.0, (now - anchor).total_seconds() / 86400.0)
        half_life = self.config.recency_half_life_days
        decay: float = 0.5 ** (days_inactive / half_life)
        return decay

    def _signal_usage(self, memory: Memory) -> float:
        """Saturating usage signal: never reaches 1, rewards repeated access."""
        k = self.config.access_saturation_k
        count = max(0, memory.access_count)
        return count / (count + k)

    def _signal_temporal(self, memory: Memory, now: datetime) -> float:
        """1.0 while ``now`` is inside the validity window, 0.0 outside it."""
        valid_from = memory.valid_from
        valid_until = memory.valid_until
        if valid_until is not None and now > valid_until:
            return 0.0
        if valid_from is not None and now < valid_from:
            return 0.0
        if valid_from is None and valid_until is None:
            return self.config.no_window_relevance
        return 1.0


    def _signal_contradiction_risk(self, memory: Memory) -> float:
        """Max strength of outgoing CONTRADICTS associations (0.0 if none).

        The contradiction module records associations on the contradicting
        memory, so scanning outgoing edges from this memory covers links
        created in both directions.
        """
        if memory.id is None:
            return 0.0
        try:
            associations = self.store.get_associations(memory.id, AssociationType.CONTRADICTS)
        except AttributeError:
            return 0.0
        strengths = [a.strength for a in associations if a.strength is not None]
        if not strengths:
            return 0.0
        return _clamp01(max(strengths))

    def _signal_resolution_detail(self, memory: Memory) -> float:
        """How much detail the current representation holds (RAW=1.0)."""
        detail_span = len(ResolutionLevel) - 1  # 5
        return 1.0 - (memory.resolution.value / detail_span)

    def is_dormant(self, memory: Memory) -> bool:
        return memory.status == MemoryStatus.DORMANT


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def create_state_signal_calculator(
    store: MemoryStore,
    config: StateSignalConfig | None = None,
) -> StateSignalCalculator:
    return StateSignalCalculator(store, config)


__all__ = [
    "MemoryStateVector",
    "StateSignalCalculator",
    "StateSignalConfig",
    "create_state_signal_calculator",
]



