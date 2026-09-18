"""Evidence ranking for reconstruction (AM v0.2.0 Phase 3).

Plan #15 / #17: reconstruction must produce a *small, high-quality evidence
package* rather than a large candidate pool. This module scores and orders the
nodes discovered by :mod:`artificial_memory.memory.graph`.

Scoring is deterministic and LLM-free. A candidate's score combines:

* ``lexical``       - query/candidate token overlap (Japanese-aware)
* ``graph``         - proximity to the seed memories that started the walk
* ``state``         - the Phase 1 state vector (future utility, confidence)
* ``temporal``      - validity window at the reconstruction time
* ``penalties``     - contradiction risk and non-current (superseded) status

Every item carries human-readable ``reasons`` so an evidence package can
explain *why* each memory was included (plan #29 auditability).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory
from artificial_memory.memory.graph import GraphNode
from artificial_memory.memory.state_signals import (
    MemoryStateVector,
    StateSignalCalculator,
)

# Latin words (2+ chars) and CJK runs are tokenized separately: CJK has no
# whitespace boundaries, so character bigrams approximate terms.
_LATIN_PATTERN = re.compile(r"[a-z0-9_]{2,}")
_CJK_PATTERN = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+")


def tokenize(text: str) -> set[str]:
    """Deterministic token set for lexical overlap (Latin words + CJK bigrams)."""
    lowered = text.lower()
    tokens: set[str] = set(_LATIN_PATTERN.findall(lowered))
    for run in _CJK_PATTERN.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
            continue
        tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass
class EvidenceRankConfig:
    """Weights and thresholds for evidence scoring."""

    lexical_weight: float = 0.40
    graph_weight: float = 0.25
    state_weight: float = 0.25
    temporal_weight: float = 0.10
    contradiction_penalty: float = 0.35
    non_current_penalty: float = 0.30
    # Below this lexical score a candidate is only kept when it was reached
    # over a strong graph path (distributed evidence).
    lexical_floor: float = 0.02
    min_score: float = 0.0
    max_items: int = 20
    include_non_current: bool = False

    def __post_init__(self) -> None:
        weight_names = [
            "lexical_weight",
            "graph_weight",
            "state_weight",
            "temporal_weight",
            "contradiction_penalty",
            "non_current_penalty",
            "lexical_floor",
        ]
        for name in weight_names:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if self.max_items <= 0:
            raise ValueError("max_items must be positive")


@dataclass
class EvidenceItem:
    """A ranked candidate contributing to a reconstructed evidence package."""

    memory: Memory
    score: float
    lexical: float = 0.0
    graph: float = 0.0
    state: float = 0.0
    temporal: float = 1.0
    penalty: float = 0.0
    depth: int = 0
    seed_id: int | None = None
    path: list[int] = field(default_factory=list)
    shared_terms: list[str] = field(default_factory=list)
    state_vector: MemoryStateVector | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def memory_id(self) -> int:
        return self.memory.id or 0

    @property
    def is_distributed(self) -> bool:
        """True when the item was reached through an association, not a match."""
        return self.depth > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "score": self.score,
            "lexical": self.lexical,
            "graph": self.graph,
            "state": self.state,
            "temporal": self.temporal,
            "penalty": self.penalty,
            "depth": self.depth,
            "seed_id": self.seed_id,
            "path": list(self.path),
            "shared_terms": list(self.shared_terms),
            "reasons": list(self.reasons),
            "resolution": self.memory.resolution.name,
        }


class EvidenceRanker:
    """Scores graph nodes as evidence for a query."""

    def __init__(
        self,
        store: MemoryStore,
        calculator: StateSignalCalculator | None = None,
        config: EvidenceRankConfig | None = None,
    ):
        self.store = store
        self.calculator = calculator or StateSignalCalculator(store)
        self.config = config or EvidenceRankConfig()

    # ==================== Public API ====================

    def rank(
        self,
        query: str,
        nodes: list[GraphNode],
        now: datetime | None = None,
        keep_ids: set[int] | None = None,
    ) -> list[EvidenceItem]:
        """Rank nodes by combined evidence score, best first.

        Ordering is total and deterministic: ``(-score, memory id)``. Identical
        inputs always produce identical output.

        ``keep_ids`` pins memories that must appear in the package regardless
        of their lexical overlap (explicitly supplied seed chains); all other
        filters still apply to unpinned candidates.
        """
        now = now or datetime.now()
        pinned = keep_ids or set()
        query_tokens = tokenize(query)
        items = [self._score_node(node, query_tokens, now) for node in nodes]
        items = [item for item in items if self._keep(item, pinned)]
        items.sort(key=lambda item: (-item.score, item.memory_id))
        return items[: self.config.max_items]

    def rank_memories(
        self,
        query: str,
        memories: list[Memory],
        now: datetime | None = None,
        keep_ids: set[int] | None = None,
    ) -> list[EvidenceItem]:
        """Convenience wrapper for plain memory lists (no traversal metadata)."""
        nodes = [
            GraphNode(
                memory=memory,
                depth=0,
                seed_id=memory.id or 0,
                is_seed=True,
                path=[memory.id or 0],
            )
            for memory in memories
            if memory.id is not None
        ]
        return self.rank(query, nodes, now=now, keep_ids=keep_ids)

    # ==================== Scoring ====================

    def _score_node(
        self,
        node: GraphNode,
        query_tokens: set[str],
        now: datetime,
    ) -> EvidenceItem:
        memory = node.memory
        lexical, shared = self._lexical_score(memory, query_tokens)
        state_vector = self.calculator.compute_state_vector(memory, now=now)
        graph = node.graph_proximity
        temporal = state_vector.signals.get("temporal", 1.0)

        penalty = 0.0
        reasons: list[str] = []

        if lexical > 0.0:
            reasons.append("lexical match on: " + ", ".join(sorted(shared)[:5]))
        if node.is_seed:
            reasons.append("direct candidate for the query")
        else:
            reasons.append(
                f"reached from memory {node.seed_id} at depth {node.depth} "
                f"(path {' -> '.join(str(p) for p in node.path)})"
            )

        contradiction_risk = state_vector.contradiction_risk
        if contradiction_risk > 0.0:
            penalty += self.config.contradiction_penalty * contradiction_risk
            reasons.append(f"contradiction risk {contradiction_risk:.2f} lowers confidence")

        if not memory.is_current:
            penalty += self.config.non_current_penalty
            reasons.append("superseded / not current")

        if temporal < 1.0:
            reasons.append("outside its validity window at reconstruction time")

        score = (
            self.config.lexical_weight * lexical
            + self.config.graph_weight * graph
            + self.config.state_weight * state_vector.future_utility
            + self.config.temporal_weight * temporal
            - penalty
        )
        reasons.append(f"future utility {state_vector.future_utility:.2f}")

        return EvidenceItem(
            memory=memory,
            score=max(0.0, min(1.0, score)),
            lexical=lexical,
            graph=graph,
            state=state_vector.future_utility,
            temporal=temporal,
            penalty=penalty,
            depth=node.depth,
            seed_id=node.seed_id,
            path=list(node.path),
            shared_terms=sorted(shared),
            state_vector=state_vector,
            reasons=reasons,
        )

    def _lexical_score(self, memory: Memory, query_tokens: set[str]) -> tuple[float, set[str]]:
        """Query-token coverage of the memory (1.0 = every query term present)."""
        if not query_tokens:
            return 0.0, set()
        memory_tokens = tokenize(memory.content)
        shared = query_tokens & memory_tokens
        if not shared:
            return 0.0, set()
        return len(shared) / len(query_tokens), shared

    def _keep(self, item: EvidenceItem, pinned: set[int] | None = None) -> bool:
        # Explicitly pinned memories (caller-supplied seed chains) always stay.
        if pinned and item.memory_id in pinned:
            return True
        if item.score < self.config.min_score:
            return False
        # Superseded memories are conflict context, surfaced by the
        # contradiction check rather than ranked as supporting evidence.
        if not item.memory.is_current and not self.config.include_non_current:
            return False
        # A seed that shares no term with the query is not evidence.
        if item.lexical <= 0.0 and item.depth == 0:
            return False
        return True


def create_evidence_ranker(
    store: MemoryStore,
    calculator: StateSignalCalculator | None = None,
    config: EvidenceRankConfig | None = None,
) -> EvidenceRanker:
    return EvidenceRanker(store, calculator, config)


__all__ = [
    "EvidenceItem",
    "EvidenceRankConfig",
    "EvidenceRanker",
    "create_evidence_ranker",
    "tokenize",
]
