from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artificial_memory.core.interfaces import MemoryStore, RecallEngine
from artificial_memory.core.models import Memory, RecallLevel, ResolutionLevel


@dataclass
class UtilityScore:
    """Utility score for a memory candidate."""
    memory_id: int
    relevance: float      # Query relevance (0-1)
    importance: float     # Memory importance (0-1)
    confidence: float     # Memory confidence (0-1)
    recency: float        # Recency factor (0-1)
    resolution_bonus: float  # Resolution appropriateness (0-1)
    token_cost: int       # Estimated tokens
    utility: float = 0.0  # Final utility = relevance / token_cost * weights

    def compute(self, weights: dict[str, float]) -> float:
        """Compute utility score with configurable weights."""
        # Normalize token cost (lower is better)
        token_efficiency = 1.0 / (1.0 + self.token_cost / 100)

        self.utility = (
            weights.get("relevance", 0.4) * self.relevance +
            weights.get("importance", 0.2) * self.importance +
            weights.get("confidence", 0.15) * self.confidence +
            weights.get("recency", 0.1) * self.recency +
            weights.get("resolution", 0.1) * self.resolution_bonus +
            weights.get("token_efficiency", 0.05) * token_efficiency
        )
        return self.utility


@dataclass
class RecallBudget:
    """Token budget with utility-based allocation."""
    max_tokens: int
    weights: dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.4,
        "importance": 0.2,
        "confidence": 0.15,
        "recency": 0.1,
        "resolution": 0.1,
        "token_efficiency": 0.05,
    })
    min_utility_threshold: float = 0.1

    def __post_init__(self):
        # Ensure weights sum to 1
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total


class AdaptiveRecallEngine:
    """Adaptive recall engine with utility-based token allocation.

    Instead of fixed recall levels, computes utility for each candidate
    and selects the optimal set within token budget.
    """

    def __init__(
        self,
        store: MemoryStore,
        base_recall_engine: RecallEngine,
        token_estimator=None,
    ):
        self.store = store
        self.base_recall = base_recall_engine
        self.token_estimator = token_estimator or self._default_token_estimator

    def _default_token_estimator(self, text: str) -> int:
        return max(1, len(text) // 3)

    def adaptive_recall(
        self,
        query: str,
        topic_id: int | None = None,
        budget: RecallBudget | None = None,
        max_tokens: int = 4000,
    ) -> tuple[list[Memory], int, list[UtilityScore]]:
        """Adaptive recall: select memories by utility within token budget."""
        budget = budget or RecallBudget(max_tokens=max_tokens)

        # Get large candidate pool from multiple recall levels
        candidates = self._get_candidate_pool(query, topic_id)

        # Score each candidate
        scored = self._score_candidates(query, candidates, budget.weights)

        # Filter by minimum utility
        scored = [s for s in scored if s.utility >= budget.min_utility_threshold]

        # Sort by utility descending
        scored.sort(key=lambda x: -x.utility)

        # Greedy selection within token budget
        selected, total_tokens = self._select_within_budget(scored, budget.max_tokens)

        return selected, total_tokens, scored

    def _get_candidate_pool(
        self,
        query: str,
        topic_id: int | None,
    ) -> list[Memory]:
        """Get candidates from multiple recall levels."""
        candidates = []
        seen_ids = set()

        # Level 0: Current only
        current = self.base_recall.recall(query, topic_id, RecallLevel.CURRENT_ONLY, max_tokens=2000)
        for m in current[0]:
            if m.id not in seen_ids:
                candidates.append(m)
                seen_ids.add(m.id)

        # Level 1: Long-term summary
        lts = self.base_recall.recall(query, topic_id, RecallLevel.LONG_TERM_SUMMARY, max_tokens=2000)
        for m in lts[0]:
            if m.id not in seen_ids:
                candidates.append(m)
                seen_ids.add(m.id)

        # Level 2: Episode
        ep = self.base_recall.recall(query, topic_id, RecallLevel.EPISODE, max_tokens=3000)
        for m in ep[0]:
            if m.id not in seen_ids:
                candidates.append(m)
                seen_ids.add(m.id)

        # Level 3: Light compression
        lc = self.base_recall.recall(query, topic_id, RecallLevel.LIGHT_COMPRESSION, max_tokens=3000)
        for m in lc[0]:
            if m.id not in seen_ids:
                candidates.append(m)
                seen_ids.add(m.id)

        return candidates

    def _score_candidates(
        self,
        query: str,
        candidates: list[Memory],
        weights: dict[str, float],
    ) -> list[UtilityScore]:
        """Score each candidate memory."""
        query_words = set(query.lower().split())
        now = datetime.now()

        scored = []

        for mem in candidates:
            # Relevance: word overlap with query
            mem_words = set(mem.content.lower().split())
            overlap = len(query_words & mem_words)
            relevance = min(1.0, overlap / max(1, len(query_words)))

            # Importance: from memory
            importance = mem.importance

            # Confidence: from memory
            confidence = mem.confidence

            # Recency: exponential decay
            days_old = (now - mem.created_at).days
            recency = max(0.1, 1.0 - days_old / 180)  # Half-life ~180 days

            # Resolution bonus: prefer appropriate resolution
            # For recall, we want SEMANTIC/LIGHT for overview, RAW/LIGHT for detail
            # This depends on query type - simplified here
            resolution_bonus = self._resolution_bonus(mem.resolution)

            # Token cost
            token_cost = self.token_estimator(mem.content)

            score = UtilityScore(
                memory_id=mem.id,
                relevance=relevance,
                importance=importance,
                confidence=confidence,
                recency=recency,
                resolution_bonus=resolution_bonus,
                token_cost=token_cost,
            )
            score.compute(weights)
            scored.append(score)

        return scored

    def _resolution_bonus(self, resolution: ResolutionLevel) -> float:
        """Bonus for appropriate resolution levels."""
        # Prefer SEMANTIC for most queries, LIGHT for detail
        bonuses = {
            ResolutionLevel.RAW: 0.3,
            ResolutionLevel.LIGHT: 0.8,
            ResolutionLevel.EPISODE: 0.7,
            ResolutionLevel.SEMANTIC: 1.0,
            ResolutionLevel.LONG_TERM: 0.6,
            ResolutionLevel.DEEP_LONG_TERM: 0.3,
        }
        return bonuses.get(resolution, 0.5)

    def _select_within_budget(
        self,
        scored: list[UtilityScore],
        max_tokens: int,
    ) -> tuple[list[Memory], int]:
        """Greedy selection by utility/token ratio."""
        selected = []
        total_tokens = 0
        selected_ids = set()

        for score in scored:
            if total_tokens + score.token_cost <= max_tokens:
                mem = self.store.get_memory(score.memory_id)
                if mem and mem.id not in selected_ids:
                    selected.append(mem)
                    selected_ids.add(mem.id)
                    total_tokens += score.token_cost
            else:
                # Try to fit a compressed version
                mem = self.store.get_memory(score.memory_id)
                if mem:
                    # Check if we have a lower resolution version
                    for res in [ResolutionLevel.LIGHT, ResolutionLevel.SEMANTIC,
                               ResolutionLevel.LONG_TERM, ResolutionLevel.DEEP_LONG_TERM]:
                        if res.value > mem.resolution.value:
                            version = self.store.get_memory_version(mem.id, res)
                            if version:
                                ver_tokens = self.token_estimator(version.content)
                                if total_tokens + ver_tokens <= max_tokens:
                                    # Create temporary memory with compressed content
                                    from artificial_memory.core.models import Memory
                                    temp_mem = Memory(
                                        id=mem.id,
                                        topic_id=mem.topic_id,
                                        memory_type=mem.memory_type,
                                        content=version.content,
                                        resolution=res,
                                        importance=mem.importance,
                                        confidence=mem.confidence,
                                        status=mem.status,
                                        valid_from=mem.valid_from,
                                        valid_until=mem.valid_until,
                                        is_current=mem.is_current,
                                        source_conversation_id=mem.source_conversation_id,
                                        source_message_id=mem.source_message_id,
                                        created_at=mem.created_at,
                                        updated_at=mem.updated_at,
                                        last_accessed=mem.last_accessed,
                                        access_count=mem.access_count,
                                    )
                                    selected.append(temp_mem)
                                    selected_ids.add(mem.id)
                                    total_tokens += ver_tokens
                                    break

        return selected, total_tokens

    def explain_selection(
        self,
        query: str,
        topic_id: int | None,
        budget: RecallBudget | None = None,
    ) -> dict[str, Any]:
        """Explain why certain memories were selected."""
        budget = budget or RecallBudget(max_tokens=4000)

        candidates = self._get_candidate_pool(query, topic_id)
        scored = self._score_candidates(query, candidates, budget.weights)
        scored.sort(key=lambda x: -x.utility)

        selected, total_tokens = self._select_within_budget(scored, budget.max_tokens)
        selected_ids = {m.id for m in selected}

        explanation = {
            "query": query,
            "budget": budget.max_tokens,
            "weights": budget.weights,
            "candidates_considered": len(scored),
            "selected_count": len(selected),
            "total_tokens": total_tokens,
            "selection": [],
            "rejected": [],
        }

        for score in scored:
            mem = self.store.get_memory(score.memory_id)
            if not mem:
                continue

            entry = {
                "memory_id": mem.id,
                "type": mem.memory_type.value,
                "resolution": mem.resolution.name,
                "utility": round(score.utility, 3),
                "relevance": round(score.relevance, 3),
                "importance": round(score.importance, 3),
                "confidence": round(score.confidence, 3),
                "recency": round(score.recency, 3),
                "token_cost": score.token_cost,
                "selected": mem.id in selected_ids,
            }

            if mem.id in selected_ids:
                explanation["selection"].append(entry)
            else:
                explanation["rejected"].append(entry)

        return explanation


def create_adaptive_recall_engine(
    store: MemoryStore,
    base_recall_engine: RecallEngine,
) -> AdaptiveRecallEngine:
    return AdaptiveRecallEngine(store, base_recall_engine)
