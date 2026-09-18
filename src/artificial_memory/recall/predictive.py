"""Predictive recall (AM v0.2.0 Phase 6).

Plan #19: prefetching is optional and must never pollute the active
context::

    Current Topic
        ↓ Likely Future Topics      (TopicPredictor)
        ↓ Candidate Memory Retrieval (utility-ranked, bounded)
        ↓ Cache                      (PrefetchCache, TTL + LRU eviction)
        ↓ Context only when needed   (serve on demand, never automatic)

Design notes
------------
* Topic prediction is deterministic and LLM-free. Three signals combine:
  association (graph edges from the current topic into others), recency
  (recent recall events, exponential decay) and keyword overlap with the
  current query.
* The prefetch result lives in the cache only. Memories are handed over
  solely when an actual query arrives ("context only when needed"); the
  caller then budgets them through the Phase 4 ``ContextAllocator``. A cache
  hit therefore cannot inject anything a query did not ask for.
* Everything is bounded: predictions, prefetched topics, memories per topic,
  cache entries and TTL are all configurable, so prefetch pressure cannot
  grow without limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory, RecallLevel, Topic
from artificial_memory.recall.engine import BasicRecallEngine

if TYPE_CHECKING:  # imported lazily at runtime to avoid an import cycle:
    # recall -> memory.__init__ -> memory.compiler -> compiler -> context -> recall
    from artificial_memory.memory.state_signals import StateSignalCalculator


@dataclass
class PredictiveRecallConfig:
    """Bounds for prediction, prefetch and caching."""

    max_predictions: int = 3
    # Prefetch only the strongest predictions (cost control).
    prefetch_topics: int = 2
    max_memories_per_topic: int = 20
    min_probability: float = 0.10
    # Signal weights (each within [0, 1]).
    association_weight: float = 0.50
    recency_weight: float = 0.30
    keyword_weight: float = 0.20
    recency_half_life_days: float = 7.0
    # Cache bounds.
    cache_max_entries: int = 8
    cache_ttl_seconds: float = 3600.0

    def __post_init__(self) -> None:
        for name in (
            "association_weight",
            "recency_weight",
            "keyword_weight",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if self.max_predictions <= 0 or self.prefetch_topics <= 0:
            raise ValueError("max_predictions and prefetch_topics must be positive")
        if self.max_memories_per_topic <= 0:
            raise ValueError("max_memories_per_topic must be positive")
        if not 0.0 <= self.min_probability <= 1.0:
            raise ValueError("min_probability must be within [0, 1]")
        if self.recency_half_life_days <= 0 or self.cache_ttl_seconds <= 0:
            raise ValueError("half-life and TTL must be positive")
        if self.cache_max_entries <= 0:
            raise ValueError("cache_max_entries must be positive")


@dataclass
class TopicPrediction:
    """One predicted future topic with its probability and evidence."""

    topic: Topic
    topic_id: int
    probability: float
    association_score: float = 0.0
    recency_score: float = 0.0
    keyword_score: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic.name,
            "probability": self.probability,
            "association_score": self.association_score,
            "recency_score": self.recency_score,
            "keyword_score": self.keyword_score,
            "evidence": list(self.evidence),
        }


class TopicPredictor:
    """Ranks likely future topics for the current conversation (plan #19)."""

    def __init__(
        self,
        store: MemoryStore,
        config: PredictiveRecallConfig | None = None,
    ):
        self.store = store
        self.config = config or PredictiveRecallConfig()

    def predict(
        self,
        current_topic_id: int,
        query: str | None = None,
        now: datetime | None = None,
    ) -> list[TopicPrediction]:
        """Predict the topics the conversation is likely to move to next."""
        now = now or datetime.now()
        from artificial_memory.memory.evidence import tokenize  # cycle guard

        current_memories = self.store.get_memories(topic_id=current_topic_id, limit=100)
        query_tokens = tokenize(query) if query else set()

        scored: list[tuple[float, int, TopicPrediction]] = []
        for topic in self.store.list_topics():
            topic_id = topic.id
            if topic_id is None or topic_id == current_topic_id:
                continue
            association, assoc_evidence = self._association_score(current_memories, topic_id)
            recency, rec_evidence = self._recency_score(topic_id, now)
            keyword, kw_evidence = self._keyword_score(topic, query_tokens)
            raw = (
                self.config.association_weight * association
                + self.config.recency_weight * recency
                + self.config.keyword_weight * keyword
            )
            if raw <= 0.0:
                continue
            evidence = [item for item in (assoc_evidence, rec_evidence, kw_evidence) if item]
            prediction = TopicPrediction(
                topic=topic,
                topic_id=topic_id,
                probability=raw,
                association_score=association,
                recency_score=recency,
                keyword_score=keyword,
                evidence=evidence,
            )
            scored.append((raw, topic_id, prediction))

        total = sum(raw for raw, _tid, _p in scored)
        if total <= 0.0:
            return []
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        predictions: list[TopicPrediction] = []
        for raw, _tid, prediction in scored:
            probability = raw / total
            if probability < self.config.min_probability and len(predictions) >= 1:
                continue
            prediction.probability = probability
            predictions.append(prediction)
            if len(predictions) >= self.config.max_predictions:
                break
        return predictions

    # ==================== Signals ====================

    def _association_score(
        self,
        current_memories: list[Memory],
        topic_id: int,
    ) -> tuple[float, str]:
        """Graph edges from the current topic's memories into ``topic_id``."""
        total = 0.0
        links = 0
        for memory in current_memories:
            if memory.id is None:
                continue
            for association in self.store.get_associations(memory.id):
                other_id = (
                    association.target_memory_id
                    if association.source_memory_id == memory.id
                    else association.source_memory_id
                    if association.target_memory_id == memory.id
                    else None
                )
                if other_id is None:
                    continue
                other = self.store.get_memory(other_id)
                if other is None or other.topic_id != topic_id:
                    continue
                total += association.strength
                links += 1
        score = min(1.0, total)
        evidence = f"{links} association link(s) into this topic" if links else ""
        return score, evidence

    def _recency_score(self, topic_id: int, now: datetime) -> tuple[float, str]:
        """Exponential decay from the topic's most recent recall activity."""
        try:
            recalls = self.store.get_recent_recalls(topic_id=topic_id, limit=20)
        except (AttributeError, TypeError):
            recalls = []
        if not recalls:
            return 0.0, ""
        latest = max(recall.created_at for recall in recalls)
        age_days = max(0.0, (now - latest).total_seconds() / 86400.0)
        score = 0.5 ** (age_days / self.config.recency_half_life_days)
        return score, f"recalled {age_days:.1f}d ago"

    def _keyword_score(
        self,
        topic: Topic,
        query_tokens: set[str],
    ) -> tuple[float, str]:
        """Overlap between the query and the topic's own text / memories."""
        from artificial_memory.memory.evidence import tokenize  # cycle guard

        if not query_tokens:
            return 0.0, ""
        texts = [topic.name, topic.description or ""]
        memories = self.store.get_memories(topic_id=topic.id, limit=5)
        texts.extend(memory.content for memory in memories)
        topic_tokens: set[str] = set()
        for text in texts:
            topic_tokens |= tokenize(text)
        shared = query_tokens & topic_tokens
        if not shared:
            return 0.0, ""
        return len(shared) / len(query_tokens), f"shared terms: {', '.join(sorted(shared)[:4])}"

# ==================== Prefetch cache ====================


@dataclass
class PrefetchEntry:
    """One cached prefetch bucket (per topic), never auto-injected."""

    topic_id: int
    memories: list[Memory]
    fetched_at: datetime
    used_count: int = 0

    @property
    def memory_ids(self) -> list[int]:
        return [memory.id or 0 for memory in self.memories]

    def is_expired(self, now: datetime, ttl_seconds: float) -> bool:
        return (now - self.fetched_at).total_seconds() > ttl_seconds


class PrefetchCache:
    """Bounded, TTL'd, deterministically-evicted prefetch cache."""

    def __init__(self, max_entries: int, ttl_seconds: float):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: dict[int, PrefetchEntry] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def put(self, entry: PrefetchEntry) -> None:
        """Insert/refresh an entry, evicting deterministically when full."""
        self._entries[entry.topic_id] = entry
        while len(self._entries) > self.max_entries:
            oldest = min(
                self._entries.values(),
                key=lambda item: (item.fetched_at, item.topic_id),
            )
            del self._entries[oldest.topic_id]
            self.evictions += 1

    def get(self, topic_id: int, now: datetime) -> PrefetchEntry | None:
        entry = self._entries.get(topic_id)
        if entry is None:
            self.misses += 1
            return None
        if entry.is_expired(now, self.ttl_seconds):
            del self._entries[topic_id]
            self.evictions += 1
            self.misses += 1
            return None
        self.hits += 1
        entry.used_count += 1
        return entry

    def peek(self, topic_id: int, now: datetime) -> PrefetchEntry | None:
        """Lookup without counting a hit (used by prefetch refresh logic)."""
        entry = self._entries.get(topic_id)
        if entry is not None and entry.is_expired(now, self.ttl_seconds):
            del self._entries[topic_id]
            self.evictions += 1
            return None
        return entry

    def clear_expired(self, now: datetime) -> int:
        expired = [
            topic_id
            for topic_id, entry in self._entries.items()
            if entry.is_expired(now, self.ttl_seconds)
        ]
        for topic_id in expired:
            del self._entries[topic_id]
            self.evictions += 1
        return len(expired)

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }

# ==================== Predictive recall engine ====================


@dataclass
class PrefetchResult:
    """Outcome of one prefetch pass (cache-only; no context injection)."""

    predictions: list[TopicPrediction] = field(default_factory=list)
    fetched: dict[int, list[int]] = field(default_factory=dict)
    reused: dict[int, list[int]] = field(default_factory=dict)

    @property
    def memory_count(self) -> int:
        return sum(len(ids) for ids in (*self.fetched.values(), *self.reused.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "fetched": {str(topic_id): ids for topic_id, ids in self.fetched.items()},
            "reused": {str(topic_id): ids for topic_id, ids in self.reused.items()},
            "memory_count": self.memory_count,
        }


class PredictiveRecallEngine:
    """Prefetches likely-needed memories into a bounded cache (plan #19).

    The engine **never** injects prefetched memories into the active context.
    ``prefetch()`` fills the cache; ``recall()`` serves from it only when an
    actual query arrives, falling back to the normal recall engine on a miss.
    The caller still budgets whatever comes back through the Phase 4
    ``ContextAllocator`` — that is the "deferred context injection" contract.
    """

    def __init__(
        self,
        store: MemoryStore,
        recall_engine: BasicRecallEngine,
        config: PredictiveRecallConfig | None = None,
        predictor: TopicPredictor | None = None,
        cache: PrefetchCache | None = None,
        calculator: StateSignalCalculator | None = None,
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.config = config or PredictiveRecallConfig()
        self.predictor = predictor or TopicPredictor(store, self.config)
        self.cache = cache or PrefetchCache(
            self.config.cache_max_entries, self.config.cache_ttl_seconds
        )
        if calculator is None:
            # Lazy import (import-cycle guard): see module-level TYPE_CHECKING note.
            from artificial_memory.memory.state_signals import StateSignalCalculator

            calculator = StateSignalCalculator(store)
        self.calculator = calculator

    # ==================== Prefetch ====================

    def prefetch(
        self,
        current_topic_id: int,
        query: str | None = None,
        now: datetime | None = None,
    ) -> PrefetchResult:
        """Fetch likely-future-topic memories into the cache (plan #19)."""
        now = now or datetime.now()
        predictions = self.predictor.predict(current_topic_id, query=query, now=now)
        result = PrefetchResult(predictions=predictions)

        for prediction in predictions[: self.config.prefetch_topics]:
            cached = self.cache.peek(prediction.topic_id, now)
            if cached is not None:
                result.reused[prediction.topic_id] = cached.memory_ids
                continue
            memories = self._top_memories(prediction.topic_id, now)
            if not memories:
                continue
            self.cache.put(
                PrefetchEntry(
                    topic_id=prediction.topic_id,
                    memories=memories,
                    fetched_at=now,
                )
            )
            result.fetched[prediction.topic_id] = [memory.id or 0 for memory in memories]
        return result

    def _top_memories(self, topic_id: int, now: datetime) -> list[Memory]:
        """The most useful current memories of a topic (bounded)."""
        # Lazy import (import-cycle guard): see module-level TYPE_CHECKING note.
        from artificial_memory.memory.state_signals import StateSignalCalculator

        memories = self.store.get_memories(topic_id=topic_id, is_current=True, limit=100)
        if not memories:
            return []
        calculator = self.calculator or StateSignalCalculator(self.store)
        ranked = calculator.rank_by_future_utility(memories)
        return [memory for memory, _utility in ranked[: self.config.max_memories_per_topic]]

    # ==================== Deferred serving ====================

    def recall(
        self,
        query: str,
        current_topic_id: int | None = None,
        now: datetime | None = None,
        level: RecallLevel = RecallLevel.CURRENT_ONLY,
        max_tokens: int = 4000,
    ) -> tuple[list[Memory], str]:
        """Serve memories for a real query: cache first, recall on miss.

        Returns ``(memories, source)`` where source is ``"cache"`` or
        ``"recall_engine"``. This is the only path by which prefetched
        memories can reach a context — an actual query must have arrived.
        """
        now = now or datetime.now()
        if current_topic_id is not None:
            for prediction in self.predictor.predict(current_topic_id, query=query, now=now):
                entry = self.cache.get(prediction.topic_id, now)
                if entry is not None:
                    return entry.memories, "cache"
        _memories, _tokens = self.recall_engine.recall(
            query, topic_id=current_topic_id, level=level, max_tokens=max_tokens
        )
        return _memories, "recall_engine"

    def stats(self) -> dict[str, int]:
        return self.cache.stats()


def create_predictive_recall_engine(
    store: MemoryStore,
    recall_engine: BasicRecallEngine,
    config: PredictiveRecallConfig | None = None,
) -> PredictiveRecallEngine:
    return PredictiveRecallEngine(store, recall_engine, config)


__all__ = [
    "PrefetchCache",
    "PrefetchEntry",
    "PrefetchResult",
    "PredictiveRecallConfig",
    "PredictiveRecallEngine",
    "TopicPrediction",
    "TopicPredictor",
    "create_predictive_recall_engine",
]



