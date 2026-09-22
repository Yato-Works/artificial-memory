"""DeepSeek-inspired recall engine (DeepSeek Raid composition layer).

Composes the raid primitives from
:mod:`artificial_memory.recall.retrieval_cache` with the frozen
:class:`~artificial_memory.recall.engine.BasicRecallEngine` semantics:

- **Retrieval Reuse (CSA2)**: repeated query families reuse the candidate
  pool -- FULL on first sight, REINDEX (re-rank cached pool) on later
  re-phrasings, REUSE (verbatim replay) on exact repeats.
- **Hierarchical Gate (HSI)**: a cheap lexical pre-filter narrows the
  candidate pool before the engine's semantic ranking runs.

The engine is strictly opt-in: ``BasicRecallEngine`` itself is untouched, so
the S0-S3 baseline players keep their pre-raid behaviour and the frozen
Scorer sees identical result shapes.
"""

from __future__ import annotations

from artificial_memory.core.models import Memory, MemoryStatus, RecallEvent, RecallLevel
from artificial_memory.recall.engine import BasicRecallEngine
from artificial_memory.recall.retrieval_cache import (
    FULL,
    REINDEX,
    REUSE,
    HierarchicalGate,
    RetrievalPlanCache,
)

__all__ = ["DeepSeekRecallEngine"]


class DeepSeekRecallEngine(BasicRecallEngine):
    """``BasicRecallEngine`` + CSA2 plan cache + Hierarchical Gate.

    Only the candidate pipeline is overridden.  Ranking, budget selection,
    access-stat updates and recall-event logging are inherited verbatim from
    :class:`BasicRecallEngine`, so results stay scorer-compatible.
    """

    def __init__(
        self,
        store,
        plan_cache: RetrievalPlanCache | None = None,
        gate: HierarchicalGate | None = None,
        candidate_window: int | None = None,
    ):
        super().__init__(store)
        self.plan_cache = plan_cache or RetrievalPlanCache()
        self.gate = gate  # None -> gate disabled (pure plan-cache mode)
        # Candidate window (Phase 8.5 funnel-audit finding): the frozen
        # engine's per-level ``limit`` caps surface only the *most recent*
        # slice of the store (e.g. 50 at level 0), which measured 89% GT loss
        # on the S4 dataset *before* any raid mechanism runs.  ``None`` keeps
        # the frozen behaviour byte-identical; an integer widens the window
        # while preserving each level's resolution-bucket ratios.
        self.candidate_window = candidate_window
        # Gate-config fingerprint for plan-cache flush (see ``recall``).
        self._gate_cfg = None
        # Observability counters for Answer.metadata.
        self.gate_applications = 0
        # Gate-config provenance for plan-cache flushing (Phase 8.11):
        # None until the first recall stamps it.
        self._gate_cfg = None

    # ------------------------------------------------------------------
    # Candidate pipeline override (HSI gate + configurable window live here)
    # ------------------------------------------------------------------
    # Frozen per-level bucket specs, mirrored verbatim from
    # ``BasicRecallEngine._get_candidates``: (resolution, is_current, limit).
    # ``None`` resolution means "no resolution filter" (level 0).
    _LEVEL_BUCKETS = None  # built lazily (models import order safety)

    @classmethod
    def _buckets_for_level(cls, level) -> list[tuple]:
        if cls._LEVEL_BUCKETS is None:
            from artificial_memory.core.models import ResolutionLevel

            R = ResolutionLevel
            cls._LEVEL_BUCKETS = {
                RecallLevel.CURRENT_ONLY: [(None, True, 50)],
                RecallLevel.LONG_TERM_SUMMARY: [
                    (R.SEMANTIC, None, 30),
                    (R.LONG_TERM, None, 30),
                    (R.DEEP_LONG_TERM, None, 30),
                ],
                RecallLevel.EPISODE: [(R.EPISODE, None, 40), (R.SEMANTIC, None, 20)],
                RecallLevel.LIGHT_COMPRESSION: [
                    (R.LIGHT, None, 30), (R.EPISODE, None, 15), (R.SEMANTIC, None, 15),
                ],
                RecallLevel.RAW: [
                    (R.LIGHT, None, 20), (R.EPISODE, None, 20), (R.SEMANTIC, None, 20),
                ],
            }
        return cls._LEVEL_BUCKETS[level]

    def _get_candidates(self, query, topic_id, level) -> list[Memory]:
        """Level-based candidate fetch, optionally narrowed by the gate.

        With ``candidate_window=None`` this is *exactly* the frozen path.
        With a window W, each resolution bucket's fetch limit becomes W.
        Buckets are disjoint partitions of the store (by resolution), so the
        window means "how deep into each partition we look": W >= store size
        recovers every active memory, while the frozen ratio-scaling (e.g.
        450 * 20/60 = 150 per bucket) would still hide older evidence in a
        100%-semantic store (Phase 8.5 funnel-audit finding).
        """
        if self.candidate_window is None:
            candidates = super()._get_candidates(query, topic_id, level)
        else:
            candidates: list[Memory] = []
            for resolution, is_current, _frozen_limit in self._buckets_for_level(level):
                candidates.extend(self.store.get_memories(
                    topic_id=topic_id,
                    resolution=resolution,
                    is_current=is_current,
                    status=MemoryStatus.ACTIVE,
                    limit=self.candidate_window,
                ))
            # Deduplicate by memory ID (frozen-path parity).
            seen: set[int] = set()
            candidates = [m for m in candidates if m.id not in seen and not seen.add(m.id)]

        if self.gate is not None and candidates:
            candidates = self.gate.filter(query, candidates)
            self.gate_applications += 1
        return candidates

    # ------------------------------------------------------------------
    # Public recall with CSA2 tiering
    # ------------------------------------------------------------------
    def recall(
        self,
        query: str,
        topic_id: int | None = None,
        level: RecallLevel = RecallLevel.CURRENT_ONLY,
        max_tokens: int = 4000,
    ) -> tuple[list[Memory], int]:
        """Recall with FULL/REINDEX/REUSE tiering (CSA2 analogue).

        FULL runs the standard engine path exactly once (access stats,
        recall events included).  REINDEX re-ranks the cached candidate pool
        under the new query text.  REUSE replays the prior selection
        verbatim.  Both hit paths still touch access stats and log a recall
        event so downstream store behaviour stays identical.
        """
        # The gate is part of the *plan's* provenance: a plan seeded by a run
        # with floor=F replaying under a different gate config would serve
        # candidates the current gate would have rejected.  Flushing on any
        # gate-parameter change keeps plans honest (Phase 8.11 finding: the
        # floor lever measured as a no-op for 60 runs because the plan cache
        # kept replaying floor=0.0 pools).
        gate_cfg = None
        if self.gate is not None:
            gate_cfg = (
                type(self.gate).__name__,
                self.gate.pool_size,
                getattr(self.gate, "max_sessions", None),
                getattr(self.gate, "score_floor", None),
                getattr(self.gate, "abstention_threshold", None),
                getattr(self.gate, "session_aggregation", None),
                getattr(self.gate, "aggregation_k", None),
            )
        if gate_cfg != self._gate_cfg:
            self.plan_cache = RetrievalPlanCache(
                max_families=self.plan_cache.max_families,
                ttl_seconds=self.plan_cache.ttl_seconds,
            )
            self._gate_cfg = gate_cfg

        fresh_result: dict[str, tuple[list[Memory], int]] = {}

        def _run_fresh(q: str) -> tuple[list[int], list[float]]:
            # Standard engine path WITH side effects (touch + log), executed
            # exactly once per FULL miss.  The returned ids seed the plan.
            memories, tokens = super(DeepSeekRecallEngine, self).recall(
                q, topic_id, level, max_tokens
            )
            fresh_result["result"] = (memories, tokens)
            return [m.id for m in memories], []

        plan, mode = self.plan_cache.lookup_or_plan(query, _run_fresh)

        if mode == FULL:
            # The closure already ran the standard path; replay its result.
            return fresh_result["result"]

        if mode == REINDEX:
            # Re-rank the cached candidate pool under the new query text.
            memories = self._memories_from_ids(list(plan.candidate_ids))
            memories = self._rank_memories(query, memories)
            selected, tokens = self._select_within_budget(memories, max_tokens)
            self._post_recall(query, topic_id, level, selected, tokens)
            return selected, tokens

        # REUSE: verbatim replay of the prior selection (still ordered).
        memories = self._memories_from_ids(list(plan.candidate_ids))
        tokens = sum(max(1, len(m.content) // 3) for m in memories)
        self._post_recall(query, topic_id, level, memories, tokens)
        return memories, tokens

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _memories_from_ids(self, ids: list[int]) -> list[Memory]:
        """Fetch memories in the plan's stored order (content always fresh)."""
        memories: list[Memory] = []
        for memory_id in ids:
            memory = self.store.get_memory(memory_id)
            if memory is not None:
                memories.append(memory)
        return memories

    def _post_recall(
        self,
        query: str,
        topic_id: int | None,
        level: RecallLevel,
        selected: list[Memory],
        tokens: int,
    ) -> None:
        """Touch access stats + log a recall event (parity with FULL path)."""
        import time as _time

        start = _time.time()
        for memory in selected:
            memory.touch()
            self.store.update_memory(memory)
        self.store.log_recall(RecallEvent(
            query=query,
            topic_id=topic_id,
            recall_level=level,
            memories_retrieved=len(selected),
            tokens_returned=tokens,
            latency_ms=int((_time.time() - start) * 1000),
        ))
