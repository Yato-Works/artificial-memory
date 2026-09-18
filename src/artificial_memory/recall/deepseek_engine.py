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

from artificial_memory.core.models import Memory, RecallEvent, RecallLevel
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
    ):
        super().__init__(store)
        self.plan_cache = plan_cache or RetrievalPlanCache()
        self.gate = gate  # None -> gate disabled (pure plan-cache mode)
        # Observability counters for Answer.metadata.
        self.gate_applications = 0

    # ------------------------------------------------------------------
    # Candidate pipeline override (HSI gate lives here)
    # ------------------------------------------------------------------
    def _get_candidates(self, query, topic_id, level) -> list[Memory]:
        """Level-based candidate fetch, optionally narrowed by the gate."""
        candidates = super()._get_candidates(query, topic_id, level)
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
