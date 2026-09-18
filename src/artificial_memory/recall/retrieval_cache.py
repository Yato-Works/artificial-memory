"""DeepSeek-inspired retrieval acceleration (the "DeepSeek Raid").

Translates three DeepSeek V4.1 systems-level design patterns -- CSA2
cache-mode tiering, the Hierarchical Sparse Indexer, and SWA Bounded Replay
-- into memory-runtime equivalents.  AM retrieves discrete memories, not KV
tensors, so each pattern becomes a retrieval-plan mechanism rather than an
attention kernel.

Pattern map (DeepSeek V4.1 -> AM):

======================  ==============================================
DeepSeek V4.1           AM equivalent
======================  ==============================================
CSA2 ``Full`` layer     fresh search for a new query family
CSA2 ``Reindex`` layer  re-rank the cached candidate pool under a new
                        query text
CSA2 ``Reuse`` layer    replay the prior candidate set unchanged
Hierarchical Sparse     ``HierarchicalGate`` -- cheap pre-filter that
Indexer                 narrows the pool before semantic ranking
SWA Bounded Replay      ``EphemeralStore`` -- discard transient state,
                        rebuild from the source log on demand
======================  ==============================================

Design invariants (frozen with Phase 8.3):

- The frozen Scorer never sees these classes; they are pure Runtime-side
  acceleration and surface their behaviour only through metadata counters.
- ``BasicRecallEngine`` semantics are unchanged when the gate is off, so the
  S0-S3 baseline players and every pre-raid test keep their exact behaviour.
- All caches are process-local and deterministic: identical query streams
  produce identical plans, so a recorded run can be replayed and re-scored
  by the frozen Scorer without invalidation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from artificial_memory.core.models import Memory

# ---------------------------------------------------------------------------
# Retrieval Reuse (CSA2 Full / Reindex / Reuse tiering)
# ---------------------------------------------------------------------------

FULL = "full"
REINDEX = "reindex"
REUSE = "reuse"

# Fresh-retrieval closure type: query -> (candidate_ids, scores)
FreshRetrieval = Callable[[str], "tuple[list[int], list[float]]"]


@dataclass
class RetrievalPlan:
    """One CSA2-style tier decision for a query family."""

    mode: str                       # FULL | REINDEX | REUSE
    query: str
    candidate_ids: tuple[int, ...]
    scores: tuple[float, ...]
    embedding_version: str = "v0.2.0"
    created_at: float = 0.0
    reuse_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "query": self.query,
            "candidate_ids": list(self.candidate_ids),
            "scores": list(self.scores),
            "embedding_version": self.embedding_version,
            "reuse_count": self.reuse_count,
        }


class RetrievalPlanCache:
    """CSA2-style plan cache: FULL -> REINDEX -> REUSE candidate reuse.

    A *query family* is a normalized query signature.  The first query in a
    family pays a full retrieval (``FULL``).  Later queries in the same
    family keep the candidate pool (the "main KV" analogue) and either re-rank
    it under the new query text (``REINDEX``) or replay it verbatim
    (``REUSE``) when the query text is identical.
    """

    def __init__(
        self,
        max_families: int = 64,
        ttl_seconds: float = 300.0,
        embedding_version: str = "v0.2.0",
    ):
        self.max_families = max_families
        self.ttl_seconds = ttl_seconds
        self.embedding_version = embedding_version
        self._families: dict[str, RetrievalPlan] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        # Observability counters (surfaced in Answer.metadata by the caller).
        self.stats: dict[str, int] = {FULL: 0, REINDEX: 0, REUSE: 0}

    # -- query family signature -------------------------------------------
    @staticmethod
    def _signature(query: str) -> str:
        """Deterministic query-family signature.

        Lowercase, strip punctuation from every token ("codename?" and
        "codename" share a family) and collapse whitespace.  Word order is
        kept: "why did X fail" vs "how did X fail" must NOT share a family.
        """
        words = []
        for raw in query.lower().split():
            token = "".join(c for c in raw if c.isalnum())
            if token:
                words.append(token)
        return " ".join(words)

    # -- public API ---------------------------------------------------------
    def lookup_or_plan(
        self,
        query: str,
        run_fresh: FreshRetrieval,
    ) -> tuple[RetrievalPlan, str]:
        """Return ``(plan, mode)`` for ``query``, running ``run_fresh`` on FULL.

        ``run_fresh`` is the expensive retrieval closure; it is only invoked
        for ``FULL`` mode.  On REINDEX/REUSE the cached candidate ids are
        replayed -- no store access, no embedding call.
        """
        sig = self._signature(query)
        now = time.perf_counter()
        with self._lock:
            plan = self._families.get(sig)
            if plan is not None and now - plan.created_at > self.ttl_seconds:
                del self._families[sig]
                self._order.remove(sig)
                plan = None

            if plan is not None:
                # HIT -> REUSE (identical text) or REINDEX (same family).
                mode = REUSE if query == plan.query else REINDEX
                self.stats[mode] += 1
                plan.reuse_count += 1
                return plan, mode

        # MISS -> FULL: run the expensive retrieval outside the lock so
        # concurrent lookups are not serialised behind it.
        ids, scores = run_fresh(query)
        plan = RetrievalPlan(
            mode=FULL,
            query=query,
            candidate_ids=tuple(ids),
            scores=tuple(scores),
            embedding_version=self.embedding_version,
            created_at=time.perf_counter(),
        )
        with self._lock:
            self._families[sig] = plan
            self._order.append(sig)
            self._evict_if_needed()
            self.stats[FULL] += 1
        return plan, FULL

    def _evict_if_needed(self) -> None:
        """Insertion-order eviction (deterministic, no LRU recency heuristic)."""
        while len(self._order) > self.max_families:
            oldest = self._order.pop(0)
            self._families.pop(oldest, None)

    def clear(self) -> None:
        with self._lock:
            self._families.clear()
            self._order.clear()
            self.stats = {FULL: 0, REINDEX: 0, REUSE: 0}

    def stats_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.stats)


# ---------------------------------------------------------------------------
# Hierarchical Retrieval Gate (DeepSeek Hierarchical Sparse Indexer analogue)
# ---------------------------------------------------------------------------


class HierarchicalGate:
    """Cheap lexical pre-filter that narrows the candidate pool before ranking.

    DeepSeek's HSI narrows ~1M tokens to ~16K candidates at the decoder
    entrance; the AM analogue narrows the candidate *pool* with a cheap
    lexical pre-score before the (more expensive) semantic ranking runs on
    the survivors.  Pure function: never mutates memories, never touches the
    store, and keeps an order-stable tie-break (earliest index wins) so a
    recorded run replays identically.
    """

    def __init__(self, pool_size: int = 64, min_keep: int = 8):
        self.pool_size = pool_size
        self.min_keep = min_keep

    def filter(self, query: str, memories: list[Memory]) -> list[Memory]:
        """Narrow ``memories`` to at most ``pool_size`` candidates.

        Scored by word-overlap Jaccard against the query.  Ties are broken
        by the original order (earliest wins), so the gate is deterministic
        for a fixed input list.
        """
        if len(memories) <= self.pool_size:
            return memories  # pool already small: gate is a no-op
        q = set(query.lower().split())
        scored: list[tuple[float, int, Memory]] = []
        for idx, mem in enumerate(memories):
            words = set(mem.content.lower().split())
            union = q | words
            jaccard = (len(q & words) / len(union)) if union else 0.0
            scored.append((jaccard, -idx, mem))
        scored.sort(key=lambda t: (-t[0], t[1]))
        keep = max(self.min_keep, min(self.pool_size, len(memories)))
        return [m for _, _, m in scored[:keep]]

    def gate_stats(self, before: int, after: int) -> dict[str, int]:
        """Observability counters for the caller's metadata."""
        return {"gate_in": before, "gate_out": after}


# ---------------------------------------------------------------------------
# Ephemeral / Replayable state (DeepSeek SWA Bounded Replay analogue)
# ---------------------------------------------------------------------------


class EphemeralStore:
    """Bounded Replay analogue: discard transient state, rebuild on demand.

    DeepSeek found that swapping a small sliding-window KV to storage costs
    more than recomputing it.  The AM equivalent: a session-scoped scratch
    store that is deliberately *not* persisted; when a key is missing it is
    rebuilt from the durable source log through a replay closure.  Nothing
    here ever touches the durable store, so the frozen Scorer's replay of a
    recorded run is unaffected.
    """

    def __init__(
        self,
        replay: Callable[[str], Any] | None = None,
        max_keys: int = 128,
    ):
        self._replay = replay
        self._scratch: dict[str, Any] = {}
        self._order: list[str] = []
        self.max_keys = max_keys
        # Observability counters.
        self.replays = 0
        self.discards = 0

    def get(self, key: str) -> Any | None:
        """Return the cached value, or rebuild it from the source log."""
        if key in self._scratch:
            return self._scratch[key]
        if self._replay is None:
            return None
        self.replays += 1
        value = self._replay(key)
        self.put(key, value)
        return value

    def put(self, key: str, value: Any) -> None:
        """Insert without persisting; bounded (FIFO eviction)."""
        if key in self._scratch:
            return
        self._scratch[key] = value
        self._order.append(key)
        while len(self._order) > self.max_keys:
            evicted = self._order.pop(0)
            self._scratch.pop(evicted, None)
            self.discards += 1

    def discard_all(self) -> None:
        """Bounded Replay core: drop scratch state without persisting."""
        self.discards += len(self._scratch)
        self._scratch.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._scratch)

    def keys(self) -> list[str]:
        return list(self._scratch)

