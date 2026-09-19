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


# ---------------------------------------------------------------------------
# Session-granular gate (Phase 8.7 scorer-sweep winner)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Shared tokenisation: lowercase, alnum-only, whitespace split."""
    words: list[str] = []
    for raw in text.lower().split():
        token = "".join(c for c in raw if c.isalnum())
        if token:
            words.append(token)
    return words


class SessionGate:
    """Session-granular candidate gate (the Phase 8.7 finding, productised).

    Phase 8.6 established that every *memory-granular* cheap scorer (Jaccard,
    BM25, embedding cosine) destroys 60-86% of the evidence set: the evidence
    for a question is a whole *session* of memories, so per-memory scoring
    always strands part of it.  Phase 8.7 measured the fix: score a memory as
    its session does -- per-session score = MAX member BM25 (one distinctive
    member proves the session is relevant), every member inherits it, with a
    small own-BM25 fraction for intra-session ordering.  Measured 85%
    evidence retention @ K=100 while cutting the pool 4x faster than hybrid
    blending (no embedding pass at all).

    Corpus statistics (IDF, avgdl) come from ``corpus_provider`` -- a
    parameterless callable returning the full memory snapshot -- and are
    rebuilt only when the snapshot grows, so repeated recalls do not
    re-tokenise the store.  If no provider is given the *pool itself* is the
    corpus (deterministic, slightly weaker IDF).
    """

    def __init__(
        self,
        session_by_id: dict[int, int] | None = None,
        pool_size: int = 100,
        own_weight: float = 0.1,
        corpus_provider: Callable[[], list[Memory]] | None = None,
        session_gap_seconds: float = 300.0,
        max_sessions: int | None = None,
    ):
        self._explicit_sessions = session_by_id
        self.pool_size = pool_size
        self.own_weight = own_weight
        self._corpus_provider = corpus_provider
        self.session_gap_seconds = session_gap_seconds
        # Phase 8.10 cost lever: keep only the top-K whole sessions (by
        # session score) instead of every session with a member in the pool.
        # None = keep all (the Phase 8.7/8.9 behaviour).
        self.max_sessions = max_sessions
        # Corpus caches (rebuilt when the snapshot grows).
        self._doc_tokens: dict[int, list[str]] = {}
        self._df: dict[str, int] = {}
        self._avgdl = 0.0
        self._corpus_size = -1
        # Observability counters.
        self.applications = 0
        self.total_in = 0
        self.total_out = 0

    # -- corpus statistics ---------------------------------------------------
    def _ensure_corpus(self, memories: list[Memory]) -> None:
        corpus = self._corpus_provider() if self._corpus_provider else memories
        if len(corpus) == self._corpus_size:
            return  # snapshot unchanged: reuse token/IDF caches
        self._corpus_size = len(corpus)
        from collections import Counter

        self._doc_tokens = {m.id: _tokenize(m.content) for m in corpus}
        df: Counter = Counter()
        for toks in self._doc_tokens.values():
            df.update(set(toks))
        self._df = dict(df)
        self._avgdl = (
            sum(len(t) for t in self._doc_tokens.values()) / len(self._doc_tokens)
            if self._doc_tokens else 0.0
        )

    def _sessions_for(self, memories: list[Memory]) -> dict[int, int]:
        """Explicit map if given; otherwise derive sessions from time gaps."""
        if self._explicit_sessions is not None:
            return self._explicit_sessions
        derived: dict[int, int] = {}
        session = 0
        last_seen = None
        for mem in sorted(memories, key=lambda m: m.created_at):
            created = mem.created_at
            if (
                last_seen is not None
                and (created - last_seen).total_seconds() > self.session_gap_seconds
            ):
                session += 1
            derived[mem.id] = session
            last_seen = created
        return derived

    def _bm25(self, query: str, mem: Memory, k1: float = 1.5, b: float = 0.75) -> float:
        import math

        q_tokens = _tokenize(query)
        doc = self._doc_tokens.get(mem.id)
        if not q_tokens or not doc:
            return 0.0
        tf: dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        dl = len(doc)
        norm = k1 * (1 - b + b * dl / (self._avgdl or 1.0))
        score = 0.0
        n_docs = len(self._doc_tokens)
        for t in set(q_tokens):
            df = self._df.get(t, 0)
            if t not in tf:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf[t] * (k1 + 1)) / (tf[t] + norm)
        return score

    # -- gate interface (matches HierarchicalGate.filter) --------------------
    def filter(self, query: str, memories: list[Memory]) -> list[Memory]:
        """Narrow ``memories`` to at most ``pool_size``, sessions kept whole.

        Deterministic for a fixed input list: order-stable sort (score desc,
        earliest index wins on ties), so a recorded run replays identically.
        """
        self._ensure_corpus(memories)
        self.applications += 1
        self.total_in += len(memories)
        if len(memories) <= self.pool_size:
            self.total_out += len(memories)
            return memories  # pool already small: gate is a no-op

        sessions = self._sessions_for(memories)
        own_scores = [self._bm25(query, m) for m in memories]

        # Session score = MAX member BM25 over pool members.
        session_scores: dict[int, float] = {}
        for mem, own in zip(memories, own_scores):
            sid = sessions.get(mem.id)
            if sid is None:
                continue
            if own > session_scores.get(sid, float("-inf")):
                session_scores[sid] = own

        scored: list[tuple[float, int, Memory]] = []
        for idx, (mem, own) in enumerate(zip(memories, own_scores)):
            s = session_scores.get(sessions.get(mem.id))
            # Sessions in whole blocks first; own_weight*own orders inside a
            # session; ungrouped memories fall back to their own score.
            combined = (s + self.own_weight * own) if s is not None else own
            scored.append((combined, -idx, mem))
        scored.sort(key=lambda t: (-t[0], t[1]))

        if self.max_sessions is not None:
            # Phase 8.10: keep only the top-K whole sessions.  Sessions are
            # ranked by score (desc) with the earliest member index as the
            # deterministic tie-break; every member of a kept session stays,
            # ordered by member score.  Same order-stable replay guarantee.
            first_idx: dict[int, int] = {}
            for idx, mem in enumerate(memories):
                sid = sessions.get(mem.id)
                if sid is not None and sid not in first_idx:
                    first_idx[sid] = idx
            ranked = sorted(
                session_scores.items(),
                key=lambda kv: (-kv[1], first_idx.get(kv[0], len(memories))),
            )
            keep_ids = {sid for sid, _ in ranked[: self.max_sessions]}
            members: dict[int, list[tuple[float, int, Memory]]] = {}
            for idx, (mem, own) in enumerate(zip(memories, own_scores)):
                sid = sessions.get(mem.id)
                if sid in keep_ids:
                    s = session_scores[sid]
                    members.setdefault(sid, []).append(
                        (s + self.own_weight * own, -idx, mem)
                    )
            kept: list[Memory] = []
            for sid, _ in ranked[: self.max_sessions]:
                block = members.get(sid, [])
                block.sort(key=lambda t: (-t[0], t[1]))
                kept.extend(m for _, _, m in block)
            kept = kept[: self.pool_size]
            self.total_out += len(kept)
            return kept

        kept = [m for _, _, m in scored[: self.pool_size]]
        self.total_out += len(kept)
        return kept

    def snapshot(self) -> dict[str, int]:
        """Observability counters for the caller's metadata."""
        return {
            "applications": self.applications,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "corpus_size": self._corpus_size,
        }

