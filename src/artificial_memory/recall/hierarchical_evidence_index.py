"""Bounded deterministic retrieval for very large memory collections.

The index is deliberately lexical and provenance-preserving: it performs no
write-side model call and never rewrites a memory.  Records are partitioned
into small chronological shards, then a query visits only matching shards and
the adjacent turns required to resolve conversational references such as
``"I made it yesterday"``.

The class is an in-process index.  A production store can persist the same
three structures (shard manifest, postings, ordered record payloads) in
SQLite/Postgres/FTS without changing its retrieval contract.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from artificial_memory.core.ir.structured import StructuredIR


@dataclass(frozen=True)
class RetrievalTrace:
    """Auditable bounds for one retrieval operation."""

    indexed_records: int
    indexed_shards: int
    matched_shards: int
    visited_shards: int
    seed_records: int
    returned_records: int


class HierarchicalEvidenceIndex:
    """Two-stage inverted index with deterministic local-turn expansion.

    Query work is bounded by ``max_shards * shard_size`` rather than the total
    number of memories.  This gives the runtime a predictable retrieval
    budget at 10M records while preserving enough neighbouring context for
    pronouns, ellipsis, and assistant/user turn pairs.
    """

    _TOKEN_RE = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b")
    _SESSION_RE = re.compile(r"\[([^\]\s]+)(?:\s+on\s+[^\]]+)?\]")
    _STOP_WORDS = frozenset({
        "a", "an", "and", "are", "at", "be", "been", "by", "can", "did", "do", "does",
        "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me", "my",
        "of", "on", "or", "our", "that", "the", "their", "them", "then", "there", "these",
        "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who",
        "why", "will", "with", "would", "you", "your",
    })

    def __init__(self, shard_size: int = 256, max_posting_shards: int = 4_096) -> None:
        if shard_size < 8:
            raise ValueError("shard_size must be at least 8")
        if max_posting_shards < 1:
            raise ValueError("max_posting_shards must be positive")
        self.shard_size = shard_size
        self.max_posting_shards = max_posting_shards
        self._records: list[StructuredIR] = []
        self._record_terms: list[frozenset[str]] = []
        self._record_sessions: list[str] = []
        self._record_positions: list[int] = []
        self._session_records: dict[str, list[int]] = defaultdict(list)
        self._shard_records: dict[tuple[str, int], list[int]] = defaultdict(list)
        self._shard_terms: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
        self._postings: dict[str, set[tuple[str, int]]] = defaultdict(set)
        self._source_fingerprint: tuple[int, str] | None = None
        self.last_trace: RetrievalTrace | None = None

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def shard_count(self) -> int:
        return len(self._shard_records)

    @staticmethod
    def _stem(token: str) -> str:
        token = token.lower()
        if len(token) > 5 and token.endswith("ing"):
            root = token[:-3]
            return root + "e" if root.endswith("k") else root
        if len(token) > 4 and token.endswith("ied"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    @classmethod
    def _terms(cls, text: str) -> frozenset[str]:
        return frozenset(
            cls._stem(token)
            for token in cls._TOKEN_RE.findall(text.lower())
            if len(token) > 2 and token.lower() not in cls._STOP_WORDS
        )

    @classmethod
    def _session_id(cls, record: StructuredIR, fallback: int) -> str:
        match = cls._SESSION_RE.search(record.raw_content or "")
        if match:
            return match.group(1)
        # Entity provides a stable locality for records without conversation
        # provenance.  The final fallback deliberately has bounded shard size.
        entity = (record.entity or "").strip().lower()
        return f"entity:{entity}" if entity else f"batch:{fallback // 256}"

    _FULL_DIGEST_MAX_RECORDS = 100_000

    @classmethod
    def _fingerprint(cls, records: Sequence[StructuredIR]) -> tuple[int, str]:
        """Content digest that survives object-id reuse across queries.

        ``id()``-based fingerprints are unsafe: a released corpus can be
        reallocated at the same address, making a stale index look like a
        match for a different question's records.

        Up to ``_FULL_DIGEST_MAX_RECORDS`` records the digest covers every
        raw_content exactly (cheap relative to the LLM query it guards).
        Beyond that, a deterministic strided sample keeps ``matches`` O(1)
        at 10M scale, where callers manage the index lifecycle explicitly
        via ``prepare_scalable_retrieval``.
        """
        if not records:
            return (0, "")
        digest = hashlib.blake2b(digest_size=16)
        count = len(records)
        if count <= cls._FULL_DIGEST_MAX_RECORDS:
            for record in records:
                digest.update((record.raw_content or "").encode("utf-8", "replace"))
        else:
            stride = count // 1024 + 1
            for idx in range(0, count, stride):
                digest.update((records[idx].raw_content or "").encode("utf-8", "replace"))
        return (count, digest.hexdigest())

    def matches(self, records: Sequence[StructuredIR]) -> bool:
        """Return whether this index was built for this exact record sequence."""
        return self._source_fingerprint == self._fingerprint(records)

    def clear(self) -> None:
        self.__init__(
            shard_size=self.shard_size,
            max_posting_shards=self.max_posting_shards,
        )

    def build(self, records: Sequence[StructuredIR]) -> None:
        """Build a fresh index.  Call once at ingestion, not once per query."""
        self.clear()
        self._source_fingerprint = self._fingerprint(records)
        self.extend(records)

    def extend(self, records: Iterable[StructuredIR]) -> None:
        """Append an ingestion batch without rebuilding existing postings."""
        for record in records:
            idx = len(self._records)
            session = self._session_id(record, idx)
            session_order = len(self._session_records[session])
            shard = (session, session_order // self.shard_size)
            terms = self._terms(" ".join(filter(None, [record.entity, record.property, record.value, record.raw_content])))

            self._records.append(record)
            self._record_terms.append(terms)
            self._record_sessions.append(session)
            self._record_positions.append(session_order)
            self._session_records[session].append(idx)
            self._shard_records[shard].append(idx)
            self._shard_terms[shard].update(terms)
            for term in terms:
                self._postings[term].add(shard)

    def retrieve(
        self,
        query: str,
        *,
        max_shards: int = 16,
        max_records: int = 96,
        adjacency_radius: int = 1,
    ) -> list[StructuredIR]:
        """Return a bounded, provenance-preserving local evidence pool."""
        if not self._records or max_shards < 1 or max_records < 1:
            self.last_trace = RetrievalTrace(self.record_count, self.shard_count, 0, 0, 0, 0)
            return []

        query_terms = self._terms(query)
        shard_scores: Counter[tuple[str, int]] = Counter()
        matched_shards: set[tuple[str, int]] = set()
        total_shards = max(1, self.shard_count)
        posting_lengths = {
            term: len(self._postings[term])
            for term in query_terms
            if term in self._postings
        }
        # Do not walk a posting list that is wider than the configured bound.
        # At 10M memories, terms like "meeting" can occur in millions of
        # shards.  Rare anchor terms form the routing layer; common terms are
        # still scored inside selected shards.  If every term is common, use
        # only the least common one and cap its deterministic prefix.  This
        # preserves a hard latency ceiling and makes a low-confidence route
        # observable through ``last_trace`` rather than silently scanning all
        # memory.
        routable_terms = [
            term for term, count in posting_lengths.items()
            if count <= self.max_posting_shards
        ]
        if not routable_terms and posting_lengths:
            least_common = min(posting_lengths, key=lambda term: (posting_lengths[term], term))
            routable_terms = [least_common]

        for term in routable_terms:
            postings = self._postings.get(term, set())
            if not postings:
                continue
            # Rare terms choose shards; common terms merely help rank within a
            # selected shard.  This is the key bounded-work scale mechanism.
            idf = math.log((total_shards + 1) / (len(postings) + 1)) + 1.0
            # Sets are unordered.  Sorting is used only for the uncommon
            # all-common fallback, producing a reproducible capped route.
            shard_iter = postings
            if len(postings) > self.max_posting_shards:
                shard_iter = sorted(postings)[:self.max_posting_shards]
            for shard in shard_iter:
                shard_scores[shard] += idf
                matched_shards.add(shard)

        selected_shards = [
            shard for shard, _ in sorted(
                shard_scores.items(), key=lambda item: (-item[1], item[0])
            )[:max_shards]
        ]
        if not selected_shards:
            self.last_trace = RetrievalTrace(
                self.record_count, self.shard_count, 0, 0, 0, 0
            )
            return []

        seed_scores: list[tuple[float, int]] = []
        for shard in selected_shards:
            for idx in self._shard_records[shard]:
                overlap = len(query_terms & self._record_terms[idx])
                if overlap:
                    # A small length-normalisation keeps verbose turns from
                    # winning solely because they contain more common words.
                    score = overlap / math.sqrt(max(1, len(self._record_terms[idx])))
                    seed_scores.append((score, idx))

        seed_ids = [idx for _, idx in sorted(seed_scores, key=lambda item: (-item[0], item[1]))]
        selected_ids: list[int] = []
        seen: set[int] = set()
        for seed_id in seed_ids:
            session = self._record_sessions[seed_id]
            ordered = self._session_records[session]
            position = self._record_positions[seed_id]
            for neighbour_pos in range(
                max(0, position - adjacency_radius),
                min(len(ordered), position + adjacency_radius + 1),
            ):
                candidate_id = ordered[neighbour_pos]
                if candidate_id not in seen:
                    seen.add(candidate_id)
                    selected_ids.append(candidate_id)
                    if len(selected_ids) >= max_records:
                        break
            if len(selected_ids) >= max_records:
                break

        self.last_trace = RetrievalTrace(
            indexed_records=self.record_count,
            indexed_shards=self.shard_count,
            matched_shards=len(matched_shards),
            visited_shards=len(selected_shards),
            seed_records=len(seed_ids),
            returned_records=len(selected_ids),
        )
        return [self._records[idx] for idx in selected_ids]
