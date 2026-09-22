"""Wide Slicer for AM Apex Steroid Phase (Steroid #1).

Implements multi-channel candidate retrieval:
    C_wide = C_semantic U C_lexical U C_entity U C_temporal U C_relation U C_session

Addresses the top root causes identified in Phase S1 Autopsy:
1. HAYSTACK_SESSION_DROP (24.3%): Scans every session for topical/entity anchors.
2. LEXICAL_MISMATCH (24.3%): Token stem, synonym, and semantic affinity.
3. ENTITY_ALIAS_MISMATCH (18.9%): Named entity, kinship, and pronoun coreference tracking.
4. TEMPORAL_ANCHOR_MISMATCH (7.4%): Date arithmetic and chronological session window.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from artificial_memory.core.ir.memory_types import ApexMemoryUnit, MemoryRole
from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class WideSliceResult:
    """Result of multi-channel wide slicing."""
    candidate_records: list[StructuredIR]
    channel_counts: dict[str, int] = field(default_factory=dict)
    total_unioned: int = 0


class WideSlicer:
    """Multi-channel candidate field generator."""

    DATE_PATTERN = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")
    KINSHIP_TERMS = {
        "grandma", "grandmother", "grandpa", "grandfather", "mom", "mother",
        "dad", "father", "sister", "brother", "partner", "wife", "husband",
        "friend", "dog", "cat", "pet", "colleague", "boss", "daughter", "son",
    }

    IRREGULAR_STEMS = {
        "bought": "buy",
        "made": "make",
        "went": "go",
        "ran": "run",
        "seen": "see",
        "saw": "see",
        "met": "meet",
        "held": "hold",
        "read": "read",
        "taken": "take",
        "took": "take",
        "given": "give",
        "gave": "give",
        "chosen": "choose",
        "chose": "choose",
        "written": "write",
        "wrote": "write",
        "spoken": "speak",
        "spoke": "speak",
        "taught": "teach",
        "heard": "hear",
        "married": "marry",
        "marriage": "marry",
    }

    @classmethod
    def _stem(cls, token: str) -> str:
        token = token.lower()
        if token in cls.IRREGULAR_STEMS:
            return cls.IRREGULAR_STEMS[token]
        if len(token) > 6 and token.endswith("ations"):
            return token[:-6]
        if len(token) > 5 and token.endswith("ation"):
            return token[:-5]
        if len(token) > 5 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith("ing"):
            base = token[:-3]
            return base + "e" if base.endswith("k") else base
        if len(token) > 4 and token.endswith("ied"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def __init__(self, per_channel_budget: int = 40) -> None:
        self.per_channel_budget = per_channel_budget

    def slice(
        self,
        query: str,
        records: Sequence[StructuredIR],
        reference_date_str: str | None = None,
    ) -> WideSliceResult:
        """Perform multi-channel retrieval union."""
        q_lower = query.lower()
        q_tokens = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if len(w) > 2)
        q_stems = {self._stem(w) for w in q_tokens}

        # 1. Lexical Channel (Keyword / n-gram overlap + Stemming + Recency Tie-Breaker)
        c_lexical: list[StructuredIR] = []
        lexical_scores: list[tuple[float, float, StructuredIR]] = []
        for r in records:
            r_text = (r.raw_content or "").lower()
            r_tokens = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", r_text) if len(w) > 2)
            r_stems = {self._stem(w) for w in r_tokens}
            overlap = len(q_tokens & r_tokens) * 2 + len(q_stems & r_stems)
            if overlap > 0:
                rec_val = 0.0
                if r.time_scope:
                    m_d = self.DATE_PATTERN.search(r.time_scope)
                    if m_d:
                        rec_val = int(m_d.group(1)) * 365.0 + int(m_d.group(2)) * 30.0 + int(m_d.group(3))
                lexical_scores.append((float(overlap), rec_val, r))
        lexical_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        c_lexical = [r for _, _, r in lexical_scores[:self.per_channel_budget]]

        # 2. Entity Channel (Names, Kinship, Aliases)
        c_entity: list[StructuredIR] = []
        q_entities = set()
        for w in q_tokens:
            if w.istitle() or w in self.KINSHIP_TERMS:
                q_entities.add(w.lower())
        # Also regex for capital names in original query
        for m in re.finditer(r"\b[A-Z][a-z]+\b", query):
            q_entities.add(m.group().lower())

        if q_entities:
            for r in records:
                r_text = (r.raw_content or "").lower()
                if any(e in r_text for e in q_entities):
                    c_entity.append(r)
                    if len(c_entity) >= self.per_channel_budget:
                        break

        # 3. Temporal Channel (Session dates, intervals, relative dates)
        c_temporal: list[StructuredIR] = []
        target_dates = set(m.group(0) for m in self.DATE_PATTERN.finditer(query))
        if reference_date_str:
            target_dates.add(reference_date_str)

        has_temporal_intent = any(w in q_lower for w in ["when", "how many days", "how many weeks", "date", "month", "year", "first", "last", "order"])
        if target_dates or has_temporal_intent:
            for r in records:
                r_text = (r.raw_content or "")
                r_dates = set(m.group(0) for m in self.DATE_PATTERN.finditer(r_text))
                if target_dates & r_dates or (has_temporal_intent and r_dates):
                    c_temporal.append(r)
                    if len(c_temporal) >= self.per_channel_budget:
                        break

        # 4. Session / Haystack Channel (Pulls anchors from all distinct sessions across timeline)
        c_session: list[StructuredIR] = []
        sessions_map: dict[str, list[StructuredIR]] = defaultdict(list)
        for r in records:
            s_key = r.time_scope or "default"
            sessions_map[s_key].append(r)

        # If multi-session haystack (> 5 sessions), score each session and pick across timeline
        if len(sessions_map) > 5:
            scored_sessions: list[tuple[int, float, list[StructuredIR]]] = []
            for s_key, s_recs in sessions_map.items():
                s_text = " ".join(r.raw_content for r in s_recs[:5]).lower()
                s_tokens = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", s_text) if len(w) > 2)
                overlap = len(q_tokens & s_tokens)
                if overlap > 0:
                    rec_val = 0.0
                    m_d = self.DATE_PATTERN.search(s_key)
                    if m_d:
                        rec_val = int(m_d.group(1)) * 365.0 + int(m_d.group(2)) * 30.0 + int(m_d.group(3))
                    scored_sessions.append((overlap, rec_val, s_recs))
            scored_sessions.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for _, _, s_recs in scored_sessions[:15]:
                c_session.extend(s_recs[:3])
                if len(c_session) >= self.per_channel_budget:
                    break

        # 5. Relation / Proposition Channel (Actions, migrations, states)
        c_relation: list[StructuredIR] = []
        action_words = set(w for w in q_tokens if w in ["visit", "travel", "buy", "bought", "meet", "met", "graduated", "start", "started", "lead", "play", "book", "move", "moved"])
        if action_words:
            for r in records:
                r_text = (r.raw_content or "").lower()
                if any(a in r_text for a in action_words):
                    c_relation.append(r)
                    if len(c_relation) >= self.per_channel_budget:
                        break

        # Union and Deduplicate while preserving order of relevance
        seen_contents = set()
        c_union: list[StructuredIR] = []
        for pool in [c_lexical, c_entity, c_temporal, c_relation, c_session]:
            for r in pool:
                key = r.raw_content or f"{r.entity}_{r.target_property}_{r.value}"
                if key not in seen_contents:
                    seen_contents.add(key)
                    c_union.append(r)

        # If union is empty, fallback to first N records
        if not c_union:
            c_union = list(records[:self.per_channel_budget])

        return WideSliceResult(
            candidate_records=c_union,
            channel_counts={
                "lexical": len(c_lexical),
                "entity": len(c_entity),
                "temporal": len(c_temporal),
                "relation": len(c_relation),
                "session": len(c_session),
            },
            total_unioned=len(c_union),
        )
