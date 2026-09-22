"""Multi-Objective Evidence Ranker for AM Apex Protein Phase (Protein #2).

Computes comprehensive composite relevance score:
    Score(e | q) = w_s * S(e, q) + w_e * E(e, q) + w_t * T(e, q) + w_r * R(e, q) + w_g * G(e, q)

Where:
- S: Semantic & lexical token overlap
- E: Entity & kinship anchor alignment
- T: Temporal scope & recency relevance
- R: Relation & action predicate alignment
- G: Graph proximity / hop distance

Prunes the 30-100 wide candidates down to top 8-12 high-precision evidence units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class ScoredEvidence:
    record: StructuredIR
    score: float
    semantic_score: float
    entity_score: float
    temporal_score: float
    relation_score: float
    graph_score: float


class EvidenceRanker:
    """Multi-objective evidence ranker for high precision."""

    DATE_PATTERN = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")
    ACTION_WORDS = {
        "travel", "visit", "bought", "buy", "graduated", "graduate", "meet", "met",
        "work", "worked", "live", "lived", "move", "moved", "read", "paint", "painted",
        "play", "played", "adopt", "adopted", "book", "booked", "fix", "fixed",
    }

    def __init__(
        self,
        w_s: float = 1.0,
        w_e: float = 2.5,
        w_t: float = 1.5,
        w_r: float = 0.0,  # Frozen: 0.0 eliminates action verb distractors
        w_g: float = 1.0,
        top_k: int = 10,
    ) -> None:
        self.w_s = w_s
        self.w_e = w_e
        self.w_t = w_t
        self.w_r = w_r
        self.w_g = w_g
        self.top_k = top_k

    def rank(
        self,
        query: str,
        records: Sequence[StructuredIR],
        target_entity: str | None = None,
        target_date: str | None = None,
        weights: dict[str, float] | None = None,
        top_k: int | None = None,
    ) -> list[ScoredEvidence]:
        """Rank and return top_k high-precision evidence units."""
        w_s = weights.get("w_s", self.w_s) if weights else self.w_s
        w_e = weights.get("w_e", self.w_e) if weights else self.w_e
        w_t = weights.get("w_t", self.w_t) if weights else self.w_t
        w_r = weights.get("w_r", self.w_r) if weights else self.w_r
        w_g = weights.get("w_g", self.w_g) if weights else self.w_g
        k = top_k if top_k is not None else self.top_k

        q_lower = query.lower()
        STOP_CONV = {
            "can", "you", "could", "would", "suggest", "recommend", "some", "that",
            "what", "which", "how", "about", "tell", "give", "help", "with", "for",
            "the", "and", "are", "have", "been", "was", "were", "any", "does", "did",
            "will", "shall", "should", "might", "must", "may", "there", "this",
            "different", "items", "item", "need", "needs", "needed", "store", "stores",
            "tips", "good", "new", "also", "like", "make", "sure", "start", "started",
            "keep", "kept", "find", "visit", "visited", "visiting",
        }
        content_q_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if len(w) > 2 and w not in STOP_CONV)
        q_words = content_q_words if content_q_words else set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if len(w) > 2)

        # Domain synonyms for multi-session & multi-hop retrieval
        DOMAIN_SYNONYMS = {
            "doctor": {"doctor", "doctors", "dr", "physician", "physicians", "specialist", "specialists", "dermatologist", "ent"},
            "doctors": {"doctor", "doctors", "dr", "physician", "physicians", "specialist", "specialists", "dermatologist", "ent"},
            "clothing": {"clothing", "clothes", "blazer", "boots", "jacket", "jeans", "shirt", "pants", "dress", "sweater"},
            "clothes": {"clothing", "clothes", "blazer", "boots", "jacket", "jeans", "shirt", "pants", "dress", "sweater"},
            "plant": {"plant", "plants", "lily", "succulent", "fern", "basil", "snake"},
            "plants": {"plant", "plants", "lily", "succulent", "fern", "basil", "snake"},
        }
        for qw in list(q_words):
            if qw in DOMAIN_SYNONYMS:
                q_words.update(DOMAIN_SYNONYMS[qw])

        # Extract entities from query if not provided
        if not target_entity:
            if any(w in q_lower.split() for w in ["i", "my", "me", "mine", "myself"]):
                target_entity = "user"
            else:
                STOP_PROPER = {"how", "what", "why", "where", "when", "did", "who", "which", "can", "could", "would", "is", "are", "do", "does"}
                for m in re.finditer(r"\b[A-Z][a-z]+\b", query):
                    if m.group().lower() not in STOP_PROPER:
                        target_entity = m.group().lower()
                        break

        # Extract target date from query if not provided
        if not target_date:
            m_date = self.DATE_PATTERN.search(query)
            if m_date:
                target_date = m_date.group(0)

        scored_items: list[ScoredEvidence] = []

        for r in records:
            content = (r.raw_content or "").lower()
            r_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", content) if len(w) > 2)

            # 1. Semantic Score (Token overlap ratio + stem bonus)
            overlap = len(q_words & r_words)
            stem_bonus = 0.0
            for qw in q_words:
                if len(qw) >= 4 and qw not in r_words:
                    if any(qw[:4] == rw[:4] for rw in r_words if len(rw) >= 4):
                        stem_bonus += 0.5
            s_score = (overlap + stem_bonus) / len(q_words) * 10.0 if q_words else 0.0

            # 2. Entity Score (Direct or alias match + Speaker Priority)
            e_score = 0.0
            r_ent = (r.entity or "").lower()
            r_src = (r.source or "").lower()
            if target_entity:
                if target_entity in ["user", "me", "i"]:
                    if r_src == "user" or (r.raw_content and ": user:" in r.raw_content.lower()):
                        e_score += 15.0
                    elif any(w in content for w in ["i ", "my ", "me ", "mine "]):
                        e_score += 10.0
                elif r_src == target_entity:
                    e_score += 15.0
                elif target_entity in r_ent:
                    e_score += 10.0
                elif target_entity in content:
                    e_score += 4.0

            # 3. Temporal Score (Date match or recency)
            t_score = 0.0
            r_date = r.time_scope or ""
            if target_date:
                if target_date in r_date or target_date in content:
                    t_score += 10.0
            elif any(w in q_lower for w in ["recent", "now", "current", "currently", "latest", "updated"]):
                m_d = self.DATE_PATTERN.search(r_date) or self.DATE_PATTERN.search(content)
                if m_d:
                    year = int(m_d.group(1))
                    month = int(m_d.group(2))
                    t_score += max(0.0, (year - 2020) * 1.5 + (month / 12.0))
                elif any(y in r_date for y in ["2023", "2024", "2025", "2026"]):
                    t_score += 5.0

            # 4. Relation Score (Action match)
            r_score = 0.0
            q_actions = q_words & self.ACTION_WORDS
            r_actions = r_words & self.ACTION_WORDS
            if q_actions and (q_actions & r_actions):
                r_score += 10.0
            elif q_actions and any(a in content for a in q_actions):
                r_score += 8.0

            # 5. Graph Proximity Score (metadata hop distance or default)
            hop = r.metadata.get("hop_distance", 0) if r.metadata else 0
            g_score = 10.0 / (1.0 + hop)

            # 6. Currency & Quantity Relevance Bonus
            is_money_q = any(w in q_lower for w in ["money", "spent", "spend", "cost", "price", "dollar", "$", "expense", "expenses", "fee", "pay", "paid"])
            focus_words = [w for w in ["bike", "hotel", "car", "trip", "plant", "doctor", "clothing", "clothes", "shoe", "shoes"] if w in q_lower]
            if is_money_q:
                has_focus = any(fw in content for fw in focus_words) if focus_words else True
                if has_focus and ("$" in content or "dollar" in content):
                    s_score += 25.0
                elif has_focus and any(w in content for w in ["spent", "cost", "price", "paid"]):
                    s_score += 10.0
            elif any(w in q_lower for w in ["how many", "how much", "total", "count"]):
                if re.search(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", content):
                    s_score += 5.0

            total_score = (
                w_s * s_score +
                w_e * e_score +
                w_t * t_score +
                w_r * r_score +
                w_g * g_score
            )

            scored_items.append(
                ScoredEvidence(
                    record=r,
                    score=total_score,
                    semantic_score=s_score,
                    entity_score=e_score,
                    temporal_score=t_score,
                    relation_score=r_score,
                    graph_score=g_score,
                )
            )

        scored_items.sort(key=lambda x: -x.score)
        return scored_items[:k]
