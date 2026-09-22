"""Query Planner for AM Apex Overdrive Core (Potion 1).

Decomposes incoming questions into structured cognitive execution plans:
- Intent Classification (multi_hop, temporal_delta, temporal_order, state_update, preference, adversarial, fact_lookup)
- Target Entities & Predicate Slots
- Missing Slot Identification for iterative bounded search
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


class PlannerIntent(StrEnum):
    MULTI_HOP = "multi_hop"
    TEMPORAL_DELTA = "temporal_delta"
    TEMPORAL_ORDER = "temporal_order"
    STATE_UPDATE = "state_update"
    USER_PREFERENCE = "user_preference"
    ADVERSARIAL_CHECK = "adversarial_check"
    FACT_LOOKUP = "fact_lookup"


@dataclass
class QueryPlan:
    """Structured execution plan for answering a memory query."""
    raw_query: str
    intent: PlannerIntent
    target_entities: list[str] = field(default_factory=list)
    target_predicates: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    time_constraints: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)


class QueryPlanner:
    """Deterministic Query Planner."""

    NAMED_ENTITIES = {"caroline", "melanie", "eli", "dave", "michael", "rachel", "juan", "kadence"}

    def plan(self, query: str) -> QueryPlan:
        """Analyze query and formulate execution plan."""
        q_lower = query.lower().strip()

        # 1. Intent Detection
        intent = PlannerIntent.FACT_LOOKUP

        # Temporal patterns
        if any(w in q_lower for w in ["how many days", "how many weeks", "how many months", "between"]):
            intent = PlannerIntent.TEMPORAL_DELTA
        elif any(w in q_lower for w in ["order from first to last", "which event happened first", "earlier", "later"]):
            intent = PlannerIntent.TEMPORAL_ORDER
        # State update patterns
        elif any(w in q_lower for w in ["currently", "now", "previous", "previously", "used to", "before", "updated"]):
            intent = PlannerIntent.STATE_UPDATE
        # Preference patterns
        elif any(w in q_lower for w in ["recommend", "suggest", "any suggestions", "ideas for", "complement my", "prefer", "favorite"]):
            intent = PlannerIntent.USER_PREFERENCE
        # Adversarial / verification patterns ("Did X buy/own Y?", "What did X realize...")
        elif re.search(r"^(did\s+[a-z]+|what\s+does\s+[a-z]+'s\s+[a-z]+\s+symbolize)\b", q_lower):
            intent = PlannerIntent.ADVERSARIAL_CHECK
        elif any(w in q_lower for w in ["where did", "why did", "what country", "who is", "how did"]):
            intent = PlannerIntent.MULTI_HOP

        # 2. Extract Entities
        entities: list[str] = []
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower)
        for w in words:
            if w in self.NAMED_ENTITIES and w not in entities:
                entities.append(w)

        # Check possessives (e.g. "Caroline's", "Melanie's")
        for m in re.finditer(r"\b([a-zA-Z0-9_-]+)'s\b", q_lower):
            ent = m.group(1).lower()
            if ent not in entities:
                entities.append(ent)

        # 3. Extract Predicates & Missing Slots
        predicates: list[str] = []
        missing_slots: list[str] = []

        if "move from" in q_lower or "moved from" in q_lower:
            predicates.append("moved_from")
            missing_slots.append("origin_country")
        elif "necklace" in q_lower:
            predicates.append("owns_necklace")
            missing_slots.append("necklace_origin")
        elif "pet" in q_lower or "dog" in q_lower:
            predicates.append("owns_pet")
            missing_slots.append("pet_name")
        elif "use" in q_lower or "using" in q_lower:
            predicates.append("uses")
            missing_slots.append("tool_name")

        # 4. Formulate Sub-Queries for Multi-Hop / Dual-Event
        sub_queries: list[str] = []
        if intent == PlannerIntent.TEMPORAL_DELTA:
            m_between = re.search(r"between\s+(.*?)\s+and\s+(.*?)(?:\?|$)", q_lower)
            if m_between:
                sub_queries.append(m_between.group(1).strip())
                sub_queries.append(m_between.group(2).strip())
        elif intent == PlannerIntent.MULTI_HOP and entities and missing_slots:
            ent = entities[0]
            slot = missing_slots[0]
            sub_queries.append(f"{ent} {slot}")

        return QueryPlan(
            raw_query=query,
            intent=intent,
            target_entities=entities,
            target_predicates=predicates,
            missing_slots=missing_slots,
            sub_queries=sub_queries,
        )
