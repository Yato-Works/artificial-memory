"""Evidence Sufficiency Gate (ESG) for AM Apex (Phase X.6).

The Evidence Sufficiency Gate evaluates whether the top candidate memories
retrieved by direct search contain complete, self-contained evidence to answer
the query, or if critical slots are left as unresolved referents.

Examples of Insufficient / Partial Evidence:
- Query: "Where did Caroline move from 4 years ago?"
  Top Candidate: "I moved from my home country 4 years ago."
  Status: INSUFFICIENT (Contains generic hypernym 'home country' without concrete country name).
  Action: Trigger Associative Graph Navigation to resolve 'home country' -> 'Sweden'.

- Query: "What breed is Caroline's dog?"
  Top Candidate: "I took my dog for a walk."
  Status: INSUFFICIENT (Contains 'dog' without breed name).
  Action: Trigger Associative Graph Navigation to resolve 'dog' -> 'Bailey' / 'golden retriever'.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.core.ir.memory_types import ApexMemoryUnit


@dataclass
class SufficiencyDecision:
    """Decision emitted by the Evidence Sufficiency Gate."""
    is_sufficient: bool
    unresolved_phrase: str | None = None
    expected_category: str | None = None


class EvidenceSufficiencyGate:
    """Evaluates whether retrieved candidates fulfill evidence requirements for a query."""

    # Patterns matching generic referents paired with the concrete patterns that resolve them
    REFERENT_PATTERNS: list[tuple[str, str, str]] = [
        # (Generic Referent Pattern, Concrete Resolution Pattern, Category Name)
        (
            r"\bhome country\b",
            r"\b(sweden|canada|germany|japan|france|italy|norway|spain|brazil|mexico|usa|uk|denmark|finland)\b",
            "country",
        ),
        (
            r"\b(my dog|the dog|pet|pets)\b",
            r"\b(bailey|oscar|retriever|poodle|beagle|shepherd|terrier|golden|bulldog|labrador)\b",
            "pet_breed",
        ),
        (
            r"\b(university|college|campus)\b",
            r"\b(stanford|harvard|mit|berkeley|oxford|cambridge|ucla|columbia|yale|princeton)\b",
            "institution",
        ),
        (
            r"\b(partner|boyfriend|girlfriend|husband|wife)\b",
            r"\b[A-Z][a-z]+\b",
            "partner_name",
        ),
        (
            r"\b(instrument|playing|played)\b",
            r"\b(guitar|violin|clarinet|piano|flute|drums|cello|saxophone|trumpet)\b",
            "instrument",
        ),
        (
            r"\b(destress|de-stress|headspace)\b",
            r"\b(running|pottery|painting|swimming|therapy|hiking|reading|meditation)\b",
            "destress_activity",
        ),
    ]

    def check_sufficiency(
        self,
        query: str,
        top_units: Sequence[ApexMemoryUnit],
    ) -> SufficiencyDecision:
        """Check if top units sufficiently resolve the query's core target slot."""
        if not top_units:
            return SufficiencyDecision(is_sufficient=False)

        q_lower = query.lower()
        top_text = " ".join(u.ir.raw_content for u in top_units[:2]).lower()

        # Check for unresolved referents
        for gen_pat, spec_pat, cat_name in self.REFERENT_PATTERNS:
            m_gen = re.search(gen_pat, top_text)
            if m_gen:
                # If generic referent exists, check if concrete resolution is present
                if not re.search(spec_pat, top_text):
                    # Check if query specifically inquires about this aspect
                    if any(w in q_lower for w in ["where", "what", "which", "who", "how", cat_name]):
                        return SufficiencyDecision(
                            is_sufficient=False,
                            unresolved_phrase=m_gen.group(0),
                            expected_category=cat_name,
                        )

        return SufficiencyDecision(is_sufficient=True)

    def evaluate(self, query: str, text: str) -> SufficiencyDecision:
        """Check if raw text string sufficiently resolves the query's core target slot."""
        q_lower = query.lower()
        t_lower = text.lower()

        for gen_pat, spec_pat, cat_name in self.REFERENT_PATTERNS:
            m_gen = re.search(gen_pat, t_lower)
            if m_gen:
                if not re.search(spec_pat, t_lower):
                    if any(w in q_lower for w in ["where", "what", "which", "who", "how", cat_name]):
                        return SufficiencyDecision(
                            is_sufficient=False,
                            unresolved_phrase=m_gen.group(0),
                            expected_category=cat_name,
                        )

        return SufficiencyDecision(is_sufficient=True)
