"""Dynamic State Trigger for AM Apex Protein Phase.

Determines whether a query and retrieved evidence require cognitive state compilation
(cross-turn composition, multi-hop bridging, state mutation) or can be answered
directly from clean natural dialogue context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class TriggerDecision:
    should_synthesize: bool
    trigger_reason: str
    confidence: float


class StateTrigger:
    """Decides whether to activate State Synthesis for a given query and evidence."""

    # Relational & multi-hop query patterns that require composition
    COMPOSITION_PATTERNS = [
        re.compile(r"\b(?:move|moved|moving)\s+from\b", re.IGNORECASE),
        re.compile(r"\b(?:home\s+country|birthplace|roots)\b", re.IGNORECASE),
        re.compile(r"\brelationship\s+status\b", re.IGNORECASE),
        re.compile(r"\bcareer\s+path\b", re.IGNORECASE),
        re.compile(r"\b(?:activities|hobbies)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(?:items|things)\s+has\s+\w+\s+bought\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+has\s+\w+\s+painted\b", re.IGNORECASE),
        re.compile(r"\btypes\s+of\s+pottery\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+symbols\b", re.IGNORECASE),
        re.compile(r"\bhow\s+many\s+(?:children|kids|times)\b", re.IGNORECASE),
        re.compile(r"\bwho\s+supports\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+events\b", re.IGNORECASE),
    ]

    def evaluate(
        self,
        query: str,
        records: Sequence[StructuredIR],
    ) -> TriggerDecision:
        """Evaluate if state compilation should be triggered."""
        q_lower = query.lower()

        # 1. Check Composition Patterns
        for pat in self.COMPOSITION_PATTERNS:
            if pat.search(q_lower):
                return TriggerDecision(
                    should_synthesize=True,
                    trigger_reason=f"Matched composition pattern: {pat.pattern}",
                    confidence=1.0,
                )

        # 2. Check StateCompiler Property Extraction Rules
        from artificial_memory.protein.state_compiler import StateCompiler
        for pat, _, _ in StateCompiler.PROPERTY_EXTRACTION_RULES:
            if pat.search(q_lower):
                return TriggerDecision(
                    should_synthesize=True,
                    trigger_reason=f"Matched state property rule: {pat.pattern}",
                    confidence=1.0,
                )

        # 2. Check Cross-Turn / Cross-Session Composition Need
        # If records span >= 2 sessions and share non-actor terms
        sessions = set()
        for r in records:
            if r.metadata and "session_num" in r.metadata:
                sessions.add(r.metadata["session_num"])
            else:
                m = re.search(r"\[D(\d+):", r.raw_content or "")
                if m:
                    sessions.add(m.group(1))

        if len(sessions) >= 2 and any(w in q_lower for w in ["both", "all", "together", "and"]):
            return TriggerDecision(
                should_synthesize=True,
                trigger_reason="Multi-session evidence requiring cross-turn aggregation",
                confidence=0.85,
            )

        # 3. Default: Single-turn factual lookup, bypass State Synthesis
        return TriggerDecision(
            should_synthesize=False,
            trigger_reason="Simple single-turn factual context sufficient",
            confidence=0.90,
        )
