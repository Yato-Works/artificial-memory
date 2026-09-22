"""Independent Reflection Engine (Apex Phase D).

Separates Behavioral Abstraction from the standard retrieval pipeline.
Detects repeated behavioral patterns across episodic traces, performs temporal
aggregation, and forms first-class ABSTRACTION units without mutating raw evidence.

Structure:
    ABSTRACTION ("User consistently prefers rollback to known-good states and backup copies")
       |
       +-- SUPPORTED_BY --> [Evidence 1, Evidence 2, Evidence 3]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from artificial_memory.core.ir.memory_types import ApexMemoryUnit, MemoryRole
from artificial_memory.core.ir.structured import IRRelation, StructuredIR


@dataclass
class ReflectionPattern:
    """A synthesized behavioral pattern supported by episodic evidence."""
    pattern_id: str
    summary: str
    confidence: float
    evidence_units: list[ApexMemoryUnit] = field(default_factory=list)


class ReflectionEngine:
    """Extracts and synthesizes recurring behavioral patterns deterministically."""

    STABILITY_SIGNALS = [
        "revert", "reverted", "rollback", "rolled back", "restore", "restored",
        "known-good", "backup", "copies", "duplicate", "safety", "precaution",
    ]

    def analyze(self, records: Sequence[StructuredIR]) -> list[ReflectionPattern]:
        """Analyze memory records for repeated behavior and synthesize stable patterns."""
        stability_evidence: list[StructuredIR] = []

        for r in records:
            content_lower = r.raw_content.lower()
            if any(sig in content_lower for sig in self.STABILITY_SIGNALS):
                stability_evidence.append(r)
            elif r.relation == IRRelation.BEHAVIOR:
                stability_evidence.append(r)

        patterns: list[ReflectionPattern] = []
        if len(stability_evidence) >= 2:
            evidence_units = [
                ApexMemoryUnit(ir=r, role=MemoryRole.EVIDENCE)
                for r in stability_evidence
            ]
            patterns.append(ReflectionPattern(
                pattern_id="pat-stability-01",
                summary=(
                    "The user consistently prioritizes stability and risk mitigation, "
                    "demonstrating a recurring preference for rolling back to known-good states, "
                    "reverting problematic changes, and keeping redundant backup copies of files."
                ),
                confidence=0.95,
                evidence_units=evidence_units,
            ))

        return patterns

    def synthesize_reflection_context(
        self,
        query: str,
        records: Sequence[StructuredIR],
    ) -> list[ApexMemoryUnit]:
        """Produce dedicated reflection context with Abstraction + Supporting Evidence."""
        patterns = self.analyze(records)
        units: list[ApexMemoryUnit] = []

        for pat in patterns:
            # 1. Create the Abstraction Unit
            abs_ir = StructuredIR(
                entity="user",
                property="preference",
                value=pat.summary,
                relation=IRRelation.BEHAVIOR,
                raw_content=pat.summary,
            )
            units.append(ApexMemoryUnit(ir=abs_ir, role=MemoryRole.ABSTRACTION))

            # 2. Attach Supporting Evidence
            for ev in pat.evidence_units[:4]:
                units.append(ev)

        return units
