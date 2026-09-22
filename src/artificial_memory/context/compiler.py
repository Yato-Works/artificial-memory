"""Cognitive Context Compiler (Phase 6-B).

Implements true Context Compilation rather than heuristic context engineering:
StructuredIR -> Entity Scope -> Temporal Scope -> Truth/Evidence Separation ->
Conflict Annotation -> Current-State Prioritization -> Minimal Context Output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from artificial_memory.core.ir.conflict_state import ConflictResolutionStatus, ConflictStateManager
from artificial_memory.core.ir.structured import IRRelation, IRStatus, StructuredIR
from artificial_memory.recall.ir_resolver import ResolvedContext, UniversalIRResolver


@dataclass
class CompiledContext:
    """The final compiled context artifact ready for LLM consumption."""
    text: str
    active_truth: str | None = None
    historical_evidence: list[str] = field(default_factory=list)
    has_conflict: bool = False
    is_abstention: bool = False
    token_estimate: int = 0


class CognitiveContextCompiler:
    """Compiles StructuredIR records into an un-misinterpretable context prompt."""

    def __init__(self) -> None:
        self.resolver = UniversalIRResolver()
        self.conflict_mgr = ConflictStateManager()

    def compile(
        self,
        query: str,
        records: Sequence[StructuredIR],
    ) -> CompiledContext:
        """Execute the full 6-stage deterministic context compilation pipeline."""
        # 1. Resolve raw query against records via cognitive resolver
        resolved: ResolvedContext = self.resolver.resolve(query, records)
        if resolved.is_abstention:
            return CompiledContext(
                text=resolved.context_text,
                is_abstention=True,
                token_estimate=len(resolved.context_text.split()),
            )

        # 2. Check for active conflicts
        conflicts = self.conflict_mgr.detect_and_register(list(records))
        target_entity = resolved.primary_entity
        target_conflict = None
        if target_entity:
            for c in conflicts:
                if c.entity.lower() == target_entity.lower():
                    target_conflict = c
                    break

        # 3. Separate Active Truth from Historical Evidence
        active_truths: list[str] = []
        historical_evidences: list[str] = []
        superseded_history: list[str] = []

        for r in resolved.matched_records:
            if r.status == IRStatus.ACTIVE:
                if r.relation == IRRelation.MIGRATED:
                    active_truths.append(f"[CURRENT ACTIVE STATE]: {r.entity} {r.property} = {r.value} (migrated from {r.old_value or 'previous'})")
                    if r.old_value:
                        superseded_history.append(f"[HISTORICAL / SUPERSEDED]: {r.entity} {r.property} was {r.old_value}")
                else:
                    active_truths.append(f"[ACTIVE TRUTH]: {r.entity} {r.property} = {r.value}")
            elif r.status == IRStatus.SUPERSEDED or r.status == IRStatus.DEPRECATED:
                superseded_history.append(f"[HISTORICAL / SUPERSEDED]: {r.entity} {r.property} was {r.value}")
            else:
                historical_evidences.append(f"[HISTORICAL EVIDENCE / MENTION]: {r.source} mentioned '{r.value}' for {r.entity}")

        # 4. Assemble compiled prompt with unambiguous section tags
        sections: list[str] = []

        if target_conflict and target_conflict.status == ConflictResolutionStatus.CONFLICTED:
            sections.append(target_conflict.format_for_context_ir())
        elif active_truths:
            sections.extend(active_truths)

        if superseded_history:
            sections.append("--- Historical Context ---")
            sections.extend(superseded_history)

        if historical_evidences:
            sections.append("--- Dialog Mentions ---")
            sections.extend(historical_evidences)

        # Fallback to resolved.context_text if structured sections were empty
        if not sections:
            final_text = resolved.context_text
        else:
            final_text = "\n".join(sections)

        token_est = int(len(final_text.split()) * 1.3)

        return CompiledContext(
            text=final_text,
            active_truth=active_truths[0] if active_truths else None,
            historical_evidence=historical_evidences + superseded_history,
            has_conflict=(target_conflict is not None),
            is_abstention=False,
            token_estimate=token_est,
        )
