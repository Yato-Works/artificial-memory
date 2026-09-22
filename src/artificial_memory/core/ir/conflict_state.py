"""Conflict State & Explicit Uncertainty Management.

Phase 5-B: Treats contradictions and conflicting historical records not as
fatal errors, but as explicit first-class system states (Stateful Uncertainty).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.core.ir.structured import IRRelation, IRStatus, StructuredIR


class ConflictSeverity(StrEnum):
    """Severity of a detected memory contradiction."""
    HIGH = "high"        # Direct opposing assertions on active state (e.g. A says Redis, B says Postgres)
    MEDIUM = "medium"    # Superseded vs proposed ambiguity (e.g. historical proposal vs current state)
    LOW = "low"          # Minor nuance / temporal boundary ambiguity


class ConflictResolutionStatus(StrEnum):
    """Status of conflict resolution."""
    CONFLICTED = "conflicted"        # Active contradiction, unresolved
    RESOLVED = "resolved"            # Explicitly resolved by user or newer authoritative turn
    HEALING_PENDING = "pending"     # System requested clarification, awaiting input


@dataclass
class ConflictingEvidence:
    """A specific piece of evidence in conflict."""
    value: str
    source: str
    relation: IRRelation
    time_scope: str | None = None
    statement: str = ""
    recorded_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConflictState:
    """Represents an explicit conflict or ambiguity on an Entity-Property pair."""
    entity: str
    property: str
    active_candidate: str | None
    conflicting_evidences: list[ConflictingEvidence] = field(default_factory=list)
    severity: ConflictSeverity = ConflictSeverity.HIGH
    status: ConflictResolutionStatus = ConflictResolutionStatus.CONFLICTED
    created_at: datetime = field(default_factory=datetime.now)
    resolved_value: str | None = None

    def add_evidence(
        self,
        value: str,
        source: str,
        relation: IRRelation = IRRelation.ASSERTS,
        time_scope: str | None = None,
        statement: str = "",
    ) -> None:
        """Add a conflicting statement to this state."""
        self.conflicting_evidences.append(
            ConflictingEvidence(
                value=value,
                source=source,
                relation=relation,
                time_scope=time_scope,
                statement=statement,
            )
        )

    def format_for_context_ir(self) -> str:
        """Format as a deterministic context tag for LLM consumption."""
        lines = [f"[STATE]: {self.entity} {self.property} = {self.active_candidate or 'UNKNOWN'}"]
        if self.status == ConflictResolutionStatus.CONFLICTED:
            lines.append(f"[STATUS]: CONFLICTED (Severity: {self.severity.value})")
            for ev in self.conflicting_evidences:
                scope_str = f" ({ev.time_scope})" if ev.time_scope else ""
                lines.append(f"  - CONFLICTING RECORD: {ev.value} via {ev.source}{scope_str}: '{ev.statement}'")
            lines.append("[SYSTEM DIRECTIVE]: Acknowledge the current candidate but explicitly state the historical conflict/proposal if asked.")
        return "\n".join(lines)

    def format_user_clarification_prompt(self) -> str:
        """Generate a natural clarification question for self-healing."""
        ev_summary = ", ".join(f"'{ev.value}' ({ev.source})" for ev in self.conflicting_evidences)
        return (
            f"Regarding {self.entity} ({self.property}): I have records pointing to {self.active_candidate}, "
            f"but also conflicting records for {ev_summary}. Which one is currently active?"
        )


class ConflictStateManager:
    """Manages active conflicts, detection, and self-healing transitions."""

    def __init__(self) -> None:
        # Key: (entity.lower(), property.lower())
        self._conflicts: dict[tuple[str, str], ConflictState] = {}

    def detect_and_register(self, ir_items: list[StructuredIR]) -> list[ConflictState]:
        """Detect contradictions across structured IR items and register conflict states."""
        # Group by (entity, property)
        grouped: dict[tuple[str, str], list[StructuredIR]] = {}
        for item in ir_items:
            key = (item.entity.strip().lower(), item.property.strip().lower())
            grouped.setdefault(key, []).append(item)

        new_conflicts: list[ConflictState] = []

        for (ent, prop), items in grouped.items():
            if len(items) <= 1:
                continue

            # Check if there are differing values
            values = {it.value.strip().lower() for it in items if it.value}
            if len(values) <= 1:
                continue

            # Check for conflicting relations or multiple active values
            active_items = [it for it in items if it.status == IRStatus.ACTIVE]
            sources = {it.source for it in items}

            # If there are multiple different active values or contradictory relations
            has_conflict = False
            severity = ConflictSeverity.LOW

            if len(active_items) > 1 and len({it.value.strip().lower() for it in active_items}) > 1:
                has_conflict = True
                severity = ConflictSeverity.HIGH
            elif any(it.relation == IRRelation.CONTRADICTS for it in items):
                has_conflict = True
                severity = ConflictSeverity.HIGH
            elif any(it.relation == IRRelation.MIGRATED for it in items):
                # Migration is a temporal transition, not necessarily an unresolved conflict
                # unless multiple target values exist
                migrated_values = {it.value.strip().lower() for it in items if it.relation == IRRelation.MIGRATED}
                if len(migrated_values) > 1:
                    has_conflict = True
                    severity = ConflictSeverity.MEDIUM
            elif len(sources) > 1:
                has_conflict = True
                severity = ConflictSeverity.MEDIUM

            if has_conflict:
                # Pick the latest active as candidate
                active_candidate = active_items[-1].value if active_items else items[-1].value
                c_state = ConflictState(
                    entity=items[0].entity,
                    property=items[0].property,
                    active_candidate=active_candidate,
                    severity=severity,
                )
                for it in items:
                    if it.value.strip().lower() != active_candidate.strip().lower():
                        c_state.add_evidence(
                            value=it.value,
                            source=it.source,
                            relation=it.relation,
                            time_scope=it.time_scope,
                            statement=it.raw_content,
                        )
                self._conflicts[(ent, prop)] = c_state
                new_conflicts.append(c_state)

        return new_conflicts

    def get_conflict(self, entity: str, property: str) -> ConflictState | None:
        return self._conflicts.get((entity.strip().lower(), property.strip().lower()))

    def resolve_conflict(self, entity: str, property: str, resolved_value: str) -> bool:
        """Resolve an active conflict with an authoritative value."""
        key = (entity.strip().lower(), property.strip().lower())
        if key in self._conflicts:
            c_state = self._conflicts[key]
            c_state.status = ConflictResolutionStatus.RESOLVED
            c_state.resolved_value = resolved_value
            c_state.active_candidate = resolved_value
            return True
        return False
