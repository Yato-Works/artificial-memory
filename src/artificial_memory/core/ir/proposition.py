"""Unified Proposition & State History Representation (Phase X.8 / Overdrive Core).

Unifies facts, preferences, temporal events, state updates, and multi-hop coreferences
into a single, structured proposition tuple:
    (subject, predicate, object, time_scope, session_id, source, state_status, confidence)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional, Sequence


class PropositionStatus(StrEnum):
    """Lifecycle status of a proposition."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    DEPRECATED = "deprecated"


@dataclass
class UnifiedProposition:
    """A canonical proposition in the Overdrive Core memory substrate."""
    id: str
    subject: str                      # Entity / Actor (e.g., "Eli", "Caroline", "workstation")
    predicate: str                    # Action / Relation (e.g., "uses", "owns", "prefers", "moved_from", "graduated_from")
    object: str                       # Target / Value / Concept (e.g., "Sony A7R IV", "Sweden", "hotel with rooftop pool")
    time_scope: Optional[str] = None  # Valid timestamp (e.g., "2023-01-08", "2023/05/29")
    session_id: str = ""              # Session identifier (e.g., "session_14", "D4:3")
    source: str = "user"              # Speaker ("user", "assistant", "teammate")
    status: PropositionStatus = PropositionStatus.ACTIVE
    confidence: float = 1.0           # Confidence score [0.0, 1.0]
    raw_text: str = ""                # Original utterance snippet
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> tuple[str, str, str, Optional[str], str, str, str, float]:
        return (
            self.subject,
            self.predicate,
            self.object,
            self.time_scope,
            self.session_id,
            self.source,
            self.status.value,
            self.confidence,
        )

    def to_natural_text(self) -> str:
        """Format proposition into a clear, natural English sentence."""
        time_suffix = f" (on {self.time_scope})" if self.time_scope else ""
        source_prefix = f"[{self.source}] " if self.source else ""
        if self.predicate in ["owns", "uses", "prefers", "likes", "loves", "dislikes"]:
            return f"{source_prefix}{self.subject} {self.predicate} {self.object}{time_suffix}."
        elif self.predicate == "is":
            return f"{source_prefix}{self.subject} is {self.object}{time_suffix}."
        elif self.predicate == "moved_from":
            return f"{source_prefix}{self.subject} moved from {self.object}{time_suffix}."
        elif self.predicate == "occurred_at":
            return f"{source_prefix}{self.subject} occurred at {self.object}{time_suffix}."
        return f"{source_prefix}{self.subject} {self.predicate} {self.object}{time_suffix}."


@dataclass
class StateSnapshot:
    """A point-in-time value snapshot for an attribute."""
    value: str
    timestamp: Optional[str]
    session_id: str
    proposition_id: str


@dataclass
class StateHistory:
    """Tracks chronological transitions for an entity's attribute (Knowledge Update)."""
    entity: str
    attribute: str
    snapshots: list[StateSnapshot] = field(default_factory=list)

    def add_snapshot(self, value: str, timestamp: Optional[str], session_id: str, prop_id: str) -> None:
        self.snapshots.append(StateSnapshot(value, timestamp, session_id, prop_id))
        # Keep sorted chronologically if timestamps are parseable
        self.snapshots.sort(key=lambda s: s.timestamp or "")

    def get_latest(self) -> Optional[StateSnapshot]:
        """Return the most current state (for 'What do they use now?')."""
        return self.snapshots[-1] if self.snapshots else None

    def get_previous(self) -> Optional[StateSnapshot]:
        """Return the immediately preceding state (for 'What did they use before?')."""
        if len(self.snapshots) >= 2:
            return self.snapshots[-2]
        return None

    def get_history_summary(self) -> str:
        """Generate a compact, deterministic state transition certificate."""
        if not self.snapshots:
            return ""
        if len(self.snapshots) == 1:
            s = self.snapshots[0]
            t = f" as of {s.timestamp}" if s.timestamp else ""
            return f"For {self.entity}, {self.attribute} is {s.value}{t}."
        
        prev = self.snapshots[-2]
        curr = self.snapshots[-1]
        prev_t = f" ({prev.timestamp})" if prev.timestamp else ""
        curr_t = f" ({curr.timestamp})" if curr.timestamp else ""
        return (
            f"[State Transition: For {self.entity}, {self.attribute} was previously "
            f"'{prev.value}'{prev_t}, and was updated to '{curr.value}'{curr_t}.]"
        )
