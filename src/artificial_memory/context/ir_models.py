from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class IRKey(StrEnum):
    """Standardized IR keys as defined in design doc."""
    # Conversation attributes
    CASUAL = "c"           # casual tone
    HESITATION = "h"       # hesitation
    QUESTIONING = "q"      # questioning
    REASONING = "r"        # reasoning
    DECISION = "d"         # decision
    INFORMAL = "i"         # informal
    TEMPORAL = "t"         # temporal marker
    STATE = "s"            # state
    KNOWLEDGE = "k"        # knowledge
    DECISION_EVENT = "de"  # decision event

    # Extended keys for richer representation
    TOPIC = "tp"           # topic reference
    ENTITY = "en"          # entity (person, project, etc.)
    ACTION = "ac"          # action item
    CONCERN = "cn"         # concern/risk
    TRADEOFF = "to"        # tradeoff
    AGREEMENT = "ag"       # agreement
    DISAGREEMENT = "dg"    # disagreement
    CLARIFICATION = "cl"   # clarification
    SUMMARY = "sm"         # summary


@dataclass
class IRUnit:
    """A single Context IR unit."""
    ir_type: str
    ir_key: str
    ir_value: str | None = None
    sequence_num: int = 0
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_compact_string(self) -> str:
        """Convert to compact string representation: 'key:value' or just 'key'."""
        if self.ir_value:
            return f"{self.ir_key}:{self.ir_value}"
        return self.ir_key

    @classmethod
    def from_compact_string(cls, s: str, seq: int = 0) -> IRUnit:
        """Parse compact string back to IRUnit."""
        if ':' in s:
            key, value = s.split(':', 1)
            return cls(ir_type="", ir_key=key, ir_value=value, sequence_num=seq)
        return cls(ir_type="", ir_key=s, sequence_num=seq)

    def estimate_tokens(self) -> int:
        """Estimate token count for this IR unit."""
        base = len(self.ir_key) + (len(self.ir_value) if self.ir_value else 0)
        return max(1, base // 3)


@dataclass
class IRSequence:
    """A sequence of IR units representing a conversation or context."""
    units: list[IRUnit] = field(default_factory=list)
    conversation_id: int | None = None
    topic_id: int | None = None
    metadata: dict = field(default_factory=dict)

    def add_unit(self, unit: IRUnit) -> None:
        unit.sequence_num = len(self.units)
        self.units.append(unit)

    def to_compact_string(self) -> str:
        """Convert entire sequence to compact string."""
        return ' '.join(u.to_compact_string() for u in self.units)

    def total_tokens(self) -> int:
        return sum(u.estimate_tokens() for u in self.units)

    def filter_by_type(self, ir_type: str) -> list[IRUnit]:
        return [u for u in self.units if u.ir_type == ir_type]

    def filter_by_key(self, ir_key: str) -> list[IRUnit]:
        return [u for u in self.units if u.ir_key == ir_key]


# IR Type definitions with descriptions
IR_TYPE_DEFINITIONS = {
    "tone": "Conversation tone/mood (casual, formal, excited, etc.)",
    "state": "Current state/status of a topic or project",
    "decision": "Explicit decision made",
    "reasoning": "Reasoning process, why something was decided",
    "temporal": "Time-related information (when, duration, sequence)",
    "knowledge": "Factual knowledge, definitions, facts",
    "event": "Significant event or occurrence",
    "topic": "Topic or subject being discussed",
    "entity": "Named entity (person, project, tool, etc.)",
    "action": "Action item or next step",
    "concern": "Concern, risk, or issue raised",
    "tradeoff": "Tradeoff analysis between options",
    "agreement": "Agreement or consensus point",
    "disagreement": "Disagreement or conflict",
    "clarification": "Clarification or question",
    "summary": "Summary or recap",
}


# Default IR key mappings for common patterns
IR_KEY_PATTERNS = {
    # Tone markers
    IRKey.CASUAL: ["casual", "informal", "friendly", "relaxed"],
    IRKey.HESITATION: ["hesitat", "uncertain", "maybe", "perhaps", "not sure"],
    IRKey.QUESTIONING: ["?", "why", "how", "what", "which", "question"],
    IRKey.REASONING: ["because", "reason", "since", "therefore", "thus", "logic"],
    IRKey.DECISION: ["decid", "choos", "select", "adopt", "reject", "go with"],
    IRKey.INFORMAL: ["yeah", "ok", "sure", "cool", "awesome", "great"],
    IRKey.TEMPORAL: ["now", "today", "yesterday", "tomorrow", "later", "before", "after"],
    IRKey.STATE: ["status", "current", "state", "progress", "done", "pending"],
    IRKey.KNOWLEDGE: ["know", "fact", "definition", "concept", "principle"],
    IRKey.DECISION_EVENT: ["decided", "agreed", "concluded", "finalized"],
}


def create_ir_unit(ir_type: str, ir_key: str, ir_value: str = None, seq: int = 0) -> IRUnit:
    """Factory function to create IR unit."""
    return IRUnit(
        ir_type=ir_type,
        ir_key=ir_key,
        ir_value=ir_value,
        sequence_num=seq,
    )
