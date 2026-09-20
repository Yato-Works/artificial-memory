"""Memory IR 2.0 & Proof-Carrying Context Types (Apex Phase A).

Defines the 4 core memory roles (STATE, EVIDENCE, EVENT, ABSTRACTION)
and the Proof-Carrying Context (Coverage Certificate) contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.core.ir.structured import StructuredIR


class MemoryRole(StrEnum):
    """The 4 fundamental cognitive memory roles in Apex Architecture."""
    STATE = "state"              # Current valid truth/configuration (e.g. database = PostgreSQL)
    EVIDENCE = "evidence"        # Historical statement, proposal, or utterance (e.g. Alice proposed Redis)
    EVENT = "event"              # State transition/mutation over time (e.g. 2026-03-01: Redis -> ClickHouse)
    ABSTRACTION = "abstraction"  # Synthesized behavioral or systemic pattern (e.g. prefers rollback/stability)


class QueryIntent(StrEnum):
    """Classification of what world the query expects to reconstruct."""
    STATE_QUERY = "state_query"            # "What is the current database?"
    EVIDENCE_QUERY = "evidence_query"      # "Did Alice propose Redis?"
    EVENT_QUERY = "event_query"            # "How did the database evolve over time?"
    REFLECTION_QUERY = "reflection_query"  # "What recurring preference does the user show?"
    ABSTENTION_QUERY = "abstention_query"  # Query asking about something never set/unknown
    AGGREGATION_QUERY = "aggregation_query"  # "How many items total across sessions?"


@dataclass
class CoverageCertificate:
    """Mathematical and semantic proof of Minimum Sufficient Context."""
    is_sufficient: bool = False
    entity_coverage: bool = False
    property_coverage: bool = False
    temporal_coverage: bool = False
    conflict_coverage: bool = False
    omitted_records_restored: int = 0
    details: list[str] = field(default_factory=list)

    def format_certificate(self) -> str:
        """Format certificate for audit and LLM consumption."""
        status = "VERIFIED_SUFFICIENT" if self.is_sufficient else "PARTIAL_COVERAGE"
        lines = [
            f"[COVERAGE CERTIFICATE: {status}]",
            f"  - Entity Scope    : {'MATCHED' if self.entity_coverage else 'PARTIAL/ABSENT'}",
            f"  - Property/Aspect : {'MATCHED' if self.property_coverage else 'UNSPECIFIED'}",
            f"  - Temporal State  : {'RESOLVED' if self.temporal_coverage else 'DEFAULT_CURRENT'}",
            f"  - Conflict Status : {'RESOLVED/ANNOTATED' if self.conflict_coverage else 'CLEAR'}",
        ]
        if self.omitted_records_restored > 0:
            lines.append(f"  - Recovery Action : Restored {self.omitted_records_restored} omitted evidence record(s)")
        return "\n".join(lines)


@dataclass
class ApexMemoryUnit:
    """An enriched memory unit in Memory IR 2.0 with explicit cognitive role."""
    ir: StructuredIR
    role: MemoryRole = MemoryRole.STATE
    event_timestamp: str | None = None
    supported_by: list[str] = field(default_factory=list)  # IDs of evidence supporting an ABSTRACTION
    confidence: float = 1.0

    @property
    def entity(self) -> str:
        return self.ir.entity

    @property
    def target_property(self) -> str:
        return self.ir.property

    @property
    def value(self) -> str:
        return self.ir.value


@dataclass
class ProofCarryingContext:
    """A Minimum Sufficient Context bundled with its Coverage Certificate."""
    context_text: str
    certificate: CoverageCertificate
    intent: QueryIntent
    token_cost: int = 0
    is_abstention: bool = False
