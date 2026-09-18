"""Evolution policy layer (AM v0.2.0 Phase 1).

Implements the plan's central safety principle: **AI-generated mutations are
proposals, not truth.** The flow is:

    proposal -> policy validation -> commit -> evolution event (audit)

``EvolutionGovernor`` wraps ``MemoryEvolutionEngine``. Nothing reaches the
store without passing deterministic policy validation. In Phase 1 all
proposals are deterministic; LLM-driven reflection (Phase 5) will submit
proposals through this same gate, which is why the gate exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.core.models import Memory
from artificial_memory.memory.evolution import (
    EvolutionOperationType,
    MemoryEvolutionEngine,
)


class PolicyVerdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


# Operations that mutate or discard an existing representation.
DESTRUCTIVE_OPERATIONS = {
    EvolutionOperationType.MERGE,
    EvolutionOperationType.SPLIT,
    EvolutionOperationType.REJECT,
    EvolutionOperationType.REINTERPRET,
}

# Operations that must carry textual evidence.
EVIDENCE_REQUIRED_OPERATIONS = {
    EvolutionOperationType.REVISE,
    EvolutionOperationType.REINTERPRET,
    EvolutionOperationType.SUPERSEDE,
}

# Operations that require structured parameters in proposal.metadata.
#   CONTRADICT  -> other_memory_id (int)
#   TRANSITION  -> to_status (MemoryStatus value)
PARAMETER_REQUIRED_OPERATIONS = {
    EvolutionOperationType.CONTRADICT,
    EvolutionOperationType.TRANSITION,
}

# The only operations meaningful for an archived memory target; anything
# else must go through RESTORE / REACTIVATE first. TRANSITION is included
# because ARCHIVED -> DEEP_ARCHIVED (deepen) and ARCHIVED -> ACTIVE
# (restore-like) are valid archive-level transitions.
ARCHIVE_SAFE_OPERATIONS = {
    EvolutionOperationType.RESTORE,
    EvolutionOperationType.REACTIVATE,
    EvolutionOperationType.KEEP,
    EvolutionOperationType.TRANSITION,
}

# Operations the governor can dispatch in Phase 2. All nine v0.2.0 lifecycle
# operations are now dispatchable; COMPRESS/ARCHIVE stay with the compression
# pipeline and status transitions, and HEAL remains an integrity path.
DISPATCHABLE_OPERATIONS = {
    EvolutionOperationType.REVISE,
    EvolutionOperationType.MERGE,
    EvolutionOperationType.SPLIT,
    EvolutionOperationType.REINFORCE,
    EvolutionOperationType.REINTERPRET,
    EvolutionOperationType.RESTORE,
    EvolutionOperationType.REACTIVATE,
    EvolutionOperationType.KEEP,
    EvolutionOperationType.REJECT,
    EvolutionOperationType.HEAL,
    # --- Phase 2 ---
    EvolutionOperationType.CONTRADICT,
    EvolutionOperationType.SUPERSEDE,
    EvolutionOperationType.TRANSITION,
}


@dataclass
class EvolutionProposal:
    """A proposed lifecycle operation, submitted for policy validation."""

    operation: EvolutionOperationType
    memory_id: int
    reason: str = ""
    evidence: str = ""
    confidence: float = 0.8
    interpretation: str = ""  # required for REINTERPRET
    merge_memory_ids: list[int] = field(default_factory=list)  # for MERGE
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PolicyDecision:
    """Outcome of policy validation (and commit, if approved)."""

    proposal: EvolutionProposal
    verdict: PolicyVerdict
    reasons: list[str] = field(default_factory=list)
    checked_rules: list[str] = field(default_factory=list)
    committed: bool = False
    resulting_memory_id: int | None = None

    @property
    def approved(self) -> bool:
        return self.verdict == PolicyVerdict.APPROVE


@dataclass
class EvolutionPolicyConfig:
    """Deterministic policy thresholds (Q7: policy first, LLM later)."""

    # Minimum proposal confidence per operation class.
    min_confidence_standard: float = 0.4
    min_confidence_destructive: float = 0.6
    # A non-empty reason is mandatory for every mutation.
    require_reason: bool = True
    # Minimum seconds between mutations of the same memory (0 disables).
    cooldown_seconds: float = 0.0


class EvolutionGovernor:
    """Validates lifecycle proposals and commits approved operations.

    The governor never invents operations and never writes to the store
    directly; it only dispatches to ``MemoryEvolutionEngine`` methods, which
    record an ``EvolutionEvent`` for every mutation (auditability).
    """

    def __init__(
        self,
        engine: MemoryEvolutionEngine,
        config: EvolutionPolicyConfig | None = None,
    ):
        self.engine = engine
        self.config = config or EvolutionPolicyConfig()

    # ==================== Public API ====================

    def submit(self, proposal: EvolutionProposal) -> PolicyDecision:
        """Validate a proposal; commit it if approved. Never raises on rejection."""
        decision = self._validate(proposal)
        if not decision.approved:
            return decision

        memory = self._commit(proposal)
        decision.committed = True
        decision.resulting_memory_id = memory.id if memory is not None else None
        decision.checked_rules.append("commit")
        return decision

    # ==================== Validation ====================

    def _validate(self, proposal: EvolutionProposal) -> PolicyDecision:
        decision = PolicyDecision(proposal=proposal, verdict=PolicyVerdict.APPROVE)
        reasons = decision.reasons
        checks = decision.checked_rules

        def reject(reason: str) -> None:
            decision.verdict = PolicyVerdict.REJECT
            reasons.append(reason)

        memory = self.engine.store.get_memory(proposal.memory_id)

        # Rule 1: dispatchability.
        checks.append("dispatchable_operation")
        if proposal.operation not in DISPATCHABLE_OPERATIONS:
            reject(
                f"Operation {proposal.operation.value} has no dispatch target "
                "(unrouted operations are rejected rather than silently dropped)"
            )

        # Rule 2: target exists.
        checks.append("target_exists")
        if memory is None:
            reject(f"Target memory {proposal.memory_id} not found")

        # Rule 3: bounded confidence.
        checks.append("confidence_in_range")
        if not 0.0 <= proposal.confidence <= 1.0:
            reject(f"Proposal confidence {proposal.confidence} outside [0, 1]")

        # Rule 4: mandatory reason (auditable decisions).
        checks.append("reason_present")
        if self.config.require_reason and not proposal.reason.strip():
            reject("A non-empty reason is required for every proposal")


        if memory is not None:
            # Rule 5: confidence threshold per operation class.
            checks.append("confidence_threshold")
            destructive = proposal.operation in DESTRUCTIVE_OPERATIONS
            threshold = (
                self.config.min_confidence_destructive
                if destructive
                else self.config.min_confidence_standard
            )
            if proposal.confidence < threshold:
                reject(
                    f"Confidence {proposal.confidence:.2f} below required "
                    f"{threshold:.2f} for "
                    f"{'destructive' if destructive else 'standard'} operation "
                    f"{proposal.operation.value}"
                )

            # Rule 6: evidence required for evidence-bearing operations.
            checks.append("evidence_present")
            if (
                proposal.operation in EVIDENCE_REQUIRED_OPERATIONS
                and not proposal.evidence.strip()
            ):
                reject(f"Operation {proposal.operation.value} requires textual evidence")

            # Rule 7: interpretation required for REINTERPRET.
            checks.append("interpretation_present")
            if proposal.operation == EvolutionOperationType.REINTERPRET:
                if not (proposal.interpretation or proposal.evidence).strip():
                    reject("REINTERPRET requires an interpretation")

            # Rule 7b: structured parameters required for parameter ops.
            checks.append("parameters_present")
            if proposal.operation == EvolutionOperationType.CONTRADICT:
                other = proposal.metadata.get("other_memory_id")
                if not isinstance(other, int) or other == proposal.memory_id:
                    reject("CONTRADICT requires an integer other_memory_id (not the target itself)")
                elif self.engine.store.get_memory(int(other)) is None:
                    reject(f"CONTRADICT target memory {other} not found")
            if proposal.operation == EvolutionOperationType.TRANSITION:
                to_status = proposal.metadata.get("to_status")
                try:
                    from artificial_memory.core.models import MemoryStatus

                    if not isinstance(to_status, str):
                        raise TypeError("to_status must be a string")
                    MemoryStatus(to_status)
                except (ValueError, TypeError):
                    reject("TRANSITION requires a valid to_status (MemoryStatus value)")

            # Rule 8: archived targets may only be restored / reactivated / kept.
            checks.append("archive_safety")
            if (
                memory.status.value in ("archived", "deep_archived")
                and proposal.operation not in ARCHIVE_SAFE_OPERATIONS
            ):
                reject(
                    f"Memory {proposal.memory_id} is archived; use RESTORE or "
                    f"REACTIVATE before {proposal.operation.value}"
                )

            # Rule 9: per-memory cooldown between mutations.
            if self.config.cooldown_seconds > 0:
                checks.append("cooldown")
                last_event = self._last_event_time(proposal.memory_id)
                if last_event is not None:
                    elapsed = (proposal.created_at - last_event).total_seconds()
                    if elapsed < self.config.cooldown_seconds:
                        reject(
                            f"Cooldown active: {elapsed:.1f}s since last mutation "
                            f"(required {self.config.cooldown_seconds:.1f}s)"
                        )

        return decision


    def _last_event_time(self, memory_id: int) -> datetime | None:
        try:
            # ``get_evolution_events`` exists on persistent store backends;
            # in-memory or minimal stores may not implement it.
            get_events = getattr(self.engine.store, "get_evolution_events", None)
            if get_events is None:
                return None
            events = get_events(memory_id=memory_id, limit=1)
        except (AttributeError, TypeError):
            return None
        if not events:
            return None
        first: datetime = events[0].created_at
        return first

    # ==================== Commit ====================

    def _commit(self, proposal: EvolutionProposal) -> Memory | None:
        engine = self.engine
        op = proposal.operation
        memory_id = proposal.memory_id
        metadata = proposal.metadata

        if op == EvolutionOperationType.REVISE:
            return engine.revise_memory(
                memory_id,
                proposal.evidence,
                evidence_source=metadata.get("evidence_source", "policy"),
                confidence=proposal.confidence,
                should_supersede=metadata.get("should_supersede", False),
            )
        if op == EvolutionOperationType.REINFORCE:
            return engine.reinforce_memory(
                memory_id,
                evidence_source=metadata.get("evidence_source", "policy"),
                amount=metadata.get("amount", 0.05),
                reason=proposal.reason,
                triggered_by=metadata.get("triggered_by", "policy"),
            )
        if op == EvolutionOperationType.REINTERPRET:
            return engine.reinterpret_memory(
                memory_id,
                proposal.interpretation or proposal.evidence,
                confidence=proposal.confidence,
                supersede=metadata.get("supersede", False),
                evidence_source=metadata.get("evidence_source", "policy"),
            )


        if op == EvolutionOperationType.MERGE:
            return engine.merge_memories(
                [memory_id, *proposal.merge_memory_ids],
                merge_strategy=metadata.get("merge_strategy", "combine"),
            )
        if op == EvolutionOperationType.SPLIT:
            return engine.split_memory(memory_id, metadata.get("split_points", []))[0]
        if op == EvolutionOperationType.RESTORE:
            return engine.restore_memory(
                memory_id,
                target_resolution=metadata.get("target_resolution"),
                triggered_by=metadata.get("triggered_by", "policy"),
            )
        if op == EvolutionOperationType.REACTIVATE:
            return engine.reactivate_memory(memory_id)
        if op == EvolutionOperationType.KEEP:
            return engine.keep_memory(memory_id, reason=proposal.reason)
        if op == EvolutionOperationType.REJECT:
            return engine.reject_memory(memory_id, reason=proposal.reason)
        if op == EvolutionOperationType.HEAL:
            return engine.heal_memory(
                memory_id,
                metadata.get("issue_type", "contradiction"),
                metadata.get("repair_action", "manual review"),
            )
        # --- Phase 2: temporal / contradiction layer ---
        if op == EvolutionOperationType.CONTRADICT:
            from artificial_memory.memory.contradiction_edges import (
                create_contradiction_edge_manager,
            )

            edge_manager = create_contradiction_edge_manager(
                engine.store, triggered_by=metadata.get("triggered_by", "policy")
            )
            edge_manager.create_edge(
                memory_id,
                metadata["other_memory_id"],
                severity=metadata.get("severity", 0.8),
                description=proposal.reason,
                triggered_by=metadata.get("triggered_by", "policy"),
            )
            return engine.store.get_memory(memory_id)
        if op == EvolutionOperationType.SUPERSEDE:
            from artificial_memory.memory.supersession import create_supersession_manager

            supersession = create_supersession_manager(engine.store)
            _old, new = supersession.supersede_memory(
                memory_id,
                proposal.evidence,
                confidence=proposal.confidence,
                reason=proposal.reason,
                triggered_by=metadata.get("triggered_by", "policy"),
            )
            return new
        if op == EvolutionOperationType.TRANSITION:
            from artificial_memory.core.models import MemoryStatus
            from artificial_memory.memory.lifecycle_transitions import (
                create_lifecycle_transition_engine,
            )

            transition_engine = create_lifecycle_transition_engine(engine.store)
            return transition_engine.transition(
                memory_id,
                MemoryStatus(metadata["to_status"]),
                reason=proposal.reason,
                triggered_by=metadata.get("triggered_by", "policy"),
            )
        return None


def create_evolution_governor(
    engine: MemoryEvolutionEngine,
    config: EvolutionPolicyConfig | None = None,
) -> EvolutionGovernor:
    return EvolutionGovernor(engine, config)


__all__ = [
    "DESTRUCTIVE_OPERATIONS",
    "DISPATCHABLE_OPERATIONS",
    "EVIDENCE_REQUIRED_OPERATIONS",
    "ARCHIVE_SAFE_OPERATIONS",
    "EvolutionGovernor",
    "EvolutionPolicyConfig",
    "EvolutionProposal",
    "PolicyDecision",
    "PolicyVerdict",
    "create_evolution_governor",
]




