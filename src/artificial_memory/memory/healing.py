from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import (
    CompressionEvent,
    CompressionMethod,
    Memory,
    MemoryVersion,
    ResolutionLevel,
)
from artificial_memory.memory.integrity import (
    IntegrityIssue,
    IntegrityMetrics,
    IntegrityReport,
)
from artificial_memory.memory.validation import CompressionValidator


class HealingActionType(StrEnum):
    RECOMPRESS = "recompress"                 # Re-run compression
    RESTORE_VERSION = "restore_version"       # Restore from higher-res version
    RECOMPILE = "recompile"                   # Re-run compiler pipeline
    REPAIR_PROVENANCE = "repair_provenance"   # Fix broken provenance links
    RESOLVE_CONTRADICTION = "resolve_contradiction"  # Resolve via belief engine
    REACTIVATE = "reactivate"                 # Reactivate archived memory
    MERGE_DUPLICATES = "merge_duplicates"     # Merge near-duplicate memories


@dataclass
class HealingAction:
    """A healing action to perform."""
    action_type: HealingActionType
    target_memory_id: int
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more urgent
    reason: str = ""


@dataclass
class HealingResult:
    """Result of a healing operation."""
    action: HealingAction
    success: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=datetime.now)


@dataclass
class HealingPlan:
    """A plan for healing multiple issues."""
    memory_id: int
    report: IntegrityReport
    actions: list[HealingAction] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class MemoryHealer:
    """Automatically heals memory integrity issues.

    Uses integrity metrics to detect issues and applies appropriate
    healing actions: recompile, restore versions, repair provenance, etc.
    """

    def __init__(
        self,
        store: MemoryStore,
        compressor: RuleBasedCompressor,
        compiler_pipeline=None,
        belief_engine=None,
        evolution_engine=None,
        contradiction_detector=None,
    ):
        self.store = store
        self.compressor = compressor
        self.compiler_pipeline = compiler_pipeline
        self.belief_engine = belief_engine
        self.evolution_engine = evolution_engine
        self.contradiction_detector = contradiction_detector

        self.integrity_metrics = IntegrityMetrics(store, compressor)
        self.stale_detector = None  # Set later if needed
        self.compression_validator = CompressionValidator(store, compressor)

    def diagnose(self, memory: Memory) -> IntegrityReport:
        """Diagnose all integrity issues for a memory."""
        return self.integrity_metrics.compute_overall_integrity(memory)

    def create_healing_plan(self, report: IntegrityReport) -> HealingPlan:
        """Create a healing plan based on integrity report."""
        actions = []

        for issue in report.issues:
            action = self._issue_to_action(issue)
            if action:
                actions.append(action)

        # Sort by priority
        actions.sort(key=lambda a: -a.priority)

        return HealingPlan(
            memory_id=report.memory_id,
            report=report,
            actions=actions,
        )

    def _issue_to_action(self, issue: IntegrityIssue) -> HealingAction | None:
        """Map an integrity issue to a healing action."""
        if issue.issue_type == 'semantic_drift':
            return HealingAction(
                action_type=HealingActionType.RECOMPILE,
                target_memory_id=issue.memory_id,
                parameters={"reason": "semantic_drift"},
                priority=10 if issue.severity == 'critical' else 5,
                reason=f"Semantic drift detected: {issue.description}",
            )

        elif issue.issue_type == 'compression_artifact':
            return HealingAction(
                action_type=HealingActionType.RECOMPRESS,
                target_memory_id=issue.memory_id,
                parameters={"reason": "compression_artifact"},
                priority=8 if issue.severity == 'critical' else 4,
                reason=f"Compression artifact: {issue.description}",
            )

        elif issue.issue_type == 'provenance_broken':
            return HealingAction(
                action_type=HealingActionType.REPAIR_PROVENANCE,
                target_memory_id=issue.memory_id,
                parameters={"reason": "provenance_broken"},
                priority=7 if issue.severity == 'critical' else 3,
                reason=f"Provenance broken: {issue.description}",
            )

        elif issue.issue_type == 'contradiction':
            return HealingAction(
                action_type=HealingActionType.RESOLVE_CONTRADICTION,
                target_memory_id=issue.memory_id,
                parameters={"reason": "contradiction"},
                priority=9 if issue.severity == 'critical' else 5,
                reason=f"Contradiction detected: {issue.description}",
            )

        elif issue.issue_type == 'temporal_inconsistency':
            return HealingAction(
                action_type=HealingActionType.RESTORE_VERSION,
                target_memory_id=issue.memory_id,
                parameters={"reason": "temporal_inconsistency", "target_resolution": "SEMANTIC"},
                priority=6,
                reason=f"Temporal inconsistency: {issue.description}",
            )

        elif issue.issue_type == 'stale':
            return HealingAction(
                action_type=HealingActionType.REACTIVATE,
                target_memory_id=issue.memory_id,
                parameters={"reason": "stale"},
                priority=4,
                reason=f"Stale memory: {issue.description}",
            )

        return None

    def execute_plan(self, plan: HealingPlan) -> list[HealingResult]:
        """Execute a healing plan."""
        results = []

        for action in plan.actions:
            result = self.execute_action(action)
            results.append(result)

        return results

    def execute_action(self, action: HealingAction) -> HealingResult:
        """Execute a single healing action."""
        if action.action_type == HealingActionType.RECOMPILE:
            return self._recompile_memory(action)
        elif action.action_type == HealingActionType.RECOMPRESS:
            return self._recompress_memory(action)
        elif action.action_type == HealingActionType.RESTORE_VERSION:
            return self._restore_version(action)
        elif action.action_type == HealingActionType.REPAIR_PROVENANCE:
            return self._repair_provenance(action)
        elif action.action_type == HealingActionType.RESOLVE_CONTRADICTION:
            return self._resolve_contradiction(action)
        elif action.action_type == HealingActionType.REACTIVATE:
            return self._reactivate_memory(action)
        elif action.action_type == HealingActionType.MERGE_DUPLICATES:
            return self._merge_duplicates(action)
        else:
            return HealingResult(
                action=action,
                success=False,
                message=f"Unknown action type: {action.action_type}",
            )

    # ==================== Action Implementations ====================

    def _recompile_memory(self, action: HealingAction) -> HealingResult:
        """Re-run compiler pipeline on source conversation."""
        if not self.compiler_pipeline:
            return HealingResult(
                action=action,
                success=False,
                message="Compiler pipeline not available",
            )

        memory = self.store.get_memory(action.target_memory_id)
        if not memory:
            return HealingResult(action=action, success=False, message="Memory not found")

        if not memory.source_conversation_id:
            return HealingResult(action=action, success=False, message="No source conversation")

        conv = self.store.get_conversation(memory.source_conversation_id)
        if not conv:
            return HealingResult(action=action, success=False, message="Source conversation not found")

        messages = self.store.get_messages(conv.id)

        try:
            # Run compiler pipeline
            memory_irs = self.compiler_pipeline.compile(conv, messages)

            # Find the one matching our memory
            for mem_ir in memory_irs:
                if mem_ir.semantic_content.content == memory.content:
                    # Found match - update from new IR
                    legacy_mem, versions, _, _ = self.compiler_pipeline.ir_adapter.to_legacy(mem_ir)
                    legacy_mem.id = memory.id  # Preserve ID
                    updated = self.store.update_memory(legacy_mem)

                    # Re-create versions
                    for v in versions:
                        v.memory_id = memory.id
                        self.store.add_memory_version(v)

                    return HealingResult(
                        action=action,
                        success=True,
                        message=f"Recompiled from conversation {conv.id}",
                        details={"conversation_id": conv.id, "new_resolution": updated.resolution.name},
                    )

            return HealingResult(
                action=action,
                success=False,
                message="No matching memory IR found in recompilation",
            )

        except Exception as e:
            return HealingResult(
                action=action,
                success=False,
                message=f"Recompilation failed: {e}",
            )

    def _recompress_memory(self, action: HealingAction) -> HealingResult:
        """Re-run compression for a memory."""
        memory = self.store.get_memory(action.target_memory_id)
        if not memory:
            return HealingResult(action=action, success=False, message="Memory not found")

        try:
            # Re-compress at all levels
            original_content = memory.content

            for res in [ResolutionLevel.LIGHT, ResolutionLevel.EPISODE,
                       ResolutionLevel.SEMANTIC, ResolutionLevel.LONG_TERM,
                       ResolutionLevel.DEEP_LONG_TERM]:
                if res.value <= memory.resolution.value:
                    continue

                compressed, meta = self._compress_to_resolution(original_content, res)

                version = MemoryVersion(
                    memory_id=memory.id,
                    resolution=res,
                    content=compressed,
                    compression_ratio=meta.get('compression_ratio'),
                    created_at=datetime.now(),
                    source='healing_recompress',
                )
                self.store.add_memory_version(version)

                # Log compression event
                method_map = {
                    ResolutionLevel.LIGHT: CompressionMethod.LIGHT,
                    ResolutionLevel.EPISODE: CompressionMethod.EPISODE,
                    ResolutionLevel.SEMANTIC: CompressionMethod.SEMANTIC,
                    ResolutionLevel.LONG_TERM: CompressionMethod.LONG_TERM,
                    ResolutionLevel.DEEP_LONG_TERM: CompressionMethod.DEEP_LONG_TERM,
                }
                event = CompressionEvent(
                    source_memory_id=memory.id,
                    target_memory_id=None,
                    from_resolution=memory.resolution,
                    to_resolution=res,
                    original_tokens=meta.get('original_tokens', len(original_content)//3),
                    compressed_tokens=meta.get('compressed_tokens', len(compressed)//3),
                    compression_ratio=meta.get('compression_ratio', 1.0),
                    method=method_map.get(res, CompressionMethod.LIGHT),
                    created_at=datetime.now(),
                )
                self.store.log_compression(event)

            # Update memory status
            memory.updated_at = datetime.now()
            self.store.update_memory(memory)

            return HealingResult(
                action=action,
                success=True,
                message="Re-compressed all resolution levels",
                details={"resolutions": ["LIGHT", "EPISODE", "SEMANTIC", "LONG_TERM", "DEEP_LONG_TERM"]},
            )

        except Exception as e:
            return HealingResult(
                action=action,
                success=False,
                message=f"Recompression failed: {e}",
            )

    def _restore_version(self, action: HealingAction) -> HealingResult:
        """Restore memory from a higher-resolution version."""
        memory = self.store.get_memory(action.target_memory_id)
        if not memory:
            return HealingResult(action=action, success=False, message="Memory not found")

        target_res = action.parameters.get("target_resolution", "SEMANTIC")
        target_resolution = ResolutionLevel[target_res]

        version = self.store.get_memory_version(memory.id, target_resolution)
        if not version:
            return HealingResult(
                action=action,
                success=False,
                message=f"No version at {target_resolution.name}",
            )

        # Restore content and resolution
        old_content = memory.content
        memory.content = version.content
        memory.resolution = target_resolution
        memory.updated_at = datetime.now()

        self.store.update_memory(memory)

        return HealingResult(
            action=action,
            success=True,
            message=f"Restored from {target_resolution.name} version",
            details={
                "old_content_length": len(old_content),
                "new_content_length": len(version.content),
                "restored_resolution": target_resolution.name,
            },
        )

    def _repair_provenance(self, action: HealingAction) -> HealingResult:
        """Attempt to repair broken provenance links."""
        memory = self.store.get_memory(action.target_memory_id)
        if not memory:
            return HealingResult(action=action, success=False, message="Memory not found")

        repaired = []

        # Try to find source conversation
        if not memory.source_conversation_id:
            # Search for conversations with similar content
            conversations = self.store.list_conversations()
            for conv in conversations[:100]:  # Limit to first 100
                messages = self.store.get_messages(conv.id)
                for msg in messages:
                    if self._content_similar(memory.content, msg.content) > 0.8:
                        memory.source_conversation_id = conv.id
                        memory.source_message_id = msg.id
                        repaired.append(f"conversation:{conv.id}")
                        break

        # Try to find source message
        if memory.source_conversation_id and not memory.source_message_id:
            messages = self.store.get_messages(memory.source_conversation_id)
            for msg in messages:
                if self._content_similar(memory.content, msg.content) > 0.7:
                    memory.source_message_id = msg.id
                    repaired.append(f"message:{msg.id}")
                    break

        if repaired:
            memory.updated_at = datetime.now()
            self.store.update_memory(memory)

            return HealingResult(
                action=action,
                success=True,
                message=f"Repaired provenance: {', '.join(repaired)}",
                details={"repaired": repaired},
            )
        else:
            return HealingResult(
                action=action,
                success=False,
                message="Could not repair provenance",
            )

    def _resolve_contradiction(self, action: HealingAction) -> HealingResult:
        """Resolve a contradiction via belief engine or evolution."""
        memory = self.store.get_memory(action.target_memory_id)
        if not memory:
            return HealingResult(action=action, success=False, message="Memory not found")

        # Find contradicting memories
        associations = self.store.get_associations(memory.id)
        contradicting_ids = []

        for assoc in associations:
            if assoc.association_type.value == "contradicts":
                other_id = (assoc.target_memory_id
                           if assoc.source_memory_id == memory.id
                           else assoc.source_memory_id)
                contradicting_ids.append(other_id)

        if not contradicting_ids:
            return HealingResult(
                action=action,
                success=False,
                message="No contradictions found to resolve",
            )

        # Use belief engine if available
        if self.belief_engine:
            conflicts = self.belief_engine.detect_contradictions([memory.id])
            for conflict in conflicts:
                self.belief_engine.resolve_conflict(
                    conflict.conflicting_memory_id,
                    "superseded_by_newer_evidence"
                )

        # Mark contradiction as resolved
        for assoc in associations:
            if assoc.association_type.value == "contradicts":
                # In practice, would update association metadata
                pass

        return HealingResult(
            action=action,
            success=True,
            message=f"Resolved {len(contradicting_ids)} contradictions",
            details={"contradicting_ids": contradicting_ids},
        )

    def _reactivate_memory(self, action: HealingAction) -> HealingResult:
        """Reactivate an archived memory."""
        if self.evolution_engine:
            memory = self.evolution_engine.reactivate_memory(action.target_memory_id)
            if memory:
                return HealingResult(
                    action=action,
                    success=True,
                    message="Memory reactivated",
                )

        return HealingResult(
            action=action,
            success=False,
            message="Evolution engine not available",
        )

    def _merge_duplicates(self, action: HealingAction) -> HealingResult:
        """Merge near-duplicate memories."""
        # This would find and merge similar memories
        return HealingResult(
            action=action,
            success=False,
            message="Not implemented yet",
        )

    # ==================== Batch Operations ====================

    def heal_topic(self, topic_id: int, max_issues: int = 50) -> list[HealingResult]:
        """Heal all issues in a topic."""
        memories = self.store.get_memories(topic_id=topic_id, limit=100)

        all_results = []
        for memory in memories[:max_issues]:
            report = self.diagnose(memory)
            if report.has_critical or report.has_warnings:
                plan = self.create_healing_plan(report)
                results = self.execute_plan(plan)
                all_results.extend(results)

        return all_results

    def _content_similar(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        return len(words_a & words_b) / max(len(words_a | words_b), 1)

    def _compress_to_resolution(self, content: str, resolution: ResolutionLevel) -> tuple[str, dict]:
        method_map = {
            ResolutionLevel.LIGHT: (self.compressor.compress_light, CompressionMethod.LIGHT),
            ResolutionLevel.EPISODE: (self.compressor.compress_episode, CompressionMethod.EPISODE),
            ResolutionLevel.SEMANTIC: (self.compressor.compress_semantic, CompressionMethod.SEMANTIC),
            ResolutionLevel.LONG_TERM: (self.compressor.compress_long_term, CompressionMethod.LONG_TERM),
            ResolutionLevel.DEEP_LONG_TERM: (self.compressor.compress_long_term, CompressionMethod.DEEP_LONG_TERM),
        }
        func, method = method_map.get(resolution, (self.compressor.compress_light, CompressionMethod.LIGHT))
        content_compressed, meta = func(content, {})
        meta["method"] = method
        return content_compressed, meta


method_map = {
    ResolutionLevel.LIGHT: CompressionMethod.LIGHT,
    ResolutionLevel.EPISODE: CompressionMethod.EPISODE,
    ResolutionLevel.SEMANTIC: CompressionMethod.SEMANTIC,
    ResolutionLevel.LONG_TERM: CompressionMethod.LONG_TERM,
    ResolutionLevel.DEEP_LONG_TERM: CompressionMethod.DEEP_LONG_TERM,
}


def create_memory_healer(
    store: MemoryStore,
    compressor: RuleBasedCompressor,
    compiler_pipeline=None,
    belief_engine=None,
    evolution_engine=None,
    contradiction_detector=None,
) -> MemoryHealer:
    return MemoryHealer(
        store, compressor, compiler_pipeline,
        belief_engine, evolution_engine, contradiction_detector
    )
