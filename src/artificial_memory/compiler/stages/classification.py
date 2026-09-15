from __future__ import annotations

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
    DiagnosticSeverity,
)
from artificial_memory.core.ir import LifecycleState, MemoryIR, MemoryType, ResolutionLevel


class ClassificationStage(BaseStage):
    """Assign final MemoryType, ResolutionLevel, importance, confidence, LifecycleState."""

    def __init__(self):
        super().__init__("classification")

    def process(self, ctx: CompilerContext) -> CompilerContext:
        for mem in ctx.memory_ir_list:
            # Ensure all fields have sensible defaults
            if mem.type is None:
                mem.type = MemoryType.SEMANTIC

            if mem.resolution is None:
                mem.resolution = ResolutionLevel.SEMANTIC

            if mem.importance is None:
                mem.importance = self._default_importance(mem.type)

            if mem.confidence is None or mem.confidence.overall == 0:
                mem.confidence.overall = self._default_confidence(mem.type)

            # Map lifecycle state to legacy status for compatibility
            mem.lifecycle_state = mem.lifecycle_state or LifecycleState.HOT

            # Validate confidence components
            self._normalize_confidence(mem)

        return ctx

    def _default_importance(self, mem_type: MemoryType) -> float:
        defaults = {
            MemoryType.DECISION: 0.85,
            MemoryType.CURRENT: 0.9,
            MemoryType.TIMELINE: 0.7,
            MemoryType.EPISODE: 0.65,
            MemoryType.SEMANTIC: 0.7,
            MemoryType.CONVERSATION_STYLE: 0.5,
        }
        return defaults.get(mem_type, 0.6)

    def _default_confidence(self, mem_type: MemoryType) -> float:
        defaults = {
            MemoryType.DECISION: 0.85,
            MemoryType.CURRENT: 0.95,
            MemoryType.TIMELINE: 0.75,
            MemoryType.EPISODE: 0.75,
            MemoryType.SEMANTIC: 0.8,
            MemoryType.CONVERSATION_STYLE: 0.7,
        }
        return defaults.get(mem_type, 0.7)

    def _normalize_confidence(self, mem: MemoryIR) -> None:
        """Ensure confidence components are in valid range."""
        if mem.confidence.memory_confidence < 0 or mem.confidence.memory_confidence > 1:
            mem.confidence.memory_confidence = max(0.0, min(1.0, mem.confidence.memory_confidence))
        if mem.confidence.retrieval_confidence < 0 or mem.confidence.retrieval_confidence > 1:
            mem.confidence.retrieval_confidence = max(0.0, min(1.0, mem.confidence.retrieval_confidence))
        if mem.confidence.temporal_confidence < 0 or mem.confidence.temporal_confidence > 1:
            mem.confidence.temporal_confidence = max(0.0, min(1.0, mem.confidence.temporal_confidence))
        if mem.confidence.source_confidence < 0 or mem.confidence.source_confidence > 1:
            mem.confidence.source_confidence = max(0.0, min(1.0, mem.confidence.source_confidence))

        # Recompute overall as weighted average
        mem.confidence.overall = (
            0.4 * mem.confidence.memory_confidence +
            0.3 * mem.confidence.retrieval_confidence +
            0.15 * mem.confidence.temporal_confidence +
            0.15 * mem.confidence.source_confidence
        )

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        diagnostics = []
        for mem in ctx.memory_ir_list:
            if mem.importance < 0 or mem.importance > 1:
                diagnostics.append(Diagnostic(
                    stage=self.name, severity=DiagnosticSeverity.ERROR,
                    message=f"Memory {mem.identity.memory_id} has invalid importance: {mem.importance}",
                ))
            if mem.confidence.overall < 0 or mem.confidence.overall > 1:
                diagnostics.append(Diagnostic(
                    stage=self.name, severity=DiagnosticSeverity.ERROR,
                    message=f"Memory {mem.identity.memory_id} has invalid confidence: {mem.confidence.overall}",
                ))
        return diagnostics
