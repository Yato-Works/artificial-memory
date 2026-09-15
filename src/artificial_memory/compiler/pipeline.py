from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from artificial_memory.context.ir_models import IRSequence, IRUnit
from artificial_memory.core.ir import (
    MemoryIR,
)
from artificial_memory.core.models import Conversation, Message


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Diagnostic:
    """Diagnostic message from a compiler stage."""
    stage: str
    severity: DiagnosticSeverity
    message: str
    location: str | None = None
    suggestion: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CompilerContext:
    """Context passed through compiler pipeline stages."""
    conversation: Conversation
    messages: list[Message]
    ir_sequence: IRSequence | None = None
    ir_units: list[IRUnit] = field(default_factory=list)
    memory_ir_list: list[MemoryIR] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_diagnostic(self, stage: str, severity: DiagnosticSeverity, message: str,
                       location: str = None, suggestion: str = None) -> None:
        self.diagnostics.append(Diagnostic(
            stage=stage, severity=severity, message=message,
            location=location, suggestion=suggestion
        ))

    def has_errors(self) -> bool:
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)


class CompilerStage(Protocol):
    """Protocol for compiler pipeline stages."""
    name: str

    def process(self, ctx: CompilerContext) -> CompilerContext: ...

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]: ...


class CompilerPipeline:
    """Deterministic compiler pipeline: Conversation -> MemoryIR[]"""

    # Bumped when stage behavior or ordering changes. Recorded in every
    # ProvenanceChain.compilation_chain entry for full reproducibility.
    COMPILER_VERSION = "compiler-v1"

    def __init__(self, stages: list[CompilerStage] | None = None):
        self.stages = stages or self._default_stages()

    def _default_stages(self) -> list[CompilerStage]:
        from .stages import (
            ClassificationStage,
            CompressionStage,
            EpisodeConstructionStage,
            FactDecisionIntentStage,
            LexicalAnalysisStage,
            OptimizationStage,
            ProvenanceLinkingStage,
            SemanticExtractionStage,
            StructuralAnalysisStage,
            TemporalLinkingStage,
        )
        return [
            LexicalAnalysisStage(),
            StructuralAnalysisStage(),
            SemanticExtractionStage(),
            FactDecisionIntentStage(),
            EpisodeConstructionStage(),
            TemporalLinkingStage(),
            ProvenanceLinkingStage(),
            ClassificationStage(),
            CompressionStage(),
            OptimizationStage(),
        ]

    def compile(self, conversation: Conversation, messages: list[Message]) -> list[MemoryIR]:
        """Run full pipeline on a conversation."""
        ir_list, _ = self.compile_with_diagnostics(conversation, messages)
        return ir_list

    def compile_with_diagnostics(self, conversation: Conversation,
                                  messages: list[Message]) -> tuple[list[MemoryIR], list[Diagnostic]]:
        """Compile and return both results and diagnostics."""
        ctx = CompilerContext(conversation=conversation, messages=messages)

        for stage in self.stages:
            try:
                ctx = stage.process(ctx)
                for diag in stage.validate_output(ctx):
                    ctx.diagnostics.append(diag)
            except Exception as e:
                ctx.add_diagnostic(
                    stage=stage.name,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Stage failed: {e}",
                )
            finally:
                # Record stage execution for provenance (P0-6)
                ctx.metadata.setdefault("stage_trace", []).append({
                    "stage": stage.name,
                    "compiler_version": self.COMPILER_VERSION,
                    "timestamp": datetime.now(),
                })

        # Populate ProvenanceChain.compilation_chain for every memory (P0-6)
        self._finalize_provenance(ctx)

        return ctx.memory_ir_list, ctx.diagnostics

    def _finalize_provenance(self, ctx: CompilerContext) -> None:
        """Attach the compilation chain (executed stages) to each MemoryIR."""
        from artificial_memory.core.ir import ProvenanceLink

        stage_trace = ctx.metadata.get("stage_trace", [])
        for mem in ctx.memory_ir_list:
            mem.source.compilation_chain = [
                ProvenanceLink(
                    timestamp=entry["timestamp"],
                    stage=entry["stage"],
                    compiler_version=entry["compiler_version"],
                )
                for entry in stage_trace
            ]


# Base stage class for common functionality
class BaseStage:
    """Base class for compiler stages."""

    def __init__(self, name: str):
        self.name = name

    def process(self, ctx: CompilerContext) -> CompilerContext:
        raise NotImplementedError

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        return []


def create_compiler_pipeline(stages: list[CompilerStage] | None = None) -> CompilerPipeline:
    """Factory function to create compiler pipeline."""
    return CompilerPipeline(stages)
