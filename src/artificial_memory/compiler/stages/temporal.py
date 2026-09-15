from __future__ import annotations

from datetime import datetime

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
    DiagnosticSeverity,
)
from artificial_memory.core.ir import (
    MemoryIR,
)


class TemporalLinkingStage(BaseStage):
    """Temporal linking: valid_from/valid_until, supersession chains."""

    def __init__(self):
        super().__init__("temporal_linking")

    def process(self, ctx: CompilerContext) -> CompilerContext:
        now = datetime.now()

        for mem in ctx.memory_ir_list:
            # Set valid_from to conversation start if not set
            if mem.temporal_scope.valid_from is None:
                mem.temporal_scope.valid_from = ctx.conversation.started_at

            # Set created_at/updated_at
            mem.temporal_scope.created_at = mem.temporal_scope.created_at or now
            mem.temporal_scope.updated_at = now

            # Link supersession for decisions
            if mem.type.value == "decision":
                self._link_supersession(mem, ctx.memory_ir_list)

            # Set default valid_until for CURRENT type (never expires)
            if mem.type.value == "current":
                mem.temporal_scope.valid_until = None

        return ctx

    def _link_supersession(self, decision_mem: MemoryIR, all_memories: list[MemoryIR]) -> None:
        """Link superseded decisions - newer decisions override older ones on same topic."""
        # Find other decisions on similar topics
        # For now, just mark the pattern - full implementation needs topic matching
        pass

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        diagnostics = []
        for mem in ctx.memory_ir_list:
            if mem.temporal_scope.valid_from is None:
                diagnostics.append(Diagnostic(
                    stage=self.name, severity=DiagnosticSeverity.WARNING,
                    message=f"Memory {mem.identity.memory_id} missing valid_from",
                ))
        return diagnostics
