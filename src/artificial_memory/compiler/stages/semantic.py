from __future__ import annotations

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
    DiagnosticSeverity,
)
from artificial_memory.context.ir_compiler import RuleBasedIRCompiler


class SemanticExtractionStage(BaseStage):
    """Semantic extraction: deterministic fact/decision/entity extraction using rules."""

    def __init__(self):
        super().__init__("semantic_extraction")
        self.ir_compiler = RuleBasedIRCompiler()

    def process(self, ctx: CompilerContext) -> CompilerContext:
        # Use existing IR compiler to extract units
        ir_sequence = self.ir_compiler.compile_to_ir(ctx.messages)
        ctx.ir_sequence = ir_sequence
        ctx.ir_units = ir_sequence.units

        # Count extracted units by type
        type_counts = {}
        for unit in ir_sequence.units:
            type_counts[unit.ir_type] = type_counts.get(unit.ir_type, 0) + 1

        ctx.metadata["ir_unit_count"] = len(ir_sequence.units)
        ctx.metadata["ir_type_distribution"] = type_counts

        return ctx

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        diagnostics = []
        if not ctx.ir_sequence:
            diagnostics.append(Diagnostic(
                stage=self.name, severity=DiagnosticSeverity.ERROR,
                message="IR sequence not generated",
            ))
        elif len(ctx.ir_sequence.units) == 0:
            diagnostics.append(Diagnostic(
                stage=self.name, severity=DiagnosticSeverity.WARNING,
                message="No IR units extracted from conversation",
                suggestion="Check if conversation has meaningful content",
            ))
        return diagnostics
