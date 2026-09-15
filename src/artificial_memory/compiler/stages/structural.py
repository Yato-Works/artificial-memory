from __future__ import annotations

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
    DiagnosticSeverity,
)
from artificial_memory.core.models import MessageRole


class StructuralAnalysisStage(BaseStage):
    """Structural analysis: role segmentation, turn boundaries, metadata extraction."""

    def __init__(self):
        super().__init__("structural_analysis")

    def process(self, ctx: CompilerContext) -> CompilerContext:
        # Analyze conversation structure
        user_messages = [m for m in ctx.messages if m.role == MessageRole.USER]
        assistant_messages = [m for m in ctx.messages if m.role == MessageRole.ASSISTANT]
        system_messages = [m for m in ctx.messages if m.role == MessageRole.SYSTEM]

        ctx.metadata["turn_count"] = len(ctx.messages)
        ctx.metadata["user_turns"] = len(user_messages)
        ctx.metadata["assistant_turns"] = len(assistant_messages)
        ctx.metadata["system_turns"] = len(system_messages)

        # First/last message analysis
        if ctx.messages:
            ctx.metadata["first_message_role"] = ctx.messages[0].role.value
            ctx.metadata["last_message_role"] = ctx.messages[-1].role.value
            ctx.metadata["first_user_msg_idx"] = next(
                (i for i, m in enumerate(ctx.messages) if m.role == MessageRole.USER), None
            )

        # Extract topic from first user message
        first_user = next((m for m in ctx.messages if m.role == MessageRole.USER), None)
        if first_user:
            ctx.metadata["inferred_topic"] = first_user.content[:100]

        return ctx

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        diagnostics = []
        if "turn_count" not in ctx.metadata:
            diagnostics.append(Diagnostic(
                stage=self.name, severity=DiagnosticSeverity.WARNING,
                message="Turn count not computed",
            ))
        return diagnostics
