from __future__ import annotations

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
    DiagnosticSeverity,
)
from artificial_memory.core.ir import MemoryIR
from artificial_memory.core.models import MessageRole


class ProvenanceLinkingStage(BaseStage):
    """Attach source conversation/message IDs to all memories."""

    def __init__(self):
        super().__init__("provenance_linking")

    def process(self, ctx: CompilerContext) -> CompilerContext:
        conv = ctx.conversation

        for mem in ctx.memory_ir_list:
            # Ensure source link exists
            if mem.source.source.conversation_id is None:
                mem.source.source.conversation_id = conv.id
                mem.source.source.timestamp = conv.started_at

            # Find the specific source message if possible
            if mem.source.source.message_id is None:
                source_msg = self._find_source_message(mem, ctx.messages)
                if source_msg:
                    mem.source.source.message_id = source_msg.id

        return ctx

    def _find_source_message(self, mem: MemoryIR, messages: list) -> any:
        """Find the message that most likely originated this memory."""
        if not mem.semantic_content.content:
            return None

        # Simple heuristic: find message with highest content overlap
        best_match = None
        best_score = 0

        for msg in messages:
            if msg.role != MessageRole.ASSISTANT:
                continue

            # Calculate word overlap
            mem_words = set(mem.semantic_content.content.lower().split())
            msg_words = set(msg.content.lower().split())
            overlap = len(mem_words & msg_words)

            if overlap > best_score:
                best_score = overlap
                best_match = msg

        return best_match if best_score > 3 else None

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        diagnostics = []
        for mem in ctx.memory_ir_list:
            if mem.source.source.conversation_id is None:
                diagnostics.append(Diagnostic(
                    stage=self.name, severity=DiagnosticSeverity.WARNING,
                    message=f"Memory {mem.identity.memory_id} missing conversation_id in provenance",
                ))
        return diagnostics
