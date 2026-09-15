from __future__ import annotations

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
)
from artificial_memory.core.ir import MemoryIR, MemoryType, ResolutionLevel


class EpisodeConstructionStage(BaseStage):
    """Group related IR units into episodic memories and create LIGHT/EPISODE summaries."""

    def __init__(self):
        super().__init__("episode_construction")

    def process(self, ctx: CompilerContext) -> CompilerContext:
        # Create LIGHT compression of full conversation
        light_memory = self._create_light_memory(ctx)
        if light_memory:
            ctx.memory_ir_list.append(light_memory)

        # Ensure EPISODE memories exist (may have been created in FactDecisionIntentStage)
        # This stage mainly ensures proper episode-level organization

        return ctx

    def _create_light_memory(self, ctx: CompilerContext) -> MemoryIR | None:
        """Create LIGHT resolution memory from full conversation."""
        if not ctx.messages:
            return None

        import hashlib
        from datetime import datetime

        from artificial_memory.compression.compressor import RuleBasedCompressor
        from artificial_memory.core.ir import (
            AccessRecord,
            ConfidenceProfile,
            LifecycleState,
            MemoryIdentity,
            ProvenanceChain,
            ProvenanceLink,
            SemanticContent,
            TemporalScope,
        )

        compressor = RuleBasedCompressor()
        full_content = '\n'.join([f"{m.role.value}: {m.content}" for m in ctx.messages])
        light_content, light_meta = compressor.compress_light(full_content, {})

        now = datetime.now()
        content_hash = hashlib.sha256(light_content.encode()).hexdigest()[:16]

        return MemoryIR(
            identity=MemoryIdentity(memory_id=0, version=1, content_hash=content_hash),
            type=MemoryType.EPISODE,
            resolution=ResolutionLevel.LIGHT,
            semantic_content=SemanticContent(content=light_content),
            source=ProvenanceChain(
                source=ProvenanceLink(
                    conversation_id=ctx.conversation.id,
                    message_id=None,
                    timestamp=ctx.conversation.started_at,
                )
            ),
            temporal_scope=TemporalScope(
                valid_from=ctx.conversation.started_at,
                valid_until=None,
                created_at=now,
                updated_at=now,
            ),
            confidence=ConfidenceProfile(
                memory_confidence=0.75, retrieval_confidence=0.7,
                temporal_confidence=1.0, source_confidence=0.9, overall=0.75
            ),
            importance=0.7,
            dependencies=[], relations=[], compression_history=[],
            access_history=AccessRecord(),
            lifecycle_state=LifecycleState.HOT,
        )

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        return []
