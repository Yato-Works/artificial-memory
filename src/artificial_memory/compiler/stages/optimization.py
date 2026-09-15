from __future__ import annotations

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
)
from artificial_memory.core.ir import LifecycleState, MemoryIR


class OptimizationStage(BaseStage):
    """Deduplicate, merge similar, assign final LifecycleState."""

    def __init__(self):
        super().__init__("optimization")

    def process(self, ctx: CompilerContext) -> CompilerContext:
        # Deduplicate memories with very similar content
        ctx.memory_ir_list = self._deduplicate(ctx.memory_ir_list)

        # Merge similar memories of same type
        ctx.memory_ir_list = self._merge_similar(ctx.memory_ir_list)

        # Assign final lifecycle state based on content and type
        for mem in ctx.memory_ir_list:
            mem.lifecycle_state = self._assign_lifecycle(mem)

        return ctx

    def _deduplicate(self, memories: list[MemoryIR]) -> list[MemoryIR]:
        """Remove memories with nearly identical content."""
        if len(memories) <= 1:
            return memories

        unique = []
        seen_hashes = set()

        for mem in memories:
            content_hash = mem.identity.content_hash
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(mem)
            else:
                # Mark as duplicate in metadata
                mem.metadata["duplicate_of"] = content_hash

        return unique

    def _merge_similar(self, memories: list[MemoryIR]) -> list[MemoryIR]:
        """Merge memories of same type with high content overlap."""
        # For now, just return as-is - full merge logic is complex
        # Would need semantic similarity comparison
        return memories

    def _assign_lifecycle(self, mem: MemoryIR) -> LifecycleState:
        """Assign lifecycle state based on type, importance, age."""
        # CURRENT memories are always HOT
        if mem.type.value == "current":
            return LifecycleState.HOT

        # DECISION memories start HOT, cool to WARM
        if mem.type.value == "decision":
            return LifecycleState.HOT

        # TIMELINE memories are WARM
        if mem.type.value == "timeline":
            return LifecycleState.WARM

        # SEMANTIC/EPISODE: base on importance
        if mem.importance >= 0.8:
            return LifecycleState.HOT
        elif mem.importance >= 0.5:
            return LifecycleState.WARM
        else:
            return LifecycleState.COLD

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        return []
