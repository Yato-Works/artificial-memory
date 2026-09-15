from __future__ import annotations

from datetime import datetime

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
    DiagnosticSeverity,
)
from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.ir import (
    CompressionMethod,
    CompressionRecord,
    MemoryIR,
    ResolutionLevel,
)


class CompressionStage(BaseStage):
    """Generate all resolution versions for each memory."""

    def __init__(self):
        super().__init__("compression")
        self.compressor = RuleBasedCompressor()

    def process(self, ctx: CompilerContext) -> CompilerContext:
        for mem in ctx.memory_ir_list:
            # Generate compressed versions for each resolution level
            self._generate_versions(mem)

        return ctx

    def _generate_versions(self, mem: MemoryIR) -> None:
        """Generate LIGHT, EPISODE, SEMANTIC, LONG_TERM, DEEP_LONG_TERM versions."""
        original_content = mem.semantic_content.content
        self.compressor.count_tokens(original_content)

        # Skip if already at highest compression
        if mem.resolution.value >= ResolutionLevel.LONG_TERM.value:
            return

        # LIGHT
        if mem.resolution.value < ResolutionLevel.LIGHT.value:
            light_content, light_meta = self.compressor.compress_light(original_content, {})
            mem.compression_history.append(CompressionRecord(
                from_resolution=mem.resolution,
                to_resolution=ResolutionLevel.LIGHT,
                original_tokens=light_meta['original_tokens'],
                compressed_tokens=light_meta['compressed_tokens'],
                compression_ratio=light_meta['compression_ratio'],
                method=CompressionMethod.LIGHT,
                timestamp=datetime.now(),
            ))

        # EPISODE
        if mem.resolution.value < ResolutionLevel.EPISODE.value:
            episode_content, episode_meta = self.compressor.compress_episode(original_content, {})
            mem.compression_history.append(CompressionRecord(
                from_resolution=ResolutionLevel.LIGHT if mem.resolution.value < ResolutionLevel.LIGHT.value else mem.resolution,
                to_resolution=ResolutionLevel.EPISODE,
                original_tokens=episode_meta['original_tokens'],
                compressed_tokens=episode_meta['compressed_tokens'],
                compression_ratio=episode_meta['compression_ratio'],
                method=CompressionMethod.EPISODE,
                timestamp=datetime.now(),
            ))

        # SEMANTIC
        if mem.resolution.value < ResolutionLevel.SEMANTIC.value:
            semantic_content, semantic_meta = self.compressor.compress_semantic(original_content, {})
            mem.compression_history.append(CompressionRecord(
                from_resolution=ResolutionLevel.EPISODE if mem.resolution.value < ResolutionLevel.EPISODE.value else mem.resolution,
                to_resolution=ResolutionLevel.SEMANTIC,
                original_tokens=semantic_meta['original_tokens'],
                compressed_tokens=semantic_meta['compressed_tokens'],
                compression_ratio=semantic_meta['compression_ratio'],
                method=CompressionMethod.SEMANTIC,
                timestamp=datetime.now(),
            ))

        # LONG_TERM
        if mem.resolution.value < ResolutionLevel.LONG_TERM.value:
            longterm_content, longterm_meta = self.compressor.compress_long_term(original_content, {})
            mem.compression_history.append(CompressionRecord(
                from_resolution=ResolutionLevel.SEMANTIC if mem.resolution.value < ResolutionLevel.SEMANTIC.value else mem.resolution,
                to_resolution=ResolutionLevel.LONG_TERM,
                original_tokens=longterm_meta['original_tokens'],
                compressed_tokens=longterm_meta['compressed_tokens'],
                compression_ratio=longterm_meta['compression_ratio'],
                method=CompressionMethod.LONG_TERM,
                timestamp=datetime.now(),
            ))

        # DEEP_LONG_TERM
        if mem.resolution.value < ResolutionLevel.DEEP_LONG_TERM.value:
            deep_content, deep_meta = self.compressor.compress_long_term(original_content, {})
            mem.compression_history.append(CompressionRecord(
                from_resolution=ResolutionLevel.LONG_TERM if mem.resolution.value < ResolutionLevel.LONG_TERM.value else mem.resolution,
                to_resolution=ResolutionLevel.DEEP_LONG_TERM,
                original_tokens=deep_meta['original_tokens'],
                compressed_tokens=deep_meta['compressed_tokens'],
                compression_ratio=deep_meta['compression_ratio'],
                method=CompressionMethod.DEEP_LONG_TERM,
                timestamp=datetime.now(),
            ))

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        diagnostics = []
        for mem in ctx.memory_ir_list:
            if mem.resolution.value < ResolutionLevel.DEEP_LONG_TERM.value:
                expected_versions = ResolutionLevel.DEEP_LONG_TERM.value - mem.resolution.value
                if len(mem.compression_history) < expected_versions:
                    diagnostics.append(Diagnostic(
                        stage=self.name, severity=DiagnosticSeverity.WARNING,
                        message=f"Memory {mem.identity.memory_id} missing compression versions "
                                f"({len(mem.compression_history)}/{expected_versions})",
                    ))
        return diagnostics
