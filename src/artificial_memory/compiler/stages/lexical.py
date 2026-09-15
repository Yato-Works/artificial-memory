from __future__ import annotations

import re

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
    DiagnosticSeverity,
)


class LexicalAnalysisStage(BaseStage):
    """Lexical analysis: tokenization, sentence splitting, language detection."""

    def __init__(self):
        super().__init__("lexical_analysis")
        # Compile sentence splitting regex (Japanese-aware)
        self.sentence_pattern = re.compile(r'(?<=[。！？.!?])\s+')

    def process(self, ctx: CompilerContext) -> CompilerContext:
        for msg in ctx.messages:
            sentences = self._split_sentences(msg.content)
            msg.metadata["sentences"] = sentences
            msg.metadata["sentence_count"] = len(sentences)
            msg.metadata["char_count"] = len(msg.content)
            msg.metadata["word_count"] = len(msg.content.split())

        # Simple language detection
        has_japanese = any(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', m.content)
                           for m in ctx.messages)
        ctx.metadata["language"] = "ja" if has_japanese else "en"

        return ctx

    def _split_sentences(self, text: str) -> list[str]:
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def validate_output(self, ctx: CompilerContext) -> list:
        diagnostics = []
        for msg in ctx.messages:
            if "sentences" not in msg.metadata:
                diagnostics.append(Diagnostic(
                    stage=self.name, severity=DiagnosticSeverity.WARNING,
                    message=f"Message {msg.id} missing sentence analysis",
                ))
        return diagnostics
