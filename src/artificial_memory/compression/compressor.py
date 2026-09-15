from __future__ import annotations

import re
import time
from dataclasses import dataclass

import tiktoken

from artificial_memory.core.interfaces import Compressor
from artificial_memory.metrics.collector import (
    CompressionMetrics,
    get_metrics_collector,
)


@dataclass
class CompressionResult:
    content: str
    metadata: dict
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float


class RuleBasedCompressor:
    """Rule-based compressor for MVP - no LLM calls."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

        # Patterns for filler words and redundancy
        self.filler_patterns = [
            r'\b(いや[〜～]?|えーっと|えーと|あの[〜～]?|うーん|まぁ|なんか|ちょっと|一応|一応|一応)\b',
            r'\b(だと思う|気がする|と思う|でしょうか|ですよね|じゃないですか)\b',
            r'\b(普通に|結構|まあ|結構|わりと|そこそこ|そこそこ)\b',
            r'(w{2,}|笑|草|www)',
        ]

        # Patterns for important markers (keep these)
        self.important_markers = [
            r'(決めた|決定|採用|却下|選択|選んだ|決断)',
            r'(理由|なぜなら|なぜ|なぜか|理由は)',
            r'(重要|大事|クリティカル|必須|マスト)',
            r'(問題|課題|懸念|リスク|不安)',
            r'(メリット|デメリット|利点|欠点|長所|短所)',
            r'(比較|検討|比較検討|トレードオフ)',
            r'(アーキテクチャ|設計|設計方針|方針)',
        ]

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def get_compression_ratio(self, original: str, compressed: str) -> float:
        orig = self.count_tokens(original)
        comp = self.count_tokens(compressed)
        return orig / comp if comp > 0 else 1.0

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words while preserving sentence structure."""
        result = text
        for pattern in self.filler_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s*([、。．，！？])\s*', r'\1 ', result)
        return result.strip()

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences (Japanese-aware)."""
        sentences = re.split(r'(?<=[。！？.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentence_importance(self, sentence: str) -> float:
        """Score sentence importance based on markers."""
        score = 0.5  # Base score
        sentence.lower()

        for pattern in self.important_markers:
            if re.search(pattern, sentence, re.IGNORECASE):
                score += 0.2

        # Longer sentences tend to have more content
        length_bonus = min(len(sentence) / 200, 0.3)
        score += length_bonus

        # Questions and decisions are important
        if '？' in sentence or '?' in sentence:
            score += 0.15
        if any(w in sentence for w in ['決めた', '決定', '採用', '選択']):
            score += 0.25

        return min(score, 1.0)

    def compress_light(self, content: str, metadata: dict) -> tuple[str, dict]:
        """Light compression: remove fillers, keep almost everything."""
        start_time = time.perf_counter()
        original_tokens = self.count_tokens(content)

        # Remove fillers
        compressed = self._remove_fillers(content)

        # Remove duplicate consecutive sentences
        sentences = self._split_sentences(compressed)
        unique_sentences = []
        prev = None
        for s in sentences:
            if s != prev:
                unique_sentences.append(s)
                prev = s

        compressed = ' '.join(unique_sentences)
        compressed_tokens = self.count_tokens(compressed)
        compression_time_ms = (time.perf_counter() - start_time) * 1000

        meta = {
            **metadata,
            "compression_method": "light",
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": self.get_compression_ratio(content, compressed),
            "sentences_removed": len(sentences) - len(unique_sentences),
        }

        # Record metrics
        collector = get_metrics_collector()
        collector.record_compression(CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=meta["compression_ratio"],
            compression_time_ms=compression_time_ms,
            method="light",
        ))

        return compressed, meta

    def compress_episode(self, content: str, metadata: dict) -> tuple[str, dict]:
        """Episode compression: summarize what happened."""
        start_time = time.perf_counter()
        original_tokens = self.count_tokens(content)
        sentences = self._split_sentences(content)

        # Score all sentences
        scored = [(s, self._score_sentence_importance(s)) for s in sentences]
        scored.sort(key=lambda x: -x[1])

        # Keep top 40% of sentences by importance
        keep_count = max(3, int(len(sentences) * 0.4))
        kept = [s for s, _ in scored[:keep_count]]

        # Restore original order
        kept_set = set(kept)
        ordered_kept = [s for s in sentences if s in kept_set]

        compressed = ' '.join(ordered_kept)
        compressed_tokens = self.count_tokens(compressed)
        compression_time_ms = (time.perf_counter() - start_time) * 1000

        meta = {
            **metadata,
            "compression_method": "episode",
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": self.get_compression_ratio(content, compressed),
            "sentences_kept": len(ordered_kept),
            "sentences_total": len(sentences),
        }

        # Record metrics
        collector = get_metrics_collector()
        collector.record_compression(CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=meta["compression_ratio"],
            compression_time_ms=compression_time_ms,
            method="episode",
        ))

        return compressed, meta

    def compress_semantic(self, content: str, metadata: dict) -> tuple[str, dict]:
        """Semantic compression: extract key decisions and outcomes."""
        start_time = time.perf_counter()
        original_tokens = self.count_tokens(content)
        sentences = self._split_sentences(content)

        # Find decision/outcome sentences
        decision_sentences = []
        reason_sentences = []
        other_important = []

        for s in sentences:
            s_lower = s.lower()
            if any(w in s_lower for w in ['決めた', '決定', '採用', '却下', '選択', '選んだ']):
                decision_sentences.append(s)
            elif any(w in s_lower for w in ['理由', 'なぜなら', 'なぜ', 'ため', 'から']):
                reason_sentences.append(s)
            elif self._score_sentence_importance(s) > 0.7:
                other_important.append(s)

        # Build semantic summary
        parts = []
        if decision_sentences:
            parts.append("**決定事項:**")
            parts.extend(decision_sentences[:3])
        if reason_sentences:
            parts.append("\n**理由:**")
            parts.extend(reason_sentences[:3])
        if other_important:
            parts.append("\n**重要ポイント:**")
            parts.extend(other_important[:2])

        compressed = '\n'.join(parts)
        compressed_tokens = self.count_tokens(compressed)
        compression_time_ms = (time.perf_counter() - start_time) * 1000

        meta = {
            **metadata,
            "compression_method": "semantic",
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": self.get_compression_ratio(content, compressed),
            "decisions_found": len(decision_sentences),
            "reasons_found": len(reason_sentences),
        }

        # Record metrics
        collector = get_metrics_collector()
        collector.record_compression(CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=meta["compression_ratio"],
            compression_time_ms=compression_time_ms,
            method="semantic",
        ))

        return compressed, meta

    def compress_long_term(self, content: str, metadata: dict) -> tuple[str, dict]:
        """Long-term compression: highly abstract summary."""
        start_time = time.perf_counter()
        original_tokens = self.count_tokens(content)
        sentences = self._split_sentences(content)

        # Only keep highest importance sentences
        scored = [(s, self._score_sentence_importance(s)) for s in sentences]
        scored.sort(key=lambda x: -x[1])

        # Keep top 2 sentences max
        kept = [s for s, score in scored[:2] if score > 0.6]

        if not kept:
            # Fallback: first and last sentence
            kept = [sentences[0]] if sentences else []
            if len(sentences) > 1:
                kept.append(sentences[-1])

        compressed = ' '.join(kept)
        compressed_tokens = self.count_tokens(compressed)
        compression_time_ms = (time.perf_counter() - start_time) * 1000

        meta = {
            **metadata,
            "compression_method": "longterm",
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": self.get_compression_ratio(content, compressed),
            "sentences_kept": len(kept),
        }

        # Record metrics
        collector = get_metrics_collector()
        collector.record_compression(CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=meta["compression_ratio"],
            compression_time_ms=compression_time_ms,
            method="longterm",
        ))

        return compressed, meta


def create_compressor(encoding_name: str = "cl100k_base") -> Compressor:
    """Factory function to create compressor."""
    return RuleBasedCompressor(encoding_name)
