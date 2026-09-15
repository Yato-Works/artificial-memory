from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory, ResolutionLevel


class ValidationStatus(StrEnum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ValidationRule(StrEnum):
    """Types of validation rules."""
    MIN_COMPRESSION_RATIO = "min_compression_ratio"       # At least 1.2x
    MAX_COMPRESSION_RATIO = "max_compression_ratio"       # Not over 20x
    CONTENT_NOT_EMPTY = "content_not_empty"               # Compressed not empty
    NO_EXCESSIVE_TRUNCATION = "no_excessive_truncation"   # Not truncated > 80%
    SEMANTIC_PRESERVATION = "semantic_preservation"       # Key info preserved
    DECISION_PRESERVATION = "decision_preservation"       # Decisions kept
    ENTITY_PRESERVATION = "entity_preservation"           # Entities kept
    NO_HALLUCINATION = "no_hallucination"                 # No new info added


@dataclass
class ValidationResult:
    """Result of a single validation rule."""
    rule: ValidationRule
    status: ValidationStatus
    score: float  # 0-1
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionValidationReport:
    """Full validation report for a memory's compression."""
    memory_id: int
    original_resolution: ResolutionLevel
    target_resolution: ResolutionLevel
    original_content: str
    compressed_content: str
    compression_ratio: float
    method: str
    overall_status: ValidationStatus
    rule_results: list[ValidationResult] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)

    @property
    def passed(self) -> bool:
        return self.overall_status == ValidationStatus.PASSED

    @property
    def has_warnings(self) -> bool:
        return any(r.status == ValidationStatus.WARNING for r in self.rule_results)

    @property
    def has_failures(self) -> bool:
        return any(r.status == ValidationStatus.FAILED for r in self.rule_results)


class CompressionValidator:
    """Validates compression quality across multiple dimensions."""

    def __init__(
        self,
        store: MemoryStore,
        compressor: RuleBasedCompressor,
        config: dict[str, Any] | None = None,
    ):
        self.store = store
        self.compressor = compressor
        self.config = config or {}

        # Configurable thresholds
        self.min_compression_ratio = self.config.get("min_compression_ratio", 1.2)
        self.max_compression_ratio = self.config.get("max_compression_ratio", 20.0)
        self.max_truncation_ratio = self.config.get("max_truncation_ratio", 0.8)
        self.min_semantic_score = self.config.get("min_semantic_score", 0.5)

    def validate_compression(
        self,
        original: Memory,
        target_resolution: ResolutionLevel,
    ) -> CompressionValidationReport:
        """Validate a single compression operation."""
        # Get or create compressed version
        version = self.store.get_memory_version(original.id, target_resolution)

        if not version:
            # Compress on the fly for validation
            compressed_content, meta = self._compress_to_resolution(
                original.content, target_resolution
            )
            compression_ratio = meta.get("compression_ratio", 1.0)
        else:
            compressed_content = version.content
            compression_ratio = version.compression_ratio or 1.0

        # Run all validation rules
        rule_results = []

        # Rule 1: Min compression ratio
        rule_results.append(self._check_min_compression_ratio(compression_ratio))

        # Rule 2: Max compression ratio
        rule_results.append(self._check_max_compression_ratio(compression_ratio))

        # Rule 3: Content not empty
        rule_results.append(self._check_content_not_empty(compressed_content))

        # Rule 4: No excessive truncation
        rule_results.append(self._check_truncation(original.content, compressed_content))

        # Rule 5: Semantic preservation
        rule_results.append(self._check_semantic_preservation(
            original.content, compressed_content
        ))

        # Rule 6: Decision preservation (for DECISION memories)
        if original.memory_type.value == "decision":
            rule_results.append(self._check_decision_preservation(
                original.content, compressed_content
            ))

        # Rule 7: Entity preservation
        rule_results.append(self._check_entity_preservation(
            original.content, compressed_content
        ))

        # Rule 8: No hallucination
        rule_results.append(self._check_no_hallucination(
            original.content, compressed_content
        ))

        # Determine overall status
        has_failed = any(r.status == ValidationStatus.FAILED for r in rule_results)
        has_warnings = any(r.status == ValidationStatus.WARNING for r in rule_results)

        if has_failed:
            overall_status = ValidationStatus.FAILED
        elif has_warnings:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED

        return CompressionValidationReport(
            memory_id=original.id,
            original_resolution=original.resolution,
            target_resolution=target_resolution,
            original_content=original.content,
            compressed_content=compressed_content,
            compression_ratio=compression_ratio,
            method=target_resolution.name,
            overall_status=overall_status,
            rule_results=rule_results,
        )

    def validate_all_versions(self, memory: Memory) -> list[CompressionValidationReport]:
        """Validate all compressed versions of a memory."""
        reports = []
        versions = self.store.get_memory_versions(memory.id)

        for version in versions:
            report = self.validate_compression(memory, version.resolution)
            reports.append(report)

        return reports

    # ==================== Rule Implementations ====================

    def _check_min_compression_ratio(self, ratio: float) -> ValidationResult:
        if ratio >= self.min_compression_ratio:
            return ValidationResult(
                rule=ValidationRule.MIN_COMPRESSION_RATIO,
                status=ValidationStatus.PASSED,
                score=1.0,
                message=f"Compression ratio {ratio:.2f}x >= {self.min_compression_ratio}x",
            )
        else:
            return ValidationResult(
                rule=ValidationRule.MIN_COMPRESSION_RATIO,
                status=ValidationStatus.WARNING,
                score=ratio / self.min_compression_ratio,
                message=f"Compression ratio {ratio:.2f}x below minimum {self.min_compression_ratio}x",
            )

    def _check_max_compression_ratio(self, ratio: float) -> ValidationResult:
        if ratio <= self.config.get("max_compression_ratio", 20.0):
            return ValidationResult(
                rule=ValidationRule.MAX_COMPRESSION_RATIO,
                status=ValidationStatus.PASSED,
                score=1.0,
                message=f"Compression ratio {ratio:.2f}x within limit",
            )
        else:
            return ValidationResult(
                rule=ValidationRule.MAX_COMPRESSION_RATIO,
                status=ValidationStatus.FAILED,
                score=max(0, 1 - (ratio - 20) / 100),
                message=f"Compression ratio {ratio:.2f}x exceeds maximum - likely over-compressed",
            )

    def _check_content_not_empty(self, content: str) -> ValidationResult:
        if content and content.strip():
            return ValidationResult(
                rule=ValidationRule.CONTENT_NOT_EMPTY,
                status=ValidationStatus.PASSED,
                score=1.0,
                message="Compressed content is not empty",
            )
        else:
            return ValidationResult(
                rule=ValidationRule.CONTENT_NOT_EMPTY,
                status=ValidationStatus.FAILED,
                score=0.0,
                message="Compressed content is empty!",
            )

    def _check_truncation(self, original: str, compressed: str) -> ValidationResult:
        if not original:
            return ValidationResult(
                rule=ValidationRule.NO_EXCESSIVE_TRUNCATION,
                status=ValidationStatus.PASSED,
                score=1.0,
                message="No original content to compare",
            )

        truncation_ratio = 1 - (len(compressed) / len(original))

        if truncation_ratio <= self.config.get("max_truncation_ratio", 0.8):
            return ValidationResult(
                rule=ValidationRule.NO_EXCESSIVE_TRUNCATION,
                status=ValidationStatus.PASSED,
                score=1 - truncation_ratio,
                message=f"Truncation {truncation_ratio:.1%} within limit",
            )
        else:
            return ValidationResult(
                rule=ValidationRule.NO_EXCESSIVE_TRUNCATION,
                status=ValidationStatus.FAILED,
                score=max(0, 1 - truncation_ratio),
                message=f"Excessive truncation: {truncation_ratio:.1%} of original removed",
            )

    def _check_semantic_preservation(self, original: str, compressed: str) -> ValidationResult:
        """Check if key semantic content is preserved."""
        if not original or not compressed:
            return ValidationResult(
                rule=ValidationRule.SEMANTIC_PRESERVATION,
                status=ValidationStatus.FAILED,
                score=0.0,
                message="Cannot compare - empty content",
            )

        # Extract key phrases from original
        key_phrases = self._extract_key_phrases(original)

        if not key_phrases:
            return ValidationResult(
                rule=ValidationRule.SEMANTIC_PRESERVATION,
                status=ValidationStatus.PASSED,
                score=1.0,
                message="No key phrases to check",
            )

        preserved = sum(1 for phrase in key_phrases if phrase.lower() in compressed.lower())
        score = preserved / len(key_phrases) if key_phrases else 1.0

        if score >= 0.7:
            return ValidationResult(
                rule=ValidationRule.SEMANTIC_PRESERVATION,
                status=ValidationStatus.PASSED,
                score=score,
                message=f"{score:.0%} of key phrases preserved",
            )
        elif score >= 0.4:
            return ValidationResult(
                rule=ValidationRule.SEMANTIC_PRESERVATION,
                status=ValidationStatus.WARNING,
                score=score,
                message=f"Only {score:.0%} of key phrases preserved",
            )
        else:
            return ValidationResult(
                rule=ValidationRule.SEMANTIC_PRESERVATION,
                status=ValidationStatus.FAILED,
                score=score,
                message=f"Only {score:.0%} of key phrases preserved - significant loss",
            )

    def _check_decision_preservation(self, original: str, compressed: str) -> ValidationResult:
        """Check if decision statements are preserved."""
        decision_keywords = [
            "決めた", "決定", "採用", "却下", "選択", "選んだ", "決断",
            "decided", "adopted", "rejected", "selected", "chose", "go with",
        ]

        original_decisions = [kw for kw in decision_keywords if kw in original.lower()]
        if not original_decisions:
            return ValidationResult(
                rule=ValidationRule.DECISION_PRESERVATION,
                status=ValidationStatus.PASSED,
                score=1.0,
                message="No decision keywords in original",
            )

        preserved = sum(1 for kw in original_decisions if kw in compressed.lower())
        score = preserved / len(original_decisions) if original_decisions else 1.0

        if score >= 0.8:
            return ValidationResult(
                rule=ValidationRule.DECISION_PRESERVATION,
                status=ValidationStatus.PASSED,
                score=score,
                message=f"Decision keywords preserved: {score:.0%}",
            )
        elif score >= 0.5:
            return ValidationResult(
                rule=ValidationRule.DECISION_PRESERVATION,
                status=ValidationStatus.WARNING,
                score=score,
                message=f"Only {score:.0%} of decision keywords preserved",
            )
        else:
            return ValidationResult(
                rule=ValidationRule.DECISION_PRESERVATION,
                status=ValidationStatus.FAILED,
                score=score,
                message=f"Critical: only {score:.0%} of decision keywords preserved",
            )

    def _check_entity_preservation(self, original: str, compressed: str) -> ValidationResult:
        """Check if named entities are preserved."""
        import re

        # Extract capitalized words (potential entities)
        entities_orig = set(re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b', original))
        entities_comp = set(re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b', compressed))

        # Filter common words
        stop_words = {
            'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'Why', 'How',
            'We', 'You', 'They', 'It', 'Is', 'Are', 'Was', 'Were', 'Have', 'Has',
            'Can', 'Will', 'Would', 'Could', 'Should', 'Must', 'Let', 'Also', 'But',
        }

        entities_orig = {e for e in entities_orig if e not in stop_words and len(e) > 2}
        entities_comp = {e for e in entities_comp if e not in stop_words and len(e) > 2}

        if not entities_orig:
            return ValidationResult(
                rule=ValidationRule.ENTITY_PRESERVATION,
                status=ValidationStatus.PASSED,
                score=1.0,
                message="No entities in original to check",
            )

        preserved = len(entities_orig & entities_comp)
        total = len(entities_orig)
        score = preserved / total if total > 0 else 1.0

        if score >= 0.7:
            return ValidationResult(
                rule=ValidationRule.ENTITY_PRESERVATION,
                status=ValidationStatus.PASSED,
                score=score,
                message=f"{preserved}/{total} entities preserved",
            )
        elif score >= 0.4:
            return ValidationResult(
                rule=ValidationRule.ENTITY_PRESERVATION,
                status=ValidationStatus.WARNING,
                score=score,
                message=f"Only {preserved}/{total} entities preserved",
            )
        else:
            return ValidationResult(
                rule=ValidationRule.ENTITY_PRESERVATION,
                status=ValidationStatus.FAILED,
                score=score,
                message=f"Only {preserved}/{total} entities preserved",
            )

    def _check_no_hallucination(self, original: str, compressed: str) -> ValidationResult:
        """Check if compressed version adds information not in original."""
        # Simple check: words in compressed that aren't in original
        orig_words = set(original.lower().split())
        comp_words = set(compressed.lower().split())

        new_words = comp_words - orig_words

        # Filter out common filler words
        filler = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'was', 'are', 'were',
                  'to', 'of', 'in', 'on', 'for', 'with', 'by', 'as', 'at', 'from',
                  'it', 'this', 'that', 'these', 'those', 'we', 'you', 'they', 'i'}

        new_words = {w for w in new_words if w not in filler and len(w) > 2}

        if not new_words:
            return ValidationResult(
                rule=ValidationRule.NO_HALLUCINATION,
                status=ValidationStatus.PASSED,
                score=1.0,
                message="No new information detected",
            )
        elif len(new_words) <= 3:
            return ValidationResult(
                rule=ValidationRule.NO_HALLUCINATION,
                status=ValidationStatus.WARNING,
                score=0.8,
                message=f"Minor new words: {', '.join(list(new_words)[:5])}",
            )
        else:
            return ValidationResult(
                rule=ValidationRule.NO_HALLUCINATION,
                status=ValidationStatus.FAILED,
                score=max(0, 1 - len(new_words) / 10),
                message=f"Potential hallucination: {len(new_words)} new words added",
            )

    def _compress_to_resolution(self, content: str, target: ResolutionLevel) -> tuple[str, dict]:
        """Compress content to target resolution."""
        method_map = {
            ResolutionLevel.LIGHT: (self.compressor.compress_light, "light"),
            ResolutionLevel.EPISODE: (self.compressor.compress_episode, "episode"),
            ResolutionLevel.SEMANTIC: (self.compressor.compress_semantic, "semantic"),
            ResolutionLevel.LONG_TERM: (self.compressor.compress_long_term, "longterm"),
            ResolutionLevel.DEEP_LONG_TERM: (self.compressor.compress_long_term, "deeplongterm"),
        }

        func, method = method_map.get(target, (self.compressor.compress_light, "light"))
        content_compressed, meta = func(content, {})
        meta["method"] = method
        return content_compressed, meta

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract key phrases from text."""
        import re

        phrases = []

        # Decision phrases
        for match in re.finditer(r'(決めた|決定|採用|却下|選択|decided|adopted|rejected|selected)[^。.]*[。.]', text):
            phrases.append(match.group(0))

        # Reason phrases
        for match in re.finditer(r'(理由|なぜなら|なぜ|because|reason|since)[^。.]*[。.]', text, re.IGNORECASE):
            phrases.append(match.group(0))

        # Key entity + action
        for match in re.finditer(r'[A-Z][a-z]{2,}\s+(?:は|が|を|に|で|が)\s*[^。.]*[。.]', text):
            phrases.append(match.group(0))

        return list(set(phrases))[:10]


def create_compression_validator(
    store: MemoryStore,
    compressor: RuleBasedCompressor,
    config: dict[str, Any] | None = None,
) -> CompressionValidator:
    return CompressionValidator(store, compressor, config)
