from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class MetricType(StrEnum):
    """Types of metrics to track."""
    COMPRESSION_RATIO = "compression_ratio"
    RECALL_ACCURACY = "recall_accuracy"
    TEMPORAL_ACCURACY = "temporal_accuracy"
    TOKEN_REDUCTION = "token_reduction"
    LATENCY_MS = "latency_ms"
    STORAGE_SIZE = "storage_size"
    RETRIEVAL_ACCURACY = "retrieval_accuracy"
    DECISION_PRESERVATION = "decision_preservation"
    CONVERSATION_STYLE_PRESERVATION = "conversation_style_preservation"
    CONTEXT_EFFECTIVENESS = "context_effectiveness"
    COMPRESSION_SPEED = "compression_speed"
    RECALL_SPEED = "recall_speed"


@dataclass
class MetricValue:
    """A single metric measurement."""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict = field(default_factory=dict)
    conversation_id: int | None = None
    topic_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "conversation_id": self.conversation_id,
            "topic_id": self.topic_id,
        }


@dataclass
class CompressionMetrics:
    """Aggregated compression metrics."""
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    compression_time_ms: float = 0.0
    method: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    conversation_id: int | None = None

    @property
    def token_reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1.0 - self.compressed_tokens / self.original_tokens) * 100


@dataclass
class RecallMetrics:
    """Aggregated recall metrics."""
    query: str = ""
    recall_level: str = ""
    memories_retrieved: int = 0
    tokens_returned: int = 0
    latency_ms: float = 0.0
    relevance_scores: list[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    conversation_id: int | None = None
    topic_id: int | None = None


@dataclass
class ContextMetrics:
    """Aggregated context building metrics."""
    raw_tokens: int = 0
    effective_tokens: int = 0
    compression_ratio: float = 0.0
    parts_count: int = 0
    selected_parts: int = 0
    tier_distribution: dict[str, int] = field(default_factory=dict)
    build_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    conversation_id: int | None = None
    topic_id: int | None = None


class MetricsCollector:
    """Collects and aggregates metrics for the Artificial Memory system."""

    def __init__(self):
        self.metrics: list[MetricValue] = []
        self.compression_history: list[CompressionMetrics] = []
        self.recall_history: list[RecallMetrics] = []
        self.context_history: list[ContextMetrics] = []

    def record_compression(self, metrics: CompressionMetrics) -> None:
        """Record compression metrics."""
        self.compression_history.append(metrics)
        self.metrics.append(MetricValue(
            metric_type=MetricType.COMPRESSION_RATIO,
            value=metrics.compression_ratio,
            unit="ratio",
            context={"method": metrics.method, "original_tokens": metrics.original_tokens},
            conversation_id=metrics.conversation_id,
        ))
        self.metrics.append(MetricValue(
            metric_type=MetricType.TOKEN_REDUCTION,
            value=metrics.token_reduction_pct,
            unit="percent",
            context={"method": metrics.method},
            conversation_id=metrics.conversation_id,
        ))
        self.metrics.append(MetricValue(
            metric_type=MetricType.COMPRESSION_SPEED,
            value=metrics.compression_time_ms,
            unit="ms",
            context={"method": metrics.method},
            conversation_id=metrics.conversation_id,
        ))

    def record_recall(self, metrics: RecallMetrics) -> None:
        """Record recall metrics."""
        self.recall_history.append(metrics)
        self.metrics.append(MetricValue(
            metric_type=MetricType.LATENCY_MS,
            value=metrics.latency_ms,
            unit="ms",
            context={"recall_level": metrics.recall_level, "memories_retrieved": metrics.memories_retrieved},
            conversation_id=metrics.conversation_id,
            topic_id=metrics.topic_id,
        ))

        if metrics.relevance_scores:
            avg_relevance = sum(metrics.relevance_scores) / len(metrics.relevance_scores)
            self.metrics.append(MetricValue(
                metric_type=MetricType.RETRIEVAL_ACCURACY,
                value=avg_relevance,
                unit="score",
                context={"recall_level": metrics.recall_level},
                conversation_id=metrics.conversation_id,
                topic_id=metrics.topic_id,
            ))

    def record_context(self, metrics: ContextMetrics) -> None:
        """Record context building metrics."""
        self.context_history.append(metrics)
        self.metrics.append(MetricValue(
            metric_type=MetricType.CONTEXT_EFFECTIVENESS,
            value=metrics.compression_ratio,
            unit="ratio",
            context={
                "parts_count": metrics.parts_count,
                "selected_parts": metrics.selected_parts,
                "tier_distribution": metrics.tier_distribution,
            },
            conversation_id=metrics.conversation_id,
            topic_id=metrics.topic_id,
        ))

    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        if not self.metrics:
            return {}

        by_type = {}
        for m in self.metrics:
            if m.metric_type.value not in by_type:
                by_type[m.metric_type.value] = []
            by_type[m.metric_type.value].append(m.value)

        summary = {}
        for metric_type, values in by_type.items():
            if values:
                summary[metric_type] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        return summary

    def get_compression_stats(self) -> dict:
        """Get compression-specific statistics."""
        if not self.compression_history:
            return {}

        ratios = [c.compression_ratio for c in self.compression_history if c.compression_ratio > 0]
        reductions = [c.token_reduction_pct for c in self.compression_history]
        times = [c.compression_time_ms for c in self.compression_history]

        return {
            "total_compressions": len(self.compression_history),
            "avg_compression_ratio": sum(ratios) / len(ratios) if ratios else 0,
            "avg_token_reduction_pct": sum(reductions) / len(reductions) if reductions else 0,
            "avg_compression_time_ms": sum(times) / len(times) if times else 0,
        }

    def get_recall_stats(self) -> dict:
        """Get recall-specific statistics."""
        if not self.recall_history:
            return {}

        latencies = [r.latency_ms for r in self.recall_history]
        retrieved = [r.memories_retrieved for r in self.recall_history]
        tokens = [r.tokens_returned for r in self.recall_history]

        return {
            "total_recalls": len(self.recall_history),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "avg_memories_retrieved": sum(retrieved) / len(retrieved) if retrieved else 0,
            "avg_tokens_returned": sum(tokens) / len(tokens) if tokens else 0,
        }

    def get_context_stats(self) -> dict:
        """Get context building statistics."""
        if not self.context_history:
            return {}

        ratios = [c.compression_ratio for c in self.context_history if c.compression_ratio > 0]
        parts = [c.parts_count for c in self.context_history]
        selected = [c.selected_parts for c in self.context_history]
        times = [c.build_time_ms for c in self.context_history]

        return {
            "total_contexts_built": len(self.context_history),
            "avg_compression_ratio": sum(ratios) / len(ratios) if ratios else 0,
            "avg_parts_count": sum(parts) / len(parts) if parts else 0,
            "avg_selected_parts": sum(selected) / len(selected) if selected else 0,
            "avg_build_time_ms": sum(times) / len(times) if times else 0,
        }

    def export_all(self) -> dict:
        """Export all metrics for external analysis."""
        return {
            "raw_metrics": [m.to_dict() for m in self.metrics],
            "compression_history": [
                {
                    "original_tokens": c.original_tokens,
                    "compressed_tokens": c.compressed_tokens,
                    "compression_ratio": c.compression_ratio,
                    "compression_time_ms": c.compression_time_ms,
                    "method": c.method,
                    "timestamp": c.timestamp.isoformat(),
                    "conversation_id": c.conversation_id,
                }
                for c in self.compression_history
            ],
            "recall_history": [
                {
                    "query": r.query,
                    "recall_level": r.recall_level,
                    "memories_retrieved": r.memories_retrieved,
                    "tokens_returned": r.tokens_returned,
                    "latency_ms": r.latency_ms,
                    "relevance_scores": r.relevance_scores,
                    "timestamp": r.timestamp.isoformat(),
                    "conversation_id": r.conversation_id,
                    "topic_id": r.topic_id,
                }
                for r in self.recall_history
            ],
            "context_history": [
                {
                    "raw_tokens": c.raw_tokens,
                    "effective_tokens": c.effective_tokens,
                    "compression_ratio": c.compression_ratio,
                    "parts_count": c.parts_count,
                    "selected_parts": c.selected_parts,
                    "tier_distribution": c.tier_distribution,
                    "build_time_ms": c.build_time_ms,
                    "timestamp": c.timestamp.isoformat(),
                    "conversation_id": c.conversation_id,
                    "topic_id": c.topic_id,
                }
                for c in self.context_history
            ],
        }


# Global metrics collector instance
_global_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics_collector() -> None:
    """Reset global metrics collector (for testing)."""
    global _global_collector
    _global_collector = MetricsCollector()
