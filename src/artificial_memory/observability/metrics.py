from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from threading import Lock
from typing import Any


class MetricType(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricUnit(StrEnum):
    COUNT = "count"
    BYTES = "bytes"
    MILLISECONDS = "ms"
    SECONDS = "s"
    PERCENT = "percent"
    RATE = "rate"


@dataclass
class MetricDefinition:
    name: str
    metric_type: MetricType
    unit: MetricUnit
    description: str = ""
    labels: list[str] = field(default_factory=list)
    buckets: list[float] | None = None


@dataclass
class MetricSample:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MetricCollector:
    def __init__(self):
        self.metrics: dict[str, MetricDefinition] = {}
        self.samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.summaries: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = Lock()

        self._register_default_metrics()

    def _register_default_metrics(self):
        self.register_metric(MetricDefinition(
            name="memory_total",
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.COUNT,
            description="Total number of memories",
        ))

        self.register_metric(MetricDefinition(
            name="memory_by_type",
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.COUNT,
            description="Memories by type",
            labels=["type"],
        ))

        self.register_metric(MetricDefinition(
            name="memory_by_resolution",
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.COUNT,
            description="Memories by resolution level",
            labels=["resolution"],
        ))

        self.register_metric(MetricDefinition(
            name="memory_by_status",
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.COUNT,
            description="Memories by status",
            labels=["status"],
        ))

        self.register_metric(MetricDefinition(
            name="conversations_total",
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.COUNT,
            description="Total conversations",
        ))

        self.register_metric(MetricDefinition(
            name="messages_total",
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Total messages logged",
        ))

        self.register_metric(MetricDefinition(
            name="recall_requests_total",
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Total recall requests",
            labels=["level"],
        ))

        self.register_metric(MetricDefinition(
            name="recall_latency_ms",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.MILLISECONDS,
            description="Recall latency in milliseconds",
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
        ))

        self.register_metric(MetricDefinition(
            name="recall_memories_retrieved",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.COUNT,
            description="Number of memories retrieved per recall",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
        ))

        self.register_metric(MetricDefinition(
            name="context_tokens_used",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.COUNT,
            description="Tokens used in context building",
            buckets=[100, 500, 1000, 2000, 4000, 8000, 16000, 32000],
        ))

        self.register_metric(MetricDefinition(
            name="context_compression_ratio",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.RATE,
            description="Context compression ratio",
            buckets=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0],
        ))

        self.register_metric(MetricDefinition(
            name="compression_ratio",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.RATE,
            description="Memory compression ratio",
            buckets=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        ))

        self.register_metric(MetricDefinition(
            name="compression_latency_ms",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.MILLISECONDS,
            description="Compression latency",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
        ))

        self.register_metric(MetricDefinition(
            name="consolidation_cycles_total",
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Total consolidation cycles run",
        ))

        self.register_metric(MetricDefinition(
            name="consolidation_memories_processed",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.COUNT,
            description="Memories processed per consolidation cycle",
            buckets=[10, 50, 100, 500, 1000, 5000, 10000],
        ))

        self.register_metric(MetricDefinition(
            name="federation_exchanges_total",
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Total federation exchanges",
            labels=["status"],
        ))

        self.register_metric(MetricDefinition(
            name="federation_memories_exchanged",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.COUNT,
            description="Memories exchanged per federation request",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
        ))

        self.register_metric(MetricDefinition(
            name="trust_evaluations_total",
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Trust evaluations performed",
            labels=["result"],
        ))

        self.register_metric(MetricDefinition(
            name="policy_violations_total",
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Policy violations detected",
            labels=["domain", "severity"],
        ))

        self.register_metric(MetricDefinition(
            name="retention_actions_total",
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Retention actions executed",
            labels=["action"],
        ))

        self.register_metric(MetricDefinition(
            name="healing_actions_total",
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Healing actions executed",
            labels=["action_type", "result"],
        ))

        self.register_metric(MetricDefinition(
            name="active_tenants",
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.COUNT,
            description="Number of active tenants",
        ))

        self.register_metric(MetricDefinition(
            name="storage_usage_bytes",
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.BYTES,
            description="Storage usage in bytes",
        ))

        self.register_metric(MetricDefinition(
            name="api_request_duration_ms",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.MILLISECONDS,
            description="API request duration",
            labels=["endpoint", "method"],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
        ))

        self.register_metric(MetricDefinition(
            name="api_requests_total",
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Total API requests",
            labels=["endpoint", "method", "status"],
        ))

    def register_metric(self, definition: MetricDefinition) -> None:
        self.metrics[definition.name] = definition

    def increment_counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None):
        key = self._make_key(name, labels)
        with self._lock:
            self.counters[key] += value
            self._record_sample(name, value, labels)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None):
        key = self._make_key(name, labels)
        with self._lock:
            self.gauges[key] = value
            self._record_sample(name, value, labels)

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None):
        key = self._make_key(name, labels)
        with self._lock:
            self.histograms[key].append(value)
            if len(self.histograms[key]) > 10000:
                self.histograms[key] = self.histograms[key][-10000:]
            self._record_sample(name, value, labels)

    def observe_summary(self, name: str, value: float, labels: dict[str, str] | None = None):
        key = self._make_key(name, labels)
        with self._lock:
            self.summaries[key].append(value)
            self._record_sample(name, value, labels)

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _record_sample(self, name: str, value: float, labels: dict[str, str] | None = None):
        sample = MetricSample(name=name, value=value, labels=labels or {})
        self.samples[name].append(sample)

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = self._make_key(name, labels)
        return self.counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        key = self._make_key(name, labels)
        return self.gauges.get(key)

    def get_histogram_stats(self, name: str, labels: dict[str, str] | None = None) -> dict[str, float]:
        key = self._make_key(name, labels)
        values = self.histograms.get(key, [])

        if not values:
            return {}

        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p50": self._percentile(values, 50),
            "p90": self._percentile(values, 90),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def get_summary_stats(self, name: str, labels: dict[str, str] | None = None) -> dict[str, float]:
        key = self._make_key(name, labels)
        values = self.summaries.get(key, [])

        if not values:
            return {}

        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_all_metrics(self) -> dict[str, Any]:
        result = {}

        for key, value in self.counters.items():
            result[f"counter_{key}"] = value

        for key, value in self.gauges.items():
            result[f"gauge_{key}"] = value

        for key, values in self.histograms.items():
            if values:
                stats = self.get_histogram_stats(key.replace("histogram_", "").replace("{", "").replace("}", ""))
                for k, v in stats.items():
                    result[f"histogram_{key}_{k}"] = v

        for key, values in self.summaries.items():
            if values:
                stats = self.get_summary_stats(key.replace("summary_", "").replace("{", "").replace("}", ""))
                for k, v in stats.items():
                    result[f"summary_{key}_{k}"] = v

        return result

    def export_prometheus(self) -> str:
        lines = []

        for key, value in self.counters.items():
            metric_name = key.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {value}")

        for key, value in self.gauges.items():
            metric_name = key.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")

        for key, values in self.histograms.items():
            if not values:
                continue
            metric_name = key.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
            lines.append(f"# TYPE {metric_name} histogram")

            stats = self.get_histogram_stats(key.replace("histogram_", "").replace("{", "").replace("}", ""))
            for stat_name, value in stats.items():
                lines.append(f"{metric_name}_{stat_name} {value}")

            for bucket in [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, float('inf')]:
                count = sum(1 for v in self.histograms[key] if v <= bucket)
                le = str(bucket) if bucket != float('inf') else "+Inf"
                lines.append(f'{metric_name}_bucket{{le="{le}"}} {count}')

        return "\n".join(lines) + "\n"

    def reset(self):
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.summaries.clear()
            self.samples.clear()


class MetricsCollector:
    def __init__(self, collector=None):
        self.collector = collector or MetricCollector()
        self._start_time = time.time()

    def record_memory_created(self, memory_type: str, resolution: str, tokens: int):
        self.collector.increment_counter("memories_created_total", labels={"type": memory_type, "resolution": resolution})
        self.collector.increment_counter("memory_tokens_total", tokens)
        self.collector.set_gauge("memory_by_type", 1, labels={"type": memory_type})
        self.collector.set_gauge("memory_by_resolution", 1, labels={"resolution": resolution})

    def record_memory_accessed(self, memory_type: str, resolution: str):
        self.collector.increment_counter("memory_accessed_total", labels={"type": memory_type, "resolution": resolution})

    def record_recall(self, level: int, memories_found: int, tokens: int, latency_ms: float):
        self.collector.increment_counter("recall_requests_total", labels={"level": str(level)})
        self.collector.observe_histogram("recall_memories_retrieved", memories_found)
        self.collector.increment_counter("recall_tokens_total", tokens)
        self.collector.observe_histogram("recall_latency_ms", latency_ms)

    def record_context_build(self, tokens_used: int, compression_ratio: float, parts_count: int):
        self.collector.observe_histogram("context_tokens_used", tokens_used)
        self.collector.observe_histogram("context_compression_ratio", compression_ratio)
        self.collector.observe_histogram("context_parts_count", parts_count)

    def record_compression(self, original_tokens: int, compressed_tokens: int, ratio: float, latency_ms: float, method: str):
        self.collector.increment_counter("compressions_total", labels={"method": method})
        self.collector.observe_histogram("compression_ratio", ratio)
        self.collector.observe_histogram("compression_latency_ms", latency_ms, labels={"method": method})
        self.collector.increment_counter("compression_tokens_saved", original_tokens - compressed_tokens)

    def record_consolidation(self, memories_processed: int, duration_ms: float):
        self.collector.increment_counter("consolidation_cycles_total")
        self.collector.observe_histogram("consolidation_memories_processed", memories_processed)
        self.collector.observe_histogram("consolidation_duration_ms", duration_ms)

    def record_federation_exchange(self, status: str, memories_count: int):
        self.collector.increment_counter("federation_exchanges_total", labels={"status": status})
        self.collector.observe_histogram("federation_memories_exchanged", memories_count)

    def record_trust_evaluation(self, result: str):
        self.collector.increment_counter("trust_evaluations_total", labels={"result": result})

    def record_policy_violation(self, domain: str, severity: str):
        self.collector.increment_counter("policy_violations_total", labels={"domain": domain, "severity": severity})

    def record_retention_action(self, action: str):
        self.collector.increment_counter("retention_actions_total", labels={"action": action})

    def record_healing_action(self, action_type: str, result: str):
        self.collector.increment_counter("healing_actions_total", labels={"action_type": action_type, "result": result})

    def set_active_tenants(self, count: int):
        self.collector.set_gauge("active_tenants", count)

    def set_storage_usage(self, bytes_used: int):
        self.collector.set_gauge("storage_usage_bytes", bytes_used)

    def record_api_request(self, endpoint: str, method: str, status: int, duration_ms: float):
        self.collector.increment_counter("api_requests_total", labels={"endpoint": endpoint, "method": method, "status": str(status)})
        self.collector.observe_histogram("api_request_duration_ms", duration_ms, labels={"endpoint": endpoint, "method": method})

    def get_metrics_summary(self) -> dict[str, Any]:
        return self.collector.get_all_metrics()

    def export_prometheus(self) -> str:
        return self.collector.export_prometheus()


# Global collector instance
_global_collector: MetricCollector | None = None


def get_metrics_collector() -> MetricCollector:
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricCollector()
    return _global_collector


def create_metrics_collector() -> MetricCollector:
    return MetricCollector()
