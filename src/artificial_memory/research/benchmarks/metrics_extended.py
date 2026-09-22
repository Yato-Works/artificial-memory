"""Extended Evaluation Metrics for AI Memory Systems (Phase 3).

Calculates comprehensive research-grade metrics:
- Accuracy: Overall ground-truth match rate
- False Positive Rate (FPR): Hallucination / over-confidence rate (critical for AM evaluation)
- Abstention Precision: Accuracy when model claims insufficient information
- Recall: Fraction of required ground-truth evidence successfully surfaced
- Token Efficiency: Context tokens per correct answer
- Latency Profiling: p50, p95, p99 wall-clock latency
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.research.benchmarks.scorer import QuestionScore


@dataclass(frozen=True)
class ExtendedMetrics:
    """Comprehensive performance and efficiency metrics for a player."""
    player_name: str
    total_questions: int

    # Accuracy & Soundness
    accuracy: float
    pass_count: int
    partial_count: int
    fail_count: int

    # Hallucination & Abstention (AM key advantage)
    false_positive_count: int
    false_positive_rate: float
    abstention_accuracy: float
    forbidden_violation_rate: float

    # Ground Truth Coverage
    mean_gt_coverage: float

    # Resource & Latency
    mean_context_tokens: float
    total_write_llm_calls: int
    total_read_llm_calls: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


def compute_extended_metrics(
    player_name: str,
    scores: Sequence[QuestionScore],
    latencies_ms: Sequence[float] | None = None,
    context_tokens: Sequence[int] | None = None,
    write_calls: int = 0,
    read_calls: int = 0,
) -> ExtendedMetrics:
    """Compute extended research metrics from score records."""
    total = len(scores)
    if total == 0:
        return ExtendedMetrics(
            player_name=player_name,
            total_questions=0,
            accuracy=0.0,
            pass_count=0,
            partial_count=0,
            fail_count=0,
            false_positive_count=0,
            false_positive_rate=0.0,
            abstention_accuracy=0.0,
            forbidden_violation_rate=0.0,
            mean_gt_coverage=0.0,
            mean_context_tokens=0.0,
            total_write_llm_calls=write_calls,
            total_read_llm_calls=read_calls,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
        )

    passes = sum(1 for s in scores if s.outcome == "pass")
    partials = sum(1 for s in scores if s.outcome == "partial")
    fails = sum(1 for s in scores if s.outcome in ["fail", "false_positive"])
    fps = sum(1 for s in scores if s.outcome == "false_positive" or s.forbidden_violation)

    # Abstention questions
    abst_scores = [s for s in scores if s.abstention_detected is not None]
    abst_acc = (
        sum(1 for s in abst_scores if s.outcome == "pass") / len(abst_scores)
        if abst_scores else 1.0
    )

    # Latency percentiles
    lats = sorted(latencies_ms) if latencies_ms else [0.0]
    p50 = float(statistics.median(lats))
    p95 = float(lats[int(len(lats) * 0.95)]) if len(lats) > 1 else lats[0]
    p99 = float(lats[int(len(lats) * 0.99)]) if len(lats) > 1 else lats[0]

    mean_ctx = float(statistics.mean(context_tokens)) if context_tokens else 0.0
    mean_cov = float(statistics.mean(s.ground_truth_coverage for s in scores))

    return ExtendedMetrics(
        player_name=player_name,
        total_questions=total,
        accuracy=sum(s.score for s in scores) / total,
        pass_count=passes,
        partial_count=partials,
        fail_count=fails,
        false_positive_count=fps,
        false_positive_rate=fps / total,
        abstention_accuracy=abst_acc,
        forbidden_violation_rate=sum(1 for s in scores if s.forbidden_violation) / total,
        mean_gt_coverage=mean_cov,
        mean_context_tokens=mean_ctx,
        total_write_llm_calls=write_calls,
        total_read_llm_calls=read_calls,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
    )
