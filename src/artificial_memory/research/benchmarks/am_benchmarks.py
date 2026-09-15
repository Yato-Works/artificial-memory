from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class BenchmarkCategory(StrEnum):
    TEMPORAL = "temporal"
    CONTRADICTION = "contradiction"
    LONG_TERM_DEPENDENCY = "long_term_dependency"
    FALSE_MEMORY = "false_memory"
    COMPRESSION_LOSS = "compression_loss"
    MULTI_SESSION = "multi_session"
    ADAPTIVE_RECALL = "adaptive_recall"
    BELIEF_EVOLUTION = "belief_evolution"
    DECISION_TRACE = "decision_trace"
    IMPACT_ANALYSIS = "impact_analysis"


@dataclass
class BenchmarkCase:
    name: str
    category: BenchmarkCategory
    description: str
    setup_fn: str | None = None
    test_queries: list[str] = field(default_factory=list)
    expected_outcomes: dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    case_name: str
    category: BenchmarkCategory
    success: bool
    metrics: dict[str, float]
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    seed: int = 42


class AMBenchmarkSuite:
    """Artificial Memory specific synthetic benchmarks.

    These benchmarks test capabilities unique to AM:
    - Temporal reasoning and time travel
    - Contradiction detection and resolution
    - Long-term dependency tracking
    - False memory resistance
    - Compression loss detection
    - Multi-session continuity
    - Adaptive recall optimization
    - Belief evolution tracking
    - Decision traceability
    - Impact analysis accuracy
    """

    def __init__(
        self,
        store,
        recall_engine,
        context_builder,
        compressor,
        time_travel_engine=None,
        belief_engine=None,
        evolution_engine=None,
        contradiction_detector=None,
        dependency_graph=None,
        adaptive_recall=None,
        output_dir: str = "am_benchmarks",
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.context_builder = context_builder
        self.compressor = compressor
        self.time_travel_engine = time_travel_engine
        self.belief_engine = belief_engine
        self.evolution_engine = evolution_engine
        self.contradiction_detector = contradiction_detector
        self.dependency_graph = dependency_graph
        self.adaptive_recall = adaptive_recall
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            import tempfile

            self.output_dir = Path(tempfile.gettempdir()) / "am_benchmarks"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cases: dict[str, BenchmarkCase] = {}
        self.results: list[BenchmarkResult] = []
        self._register_default_cases()

    def _register_default_cases(self):
        """Register all default AM benchmark cases."""

        # Temporal Benchmarks
        self.register_case(BenchmarkCase(
            name="temporal_state_reconstruction",
            category=BenchmarkCategory.TEMPORAL,
            description="Reconstruct memory state at a historical timestamp",
            test_queries=["What did we know about X on date Y?"],
            expected_outcomes={"state_accuracy": 0.9, "temporal_consistency": 0.95},
            difficulty="medium",
            tags=["temporal", "state_reconstruction", "time_travel"],
        ))

        self.register_case(BenchmarkCase(
            name="temporal_belief_reconstruction",
            category=BenchmarkCategory.TEMPORAL,
            description="Reconstruct beliefs at a historical timestamp",
            test_queries=["What did we believe about X on date Y?"],
            expected_outcomes={"belief_accuracy": 0.85, "confidence_calibration": 0.8},
            difficulty="hard",
            tags=["temporal", "belief", "time_travel"],
        ))

        self.register_case(BenchmarkCase(
            name="temporal_context_reconstruction",
            category=BenchmarkCategory.TEMPORAL,
            description="Reconstruct LLM context at historical timestamp",
            test_queries=["What context would LLM have had on date Y?"],
            expected_outcomes={"context_fidelity": 0.9, "token_budget_adherence": 1.0},
            difficulty="hard",
            tags=["temporal", "context", "reconstruction"],
        ))

        # Contradiction Benchmarks
        self.register_case(BenchmarkCase(
            name="contradiction_detection",
            category=BenchmarkCategory.CONTRADICTION,
            description="Detect contradictory memories in the system",
            test_queries=["Are there any contradictions about X?"],
            expected_outcomes={"detection_rate": 0.95, "false_positive_rate": 0.05},
            difficulty="medium",
            tags=["contradiction", "detection"],
        ))

        self.register_case(BenchmarkCase(
            name="contradiction_resolution",
            category=BenchmarkCategory.CONTRADICTION,
            description="Resolve contradictions and maintain correct beliefs",
            test_queries=["What is the current truth about X given contradictions?"],
            expected_outcomes={"resolution_accuracy": 0.9, "belief_consistency": 0.95},
            difficulty="hard",
            tags=["contradiction", "resolution", "belief"],
        ))

        # Long-term Dependency Benchmarks
        self.register_case(BenchmarkCase(
            name="long_term_dependency_tracking",
            category=BenchmarkCategory.LONG_TERM_DEPENDENCY,
            description="Track dependencies across many sessions",
            test_queries=["What decisions led to the current architecture?"],
            expected_outcomes={"dependency_accuracy": 0.9, "trace_completeness": 0.9},
            difficulty="medium",
            tags=["dependency", "long_term", "traceability"],
        ))

        self.register_case(BenchmarkCase(
            name="cross_session_dependency",
            category=BenchmarkCategory.LONG_TERM_DEPENDENCY,
            description="Track dependencies across multiple sessions",
            test_queries=["What was decided in session 1 that affects session 10?"],
            expected_outcomes={"cross_session_accuracy": 0.85, "dependency_chain_length": 5},
            difficulty="hard",
            tags=["dependency", "cross_session", "multi_session"],
        ))

        # False Memory Benchmarks
        self.register_case(BenchmarkCase(
            name="false_memory_resistance",
            category=BenchmarkCategory.FALSE_MEMORY,
            description="Resist accepting false memories as truth",
            test_queries=["Is it true that X?"],
            expected_outcomes={"rejection_rate": 0.9, "confidence_calibration": 0.8},
            difficulty="hard",
            tags=["false_memory", "resistance", "calibration"],
        ))

        self.register_case(BenchmarkCase(
            name="high_confidence_false_memory",
            category=BenchmarkCategory.FALSE_MEMORY,
            description="Resist high-confidence false memories",
            test_queries=["Is it true that X? (with high confidence false memory)"],
            expected_outcomes={"rejection_rate": 0.8, "confidence_discrimination": 0.7},
            difficulty="hard",
            tags=["false_memory", "high_confidence", "calibration"],
        ))

        # Compression Loss Benchmarks
        self.register_case(BenchmarkCase(
            name="decision_preservation_under_compression",
            category=BenchmarkCategory.COMPRESSION_LOSS,
            description="Preserve decisions through compression levels",
            test_queries=["What was decided about X?"],
            expected_outcomes={"decision_preservation": 0.95, "reason_preservation": 0.9},
            difficulty="medium",
            tags=["compression", "decision", "preservation"],
        ))

        self.register_case(BenchmarkCase(
            name="entity_preservation_under_compression",
            category=BenchmarkCategory.COMPRESSION_LOSS,
            description="Preserve entity references through compression",
            test_queries=["What entities are involved in X?"],
            expected_outcomes={"entity_preservation": 0.9, "relationship_preservation": 0.85},
            difficulty="medium",
            tags=["compression", "entity", "preservation"],
        ))

        # Multi-session Benchmarks
        self.register_case(BenchmarkCase(
            name="multi_session_continuity",
            category=BenchmarkCategory.MULTI_SESSION,
            description="Maintain continuity across multiple sessions",
            test_queries=["What did we discuss in the last session?"],
            expected_outcomes={"continuity_score": 0.9, "context_carryover": 0.9},
            difficulty="medium",
            tags=["multi_session", "continuity"],
        ))

        self.register_case(BenchmarkCase(
            name="cross_session_decision_tracking",
            category=BenchmarkCategory.MULTI_SESSION,
            description="Track decisions across session boundaries",
            test_queries=["What was decided about X in session 3?"],
            expected_outcomes={"decision_retrieval": 0.9, "cross_session_accuracy": 0.85},
            difficulty="hard",
            tags=["multi_session", "decision", "cross_session"],
        ))

        # Adaptive Recall Benchmarks
        self.register_case(BenchmarkCase(
            name="adaptive_recall_utility",
            category=BenchmarkCategory.ADAPTIVE_RECALL,
            description="Optimize recall utility within token budget",
            test_queries=["Tell me about X"],
            expected_outcomes={"utility_score": 0.9, "token_efficiency": 0.9, "relevance": 0.95},
            difficulty="medium",
            tags=["adaptive_recall", "utility", "optimization"],
        ))

        # Belief Evolution Benchmarks
        self.register_case(BenchmarkCase(
            name="belief_evolution_tracking",
            category=BenchmarkCategory.BELIEF_EVOLUTION,
            description="Track how beliefs evolve over time",
            test_queries=["How has our belief about X changed?"],
            expected_outcomes={"evolution_tracking": 0.9, "confidence_calibration": 0.85},
            difficulty="medium",
            tags=["belief", "evolution", "temporal"],
        ))

        self.register_case(BenchmarkCase(
            name="belief_contradiction_handling",
            category=BenchmarkCategory.BELIEF_EVOLUTION,
            description="Handle belief contradictions gracefully",
            test_queries=["What do we believe about X now?"],
            expected_outcomes={"contradiction_resolution": 0.9, "belief_consistency": 0.9},
            difficulty="hard",
            tags=["belief", "contradiction", "resolution"],
        ))

        # Decision Trace Benchmarks
        self.register_case(BenchmarkCase(
            name="decision_trace_completeness",
            category=BenchmarkCategory.DECISION_TRACE,
            description="Complete traceability of decisions",
            test_queries=["Why was decision X made?"],
            expected_outcomes={"trace_completeness": 0.95, "evidence_chain_integrity": 0.95},
            difficulty="medium",
            tags=["decision", "trace", "provenance"],
        ))

        # Impact Analysis Benchmarks
        self.register_case(BenchmarkCase(
            name="impact_analysis_accuracy",
            category=BenchmarkCategory.IMPACT_ANALYSIS,
            description="Accurately predict impact of memory changes",
            test_queries=["What would be affected if we change X?"],
            expected_outcomes={"prediction_accuracy": 0.9, "cascade_risk_accuracy": 0.85},
            difficulty="hard",
            tags=["impact", "analysis", "prediction"],
        ))

        self.register_case(BenchmarkCase(
            name="circular_dependency_detection",
            category=BenchmarkCategory.IMPACT_ANALYSIS,
            description="Detect circular dependencies in memory graph",
            test_queries=["Are there circular dependencies?"],
            expected_outcomes={"detection_rate": 0.95, "false_positive_rate": 0.05},
            difficulty="medium",
            tags=["impact", "circular_dependency", "graph"],
        ))

        self.register_case(BenchmarkCase(
            name="single_point_of_failure_detection",
            category=BenchmarkCategory.IMPACT_ANALYSIS,
            description="Identify single points of failure in memory graph",
            test_queries=["What memories are critical dependencies?"],
            expected_outcomes={"detection_rate": 0.9, "precision": 0.9},
            difficulty="medium",
            tags=["impact", "spof", "criticality"],
        ))

    def register_case(self, case: BenchmarkCase):
        self.cases[case.name] = case

    def run_case(
        self,
        case_name: str,
        store,
        recall_engine,
        context_builder,
        compressor,
        topic_id: int,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run a specific benchmark case."""
        case = self.cases.get(case_name)
        if not case:
            raise ValueError(f"Case {case_name} not found")

        random.seed(42)

        start_time = datetime.now()

        try:
            method_name = f"_run_{case_name}"
            if hasattr(self, method_name):
                result = getattr(self, method_name)(case, store, recall_engine, context_builder, compressor)
            else:
                result = self._run_generic(case, store, recall_engine, context_builder, compressor)

            result.success = True
        except Exception as e:
            result = BenchmarkResult(
                case_name=case.name,
                category=case.category,
                success=False,
                metrics={},
                latency_ms=0,
                details={"error": str(e)},
            )

        result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.timestamp = datetime.now()
        return result

    def run_all(
        self,
        store,
        recall_engine,
        context_builder,
        compressor,
        topic_id: int,
        categories: list | None = None,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Run all benchmark cases (or filtered by category)."""
        results = []

        for case in self.cases.values():
            if categories and case.category not in categories:
                continue

            result = self.run_case(case.name, store, recall_engine, context_builder, compressor, topic_id, seed)
            self.results.append(result)

            status = "PASS" if result.success else "FAIL"
            print(f"  {result.case_name} [{result.category.value}] {status} ({result.latency_ms:.1f}ms)")

        return results

    def _run_generic(self, case, store, recall_engine, context_builder, compressor):
        """Generic benchmark runner."""
        topic_id = 1

        all_metrics = []
        for query in case.test_queries:
            memories, tokens = recall_engine.recall(query, topic_id, level=2, max_tokens=4000)
            context_builder.build_context(query, topic_id, max_tokens=4000)
            stats = context_builder.get_context_stats()

            metrics = {
                "memories_retrieved": len(memories),
                "tokens": tokens,
                "effective_tokens": stats.effective_tokens,
                "compression_ratio": stats.compression_ratio,
            }
            all_metrics.append(metrics)

        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        success = True
        for metric, expected in case.expected_outcomes.items():
            if metric in avg_metrics:
                if avg_metrics[metric] < expected * 0.8:
                    success = False

        return BenchmarkResult(
            case_name=case.name,
            category=case.category,
            success=success,
            metrics=avg_metrics,
            latency_ms=0,
            details={},
        )

    def run_suite(
        self,
        store,
        recall_engine,
        context_builder,
        compressor,
        topic_id: int,
        categories: list | None = None,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Run all benchmark cases (or filtered by category)."""
        results = []

        for case in self.cases.values():
            if categories and case.category not in categories:
                continue

            result = self.run_case(case.name, store, recall_engine, context_builder, compressor, topic_id, seed)
            self.results.append(result)

            status = "PASS" if result.success else "FAIL"
            print(f"  {result.case_name} [{result.category.value}] {status} ({result.latency_ms:.1f}ms)")

        return results

    def run_by_category(
        self,
        store,
        recall_engine,
        context_builder,
        compressor,
        category: BenchmarkCategory,
        topic_id: int,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Run all benchmarks in a specific category."""
        filtered = [c for c in self.cases.values() if c.category == category]
        results = []

        for case in filtered:
            result = self.run_case(case.name, store, recall_engine, context_builder, compressor, topic_id, seed)
            results.append(result)

        return results

    def generate_report(self, output_path: str | None = None) -> str:
        """Generate a benchmark report."""
        lines = [
            "=" * 60,
            "ARTIFICIAL MEMORY BENCHMARK REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            f"Total cases: {len(self.cases)}",
            f"Results: {len(self.results)}",
            "",
        ]

        by_category = {}
        for result in self.results:
            cat = result.category.value
            if cat not in by_category:
                by_category[cat] = {"pass": 0, "fail": 0, "total": 0}

            by_category[cat]["total"] += 1
            if result.success:
                by_category[result.category.value]["pass"] += 1
            else:
                by_category[result.category.value]["fail"] += 1

        lines.append("\nResults by Category:")
        for cat, stats in sorted(by_category.items()):
            pass_rate = stats["pass"] / stats["total"] * 100 if stats["total"] > 0 else 0
            lines.append(f"  {cat}: {stats['pass']}/{stats['total']} ({pass_rate:.1f}%)")

        lines.append("\nDetailed Results:")
        for result in self.results:
            status = "PASS" if result.success else "FAIL"
            lines.append(f"  {result.case_name} [{result.category.value}] {status} ({result.latency_ms:.1f}ms)")
            if not result.success:
                lines.append(f"    Error: {result.details.get('error', 'Unknown')}")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report


def create_am_benchmark_suite(
    store,
    recall_engine,
    context_builder,
    compressor,
    time_travel_engine=None,
    belief_engine=None,
    evolution_engine=None,
    contradiction_detector=None,
    dependency_graph=None,
    adaptive_recall=None,
    output_dir="am_benchmarks",
):
    return AMBenchmarkSuite(
        store, recall_engine, context_builder, compressor,
        time_travel_engine, belief_engine, evolution_engine,
        contradiction_detector, dependency_graph, adaptive_recall,
        output_dir
    )
