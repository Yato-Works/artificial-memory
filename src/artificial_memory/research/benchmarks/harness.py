from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MetricType(StrEnum):
    RECALL_ACCURACY = "recall_accuracy"
    SEMANTIC_PRESERVATION = "semantic_preservation"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    TOKEN_CONSUMPTION = "token_consumption"
    LATENCY = "latency"
    MEMORY_STORAGE_SIZE = "memory_storage_size"
    INDEX_COST = "index_cost"
    EXPANSION_FREQUENCY = "expansion_frequency"
    CONTEXT_RELEVANCE = "context_relevance"
    CONTEXT_COMPLETENESS = "context_completeness"
    DOWNSTREAM_ANSWER_QUALITY = "downstream_answer_quality"
    UNNECESSARY_RETRIEVAL_RATE = "unnecessary_retrieval_rate"
    STALE_MEMORY_RESISTANCE = "stale_memory_resistance"
    CONTRADICTION_RESISTANCE = "contradiction_resistance"
    FALSE_MEMORY_RESISTANCE = "false_memory_resistance"
    COMPRESSION_FAILURE_RECOVERY = "compression_failure_recovery"
    PROVENANCE_COMPLETENESS = "provenance_completeness"
    RECALL_EXPLANATION_CORRECTNESS = "recall_explanation_correctness"
    DECISION_TRACEABILITY = "decision_traceability"


@dataclass
class BenchmarkMetric:
    name: MetricType
    value: float
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: datetime | None = None
    metrics: list[BenchmarkMetric] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    seed: int | None = None
    git_commit: str | None = None
    config_hash: str | None = None


@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    experiment_type: str = "benchmark"
    runner_class: str = "BenchmarkRunner"
    config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    timeout_seconds: int = 3600
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    expected_metrics: list[MetricType] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    name: str
    description: str = ""
    experiments: list[ExperimentConfig] = field(default_factory=list)
    setup_fn: Callable | None = None
    teardown_fn: Callable | None = None
    tags: list[str] = field(default_factory=list)


class BenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: list[ExperimentResult] = []
        self._rng = random.Random(config.seed) if config.seed else random.Random()

    @abstractmethod
    def setup(self) -> None:
        """Setup before running experiments."""
        pass

    @abstractmethod
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup after experiments."""
        pass

    def run_suite(self, suite: BenchmarkSuite) -> list[ExperimentResult]:
        """Run a full benchmark suite."""
        if suite.setup_fn:
            suite.setup_fn()

        try:
            for exp_config in suite.experiments:
                result = self.run_experiment(exp_config)
                self.results.append(result)

            if suite.teardown_fn:
                suite.teardown_fn()
        except Exception:
            if suite.teardown_fn:
                suite.teardown_fn()
            raise

        return self.results


class BenchmarkHarness:
    """Main harness for running benchmarks with full reproducibility."""

    def __init__(
        self,
        output_dir: str = "benchmark_results",
        default_seed: int = 42,
        max_parallel: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_seed = default_seed
        self.max_parallel = max_parallel
        self.suites: dict[str, BenchmarkSuite] = {}
        self.runners: dict[str, type[BenchmarkRunner]] = {}
        self.global_results: list[ExperimentResult] = []
        self._run_id = hashlib.md5(
            f"{datetime.now().isoformat()}{random.random()}".encode()
        ).hexdigest()[:8]

    def register_runner(self, name: str, runner_class: type[BenchmarkRunner]):
        """Register a benchmark runner."""
        self.runners[name] = runner_class

    def register_suite(self, suite: BenchmarkSuite):
        """Register a benchmark suite."""
        self.suites[suite.name] = suite

    def create_experiment(
        self,
        name: str,
        runner_name: str,
        config: dict[str, Any],
        seed: int | None = None,
        **kwargs
    ) -> ExperimentConfig:
        """Create an experiment configuration."""
        runner_class = self.runners.get(runner_name)
        if not runner_class:
            raise ValueError(f"Runner {runner_name} not registered")

        return ExperimentConfig(
            name=name,
            experiment_type="benchmark",
            runner_class=runner_name,
            config=config,
            seed=seed or self.default_seed,
            **kwargs
        )

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with full reproducibility."""
        runner_class = self.runners[config.runner_class]
        runner = runner_class(config)

        result = ExperimentResult(
            experiment_id=hashlib.md5(
                f"{config.name}{config.seed}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12],
            experiment_name=config.name,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now(),
            seed=config.seed,
            git_commit=self._get_git_commit(),
            config_hash=self._hash_config(config.config),
        )

        try:
            runner.setup()
            result = runner.run_experiment(config)
            result.status = ExperimentStatus.COMPLETED
        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error = str(e)
        finally:
            runner.teardown()
            result.end_time = datetime.now()

        self.global_results.append(result)
        self._save_result(result)
        return result

    def run_suite(self, suite_name: str) -> list[ExperimentResult]:
        """Run a full benchmark suite."""
        suite = self.suites.get(suite_name)
        if not suite:
            raise ValueError(f"Suite {suite_name} not found")

        runner_class = self.runners.get(suite.experiments[0].runner_class)
        if not runner_class:
            raise ValueError(f"No runner for suite {suite_name}")

        runner_class(suite.experiments[0])

        if suite.setup_fn:
            suite.setup_fn()

        results = []
        try:
            for exp_config in suite.experiments:
                result = self.run_experiment(exp_config)
                results.append(result)

            if suite.teardown_fn:
                suite.teardown_fn()
        except Exception:
            if suite.teardown_fn:
                suite.teardown_fn()
            raise

        return results

    def run_all(self) -> list[ExperimentResult]:
        """Run all registered suites."""
        all_results = []
        for suite_name in self.suites:
            results = self.run_suite(suite_name)
            all_results.extend(results)
        return all_results

    def _get_git_commit(self) -> str | None:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _hash_config(self, config: dict[str, Any]) -> str:
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

    def _save_result(self, result: ExperimentResult):
        """Save result to disk."""
        filename = f"{result.experiment_id}_{result.experiment_name}.json"
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self._serialize_result(result), f, indent=2)

    def _serialize_result(self, result: ExperimentResult) -> dict[str, Any]:
        return {
            "experiment_id": result.experiment_id,
            "experiment_name": result.experiment_name,
            "status": result.status.value,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "metrics": [
                {
                    "name": m.name.value,
                    "value": m.value,
                    "unit": m.unit,
                    "metadata": m.metadata,
                }
                for m in result.metrics
            ],
            "metadata": result.metadata,
            "error": result.error,
            "seed": result.seed,
            "git_commit": result.git_commit,
            "config_hash": result.config_hash,
        }

    def export_summary(self, filepath: str | None = None) -> str:
        """Export a summary of all results."""
        if filepath is None:
            filepath = self.output_dir / f"summary_{self._run_id}.json"

        summary = {
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.global_results),
            "completed": sum(1 for r in self.global_results if r.status == ExperimentStatus.COMPLETED),
            "failed": sum(1 for r in self.global_results if r.status == ExperimentStatus.FAILED),
            "experiments": [self._serialize_result(r) for r in self.global_results],
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        return str(filepath)

    def load_results(self, filepath: str) -> list[ExperimentResult]:
        """Load results from a file."""
        with open(filepath) as f:
            data = json.load(f)

        results = []
        for item in data.get("experiments", []):
            result = ExperimentResult(
                experiment_id=item["experiment_id"],
                experiment_name=item["experiment_name"],
                status=ExperimentStatus(item["status"]),
                start_time=datetime.fromisoformat(item["start_time"]),
                end_time=datetime.fromisoformat(item["end_time"]) if item["end_time"] else None,
                metrics=[
                    BenchmarkMetric(
                        name=MetricType(m["name"]),
                        value=m["value"],
                        unit=m["unit"],
                        metadata=m["metadata"],
                    )
                    for m in item["metrics"]
                ],
                metadata=item["metadata"],
                error=item["error"],
                seed=item["seed"],
                git_commit=item["git_commit"],
                config_hash=item["config_hash"],
            )
            results.append(result)

        self.global_results = results
        return results

    def compare_runs(self, run_a: str, run_b: str) -> dict[str, Any]:
        """Compare two benchmark runs."""
        results_a = self.load_results(run_a)
        results_b = self.load_results(run_b)

        # Group by experiment name
        map_a = {r.experiment_name: r for r in results_a}
        map_b = {r.experiment_name: r for r in results_b}

        all_names = set(map_a.keys()) | set(map_b.keys())

        comparison = {
            "run_a": run_a,
            "run_b": run_b,
            "comparisons": [],
        }

        for name in all_names:
            if name in map_a and name in map_b:
                ra, rb = map_a[name], map_b[name]
                comparison = {
                    "experiment": name,
                    "metrics": [],
                }
                metrics_a = {m.name: m.value for m in ra.metrics}
                metrics_b = {m.name: m.value for m in rb.metrics}

                all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
                for metric in all_metrics:
                    if metric in metrics_a and metric in metrics_b:
                        comparison["metrics"].append({
                            "metric": metric.value,
                            "run_a": metrics_a[metric],
                            "run_b": metrics_b[metric],
                            "delta": metrics_b[metric] - metrics_a[metric],
                            "pct_change": ((metrics_b[metric] - metrics_a[metric]) / metrics_a[metric] * 100) if metrics_a[metric] != 0 else float('inf'),
                        })

                comparison["comparisons"].append(comparison)

        return comparison


def create_benchmark_harness(
    output_dir: str = "benchmark_results",
    default_seed: int = 42,
) -> BenchmarkHarness:
    return BenchmarkHarness(output_dir, default_seed)


