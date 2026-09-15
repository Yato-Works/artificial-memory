from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class ExperimentType(StrEnum):
    BENCHMARK = "benchmark"
    ABLATION = "ablation"
    REDTEAM = "redteam"
    REGRESSION = "regression"
    INTEGRATION = "integration"


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExperimentManifest:
    """Declarative specification of an experiment."""
    name: str
    experiment_type: ExperimentType
    description: str = ""

    # Runner specification
    runner: str = "default"  # Runner class name

    # Configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Reproducibility
    seed: int = 42
    git_commit: str | None = None
    dependencies: list[str] = field(default_factory=list)
    required_resources: dict[str, Any] = field(default_factory=dict)

    # Execution
    timeout_seconds: int = 3600
    retry_count: int = 0
    retry_delay_seconds: int = 60

    # Validation
    expected_metrics: dict[str, float] = field(default_factory=dict)
    pass_threshold: float = 0.8

    # Metadata
    tags: list[str] = field(default_factory=list)
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def compute_hash(self) -> str:
        """Compute deterministic hash of the manifest."""
        data = {
            "name": self.name,
            "type": self.experiment_type.value,
            "config": self.config,
            "seed": self.seed,
            "version": self.version,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class ExperimentManifestV2:
    """Version 2 manifest with enhanced features."""
    name: str
    experiment_type: ExperimentType
    description: str = ""

    # Pipeline
    steps: list[dict[str, Any]] = field(default_factory=list)

    # Parameters
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_sweep: dict[str, list[Any]] | None = None

    # Environment
    environment: dict[str, str] = field(default_factory=dict)
    container_image: str | None = None

    # Resources
    resources: dict[str, Any] = field(default_factory=dict)

    # Outputs
    outputs: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    # Validation
    success_criteria: dict[str, Any] = field(default_factory=dict)

    # Metadata
    seed: int = 42
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "experiment_type": self.experiment_type.value,
            "description": self.description,
            "steps": self.steps,
            "parameters": self.parameters,
            "parameter_sweep": self.parameter_sweep,
            "environment": self.environment,
            "container_image": self.container_image,
            "resources": self.resources,
            "outputs": self.outputs,
            "artifacts": self.artifacts,
            "success_criteria": self.success_criteria,
            "seed": self.seed,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentManifestV2:
        return cls(**data)

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> ExperimentManifestV2:
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ExperimentRun:
    """A single execution of an experiment."""
    manifest: ExperimentManifestV2
    run_id: str
    status: str = "pending"
    start_time: datetime | None = None
    end_time: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    git_commit: str | None = None
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "run_id": self.run_id,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics,
            "logs": self.logs,
            "artifacts": self.artifacts,
            "error": self.error,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
        }


class ExperimentRunner:
    """Executes experiments with full reproducibility."""

    def __init__(
        self,
        output_dir: str = "experiment_runs",
        default_timeout: int = 3600,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_timeout = default_timeout
        self.runs: list[ExperimentRun] = []

    def run(
        self,
        manifest: ExperimentManifestV2,
        runner_fn: Callable[[dict[str, Any]], dict[str, float]],
        seed: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run an experiment with the given manifest."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(manifest.name.encode()).hexdigest()[:8]}"

        run = ExperimentRun(
            manifest=manifest,
            run_id=run_id,
            start_time=datetime.now(),
        )

        # Set seed
        seed = seed or manifest.seed
        import random
        random.seed(seed)

        # Capture git commit
        run.git_commit = self._get_git_commit()

        # Compute config hash
        run.config_hash = hashlib.sha256(
            json.dumps(manifest.parameters, sort_keys=True).encode()
        ).hexdigest()[:16]

        run.status = "running"

        try:
            # Merge parameters
            params = dict(manifest.parameters)
            if extra_params:
                params.update(extra_params)

            # Run the experiment
            metrics = runner_fn(params)

            run.metrics = metrics
            run.status = "completed"
            run.end_time = datetime.now()

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.end_time = datetime.now()
            raise
        finally:
            self.runs.append(run)
            self._save_run(run)

        return {
            "run_id": run.run_id,
            "status": run.status,
            "metrics": run.metrics,
            "error": run.error,
        }

    def _get_git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _save_run(self, run: ExperimentRun):
        """Save run to disk."""
        filename = f"{run.run_id}.json"
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(run.to_dict(), f, indent=2)

    def load_run(self, run_id: str) -> ExperimentRun | None:
        filepath = self.output_dir / f"{run_id}.json"
        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)

        # Reconstruct run (simplified)
        run = ExperimentRun(
            manifest=ExperimentManifestV2(**data["manifest"]),
            run_id=data["run_id"],
            status=data["status"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            metrics=data["metrics"],
            logs=data["logs"],
            artifacts=data["artifacts"],
            error=data["error"],
            git_commit=data["git_commit"],
            config_hash=data["config_hash"],
        )
        return run

    def list_runs(self, status: str | None = None) -> list[dict[str, Any]]:
        runs = []
        for filepath in self.output_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)

                if status and data["status"] != status:
                    continue

                runs.append({
                    "run_id": data["run_id"],
                    "experiment": data["manifest"]["name"],
                    "status": data["status"],
                    "start_time": data["start_time"],
                    "end_time": data["end_time"],
                    "metrics": data["metrics"],
                })
            except Exception:
                continue

        return sorted(runs, key=lambda x: x["start_time"], reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.load_run(run_id)
        if run:
            return run.to_dict()
        return None

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        run_a = self.load_run(run_id_a)
        run_b = self.load_run(run_id_b)

        if not run_a or not run_b:
            return {"error": "One or both runs not found"}

        metrics_a = run_a.metrics
        metrics_b = run_b.metrics

        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())

        comparison = {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "comparisons": [],
        }

        for metric in all_metrics:
            val_a = metrics_a.get(metric)
            val_b = metrics_b.get(metric)

            if val_a is not None and val_b is not None:
                diff = val_b - val_a
                pct = (diff / val_a * 100) if val_a != 0 else float('inf')
                comparison["metrics"].append({
                    "metric": metric,
                    "run_a": val_a,
                    "run_b": val_b,
                    "diff": diff,
                    "pct_change": pct,
                })

        return comparison


def create_experiment_runner(
    output_dir: str = "experiment_runs",
    default_timeout: int = 3600,
) -> ExperimentRunner:
    return ExperimentRunner(output_dir, default_timeout)


# ==================== Standard Experiment Templates ====================

def create_benchmark_manifest(
    name: str,
    benchmark_name: str,
    parameters: dict[str, Any],
    expected_metrics: dict[str, float],
    seed: int = 42,
) -> ExperimentManifestV2:
    """Create a standard benchmark experiment manifest."""
    return ExperimentManifestV2(
        name=name,
        experiment_type=ExperimentType.BENCHMARK,
        description=f"Benchmark: {benchmark_name}",
        parameters={
            "benchmark_name": benchmark_name,
            "benchmark_parameters": parameters,
        },
        expected_metrics=expected_metrics,
        seed=42,
        tags=["benchmark", benchmark_name],
    )


def create_ablation_manifest(
    name: str,
    ablation_name: str,
    target_component: str,
    ablation_type: str,
    parameter_values: list[Any],
    baseline_value: Any,
    seed: int = 42,
) -> ExperimentManifestV2:
    """Create an ablation study manifest."""
    return ExperimentManifestV2(
        name=name,
        experiment_type=ExperimentType.ABLATION,
        description=f"Ablation: {ablation_name} on {target_component}",
        parameters={
            "ablation_name": ablation_name,
            "target_component": target_component,
            "ablation_type": ablation_type,
            "parameter_name": "value",
            "parameter_values": parameter_values,
            "baseline_value": baseline_value,
        },
        seed=42,
        tags=["ablation", ablation_name, target_component],
    )


def create_redteam_manifest(
    name: str,
    attack_types: list[str],
    test_queries: list[str],
    topic_id: int,
    seed: int = 42,
) -> ExperimentManifestV2:
    """Create a red-team evaluation manifest."""
    return ExperimentManifestV2(
        name=name,
        experiment_type=ExperimentType.REDTEAM,
        description=f"Red-team evaluation with {len(attack_types)} attacks",
        parameters={
            "attack_types": attack_types,
            "test_queries": test_queries,
            "topic_id": topic_id,
        },
        seed=42,
        tags=["redteam", "security", "robustness"],
    )


def create_regression_manifest(
    name: str,
    baseline_run_id: str,
    test_config: dict[str, Any],
    regression_threshold: float = 0.05,
    seed: int = 42,
) -> ExperimentManifestV2:
    """Create a regression test manifest."""
    return ExperimentManifestV2(
        name=name,
        experiment_type=ExperimentType.REGRESSION,
        description=f"Regression test against baseline {baseline_run_id}",
        parameters={
            "baseline_run_id": baseline_run_id,
            "test_config": test_config,
            "regression_threshold": regression_threshold,
        },
        seed=42,
        tags=["regression", "ci"],
    )


def create_integration_manifest(
    name: str,
    components: list[str],
    test_scenarios: list[dict[str, Any]],
    seed: int = 42,
) -> ExperimentManifestV2:
    """Create an integration test manifest."""
    return ExperimentManifestV2(
        name=name,
        experiment_type=ExperimentType.INTEGRATION,
        description=f"Integration test for {', '.join(components)}",
        parameters={
            "components": components,
            "test_scenarios": test_scenarios,
        },
        seed=42,
        tags=["integration", "e2e"],
    )
