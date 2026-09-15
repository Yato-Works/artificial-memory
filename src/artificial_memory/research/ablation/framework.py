from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class AblationType(StrEnum):
    """Types of ablation experiments."""
    COMPONENT_REMOVAL = "component_removal"
    PARAMETER_VARIATION = "parameter_variation"
    ALGORITHM_SWAP = "algorithm_swap"
    FEATURE_TOGGLE = "feature_toggle"
    DATA_SUBSET = "data_subset"


@dataclass
class AblationConfig:
    name: str
    ablation_type: AblationType
    target_component: str
    description: str = ""
    parameter_name: str | None = None
    parameter_values: list[Any] = field(default_factory=list)
    baseline_value: Any = None
    alternative_implementation: str | None = None
    feature_name: str | None = None
    enabled: bool = True
    subset_ratio: float = 1.0
    subset_strategy: str = "random"
    seed: int = 42
    num_trials: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    ablation_name: str
    trial_id: int
    parameter_value: Any
    metrics: dict[str, float]
    latency_ms: float
    success: bool
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationExperimentResult:
    ablation_config: AblationConfig
    baseline_metrics: dict[str, float]
    trial_results: list[AblationResult] = field(default_factory=list)
    aggregated_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    statistical_significance: dict[str, float] = field(default_factory=dict)
    effect_size: dict[str, float] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> dict[str, Any]:
        return {
            "ablation_name": self.ablation_config.name,
            "baseline": self.baseline_metrics,
            "trials": len(self.trial_results),
            "aggregated": self.aggregated_metrics,
            "effect_size": self.effect_size,
            "significance": self.statistical_significance,
        }


class AblationFramework:
    def __init__(
        self,
        store,
        recall_engine,
        context_builder,
        compressor,
        default_seed: int = 42,
        output_dir: str = "ablation_results",
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.context_builder = context_builder
        self.compressor = compressor
        self.default_seed = default_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiments: dict[str, AblationConfig] = {}
        self.results: dict[str, AblationExperimentResult] = {}

    def register_ablation(self, config):
        self.experiments[config.name] = config

    def run_ablation(
        self,
        ablation_name: str,
        baseline_runner: Callable,
        modified_runner: Callable,
        test_queries: list[str],
        topic_id: int,
    ):
        config = self.experiments.get(ablation_name)
        if not config:
            raise ValueError(f"Ablation {ablation_name} not registered")

        baseline_metrics = self._run_baseline(baseline_runner, config)

        if config.ablation_type == "parameter_variation":
            trial_results = self._run_parameter_variation(config, test_queries, topic_id)
        elif config.ablation_type == "component_removal":
            trial_results = self._run_component_removal(config, test_queries, topic_id)
        elif config.ablation_type == "algorithm_swap":
            trial_results = self._run_algorithm_swap(config, test_queries, topic_id)
        elif config.ablation_type == "feature_toggle":
            trial_results = self._run_feature_toggle(config, test_queries, topic_id)
        elif config.ablation_type == "data_subset":
            trial_results = self._run_data_subset(config, test_queries, topic_id)

        aggregated = self._aggregate_results(trial_results)
        stats = self._compute_statistics(baseline_metrics, trial_results)
        effect_sizes = self._compute_effect_sizes(baseline_metrics, trial_results)

        result = AblationExperimentResult(
            ablation_config=config,
            baseline_metrics=baseline_metrics,
            trial_results=trial_results,
            aggregated_metrics=aggregated,
            statistical_significance=stats,
            effect_size=effect_sizes,
        )

        self.results[ablation_name] = result
        return result

    def _run_baseline(self, runner, config):
        return {"tokens": 1000, "memories": 10, "latency": 100}

    def _run_parameter_variation(self, config, test_queries, topic_id):
        return []

    def _run_component_removal(self, config, test_queries, topic_id):
        return []

    def _run_algorithm_swap(self, config, test_queries, topic_id):
        return []

    def _run_feature_toggle(self, config, test_queries, topic_id):
        return []

    def _run_data_subset(self, config, test_queries, topic_id):
        return []

    def _aggregate_results(self, trial_results):
        return {}

    def _compute_statistics(self, baseline, trial_results):
        return {}

    def _compute_effect_sizes(self, baseline, trial_results):
        return {}


def create_ablation_framework(store, recall_engine, context_builder, compressor, default_seed=42, output_dir="ablation_results"):
    pass
