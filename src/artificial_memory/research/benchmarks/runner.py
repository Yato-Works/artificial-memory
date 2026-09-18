"""Arena Runner for Memory Arena (Phase 8.1).

Executes benchmark runs per Benchmark_Plan.txt Rev.2:
- Controlled Arena: fixed 2K budget, write-side cost, repeat protocol
- Real-System Arena: native behavior, actual token measurement
- Manifest generation with full reproducibility
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from artificial_memory.research.benchmarks.arena import (
    ArenaDataset,
    ArenaQuestion,
)
from artificial_memory.research.benchmarks.llm import (
    FROZEN_MODEL,
    FROZEN_PROMPT_HASHES,
    frozen_config_sha256,
)
from artificial_memory.research.benchmarks.player import (
    Answer,
    CostReport,
    Player,
    majority_index,
)
from artificial_memory.research.experiments.manifest import (
    ExperimentManifestV2,
    create_benchmark_manifest,
)

# ==================== Configuration ====================

CONTROLLED_BUDGET = 2000  # Official Context Budget per §5
ANSWER_REPEATS = 3        # §9 Repeat Protocol: 回答3回
LATENCY_REPEATS = 5       # §9 Repeat Protocol: latency 5回以上でp50/p95
ARENA_VERSION = "v0.2.0-arena-1"


def selected_questions(
    dataset: ArenaDataset, question_ids: Sequence[str] | None
) -> tuple[ArenaQuestion, ...]:
    """Questions a run will ask: the full dataset, or a frozen subset.

    Subset runs (smoke / E2E validation, §12 Phase 8.3.4) keep the *full*
    dataset hash in the manifest and additionally record the applied filter, so
    a partial run remains traceable to the same freeze and can never be
    mistaken for a production run. Order always follows dataset order, and
    unknown ids are ignored.
    """
    if question_ids is None:
        return tuple(dataset.questions)
    wanted = set(question_ids)
    return tuple(q for q in dataset.questions if q.question_id in wanted)


def question_ids_sha256(questions: Sequence[ArenaQuestion]) -> str:
    """Stable digest of the asked question ids (manifest provenance)."""
    payload = "\n".join(q.question_id for q in questions)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class RunConfig:
    """Configuration for a single arena run."""
    arena_class: str  # "controlled" | "real"
    players: list[Player]
    dataset: ArenaDataset
    answer_repeats: int = ANSWER_REPEATS
    latency_repeats: int = LATENCY_REPEATS
    seed: int = 42
    output_dir: str = "benchmark/results"
    # Optional question subset for smoke / E2E validation runs (§12 Phase 8.3.4).
    # The dataset hash recorded in the manifest always stays the *full* frozen
    # dataset hash, so a subset run is still traceable to the same freeze.
    question_ids: list[str] | None = None


@dataclass
class QuestionResult:
    """Result for a single question across all repeats."""
    question: ArenaQuestion
    answers: list[Answer] = field(default_factory=list)
    latency_samples: list[float] = field(default_factory=list)

    @property
    def majority_answer(self) -> Answer | None:
        """Majority vote on answer text (Repeat Protocol §9).

        Uses :func:`majority_index` so the runner and the Scorer share one
        deterministic tie-break rule (smallest ``run_index`` wins), rather than
        relying on dict insertion order.
        """
        if not self.answers:
            return None
        index = majority_index([ans.text for ans in self.answers])
        if index < 0:
            return self.answers[0]
        return self.answers[index]

    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        return sorted_samples[len(sorted_samples) // 2]

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]


@dataclass
class PlayerRunResult:
    """Complete results for one player."""
    player_name: str
    arena_class: str
    question_results: dict[str, QuestionResult] = field(default_factory=dict)
    final_costs: CostReport | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "arena_class": self.arena_class,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "manifest": self.manifest,
            "final_costs": self.final_costs.to_dict() if self.final_costs else None,
            "questions": {
                qid: {
                    "question_id": qr.question.question_id,
                    "category": qr.question.category.value,
                    "answers": [a.to_dict() for a in qr.answers],
                    "majority_answer": qr.majority_answer.to_dict() if qr.majority_answer else None,
                    "avg_latency_ms": qr.avg_latency_ms,
                    "p50_latency_ms": qr.p50_latency_ms,
                    "p95_latency_ms": qr.p95_latency_ms,
                }
                for qid, qr in self.question_results.items()
            },
        }


@dataclass
class ArenaRunResult:
    """Complete results for an arena run (all players)."""
    run_id: str
    arena_class: str
    dataset_hash: str
    dataset_version: str
    player_results: dict[str, PlayerRunResult] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    failures: list[dict[str, Any]] = field(default_factory=list)
    question_count: int = 0
    question_ids_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "arena_class": self.arena_class,
            "dataset_hash": self.dataset_hash,
            "dataset_version": self.dataset_version,
            "question_count": self.question_count,
            "question_ids_sha256": self.question_ids_sha256,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "failures": self.failures,
            "players": {
                name: pr.to_dict()
                for name, pr in self.player_results.items()
            },
        }


class ArenaRunner:
    """Executes Memory Arena benchmark runs.

    Supports both Controlled and Real-System arenas per Benchmark_Plan.txt Rev.2.
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.arena_class
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._failures: list[dict[str, Any]] = []

    def run(self) -> ArenaRunResult:
        """Execute the full arena run for all players."""
        run_id = f"{self.config.arena_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_hash = self.config.dataset.content_hash()

        # Run-specific directory
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.failures_path = run_dir / "failures.jsonl"
        # Stable output schema: failures.jsonl always exists (empty when clean)
        self.failures_path.touch()

        questions = selected_questions(self.config.dataset, self.config.question_ids)

        result = ArenaRunResult(
            run_id=run_id,
            arena_class=self.config.arena_class,
            dataset_hash=dataset_hash,
            dataset_version=self.config.dataset.version,
            started_at=datetime.now().isoformat(),
            question_count=len(questions),
            question_ids_sha256=question_ids_sha256(questions),
        )

        # Ingest phase: each player processes the shared history
        failed_ingest: set[str] = set()
        for player in self.config.players:
            player.reset_costs()
            try:
                start = time.perf_counter()
                player.ingest(list(self.config.dataset.scenarios))
            except Exception as e:
                self._record_failure(player.name, "ingest", str(e))
                failed_ingest.add(player.name)

        # Question phase: each player answers all questions. A player whose
        # ingest failed is skipped entirely: answering from an empty store
        # would record meaningless quality numbers for a broken run.
        for player in self.config.players:
            if player.name in failed_ingest:
                continue

            player_result = PlayerRunResult(
                player_name=player.name,
                arena_class=self.config.arena_class,
                started_at=datetime.now().isoformat(),
            )

            for question in questions:
                q_result = QuestionResult(question=question)

                # Answer repeats (3x for scoring)
                for repeat_idx in range(self.config.answer_repeats):
                    try:
                        start = time.perf_counter()
                        answer = player.answer(question, self.config.players[0].config.get('context_budget', CONTROLLED_BUDGET))
                        elapsed_ms = (time.perf_counter() - start) * 1000

                        answer.run_index = repeat_idx
                        answer.answer_latency_ms = elapsed_ms
                        q_result.answers.append(answer)
                        q_result.latency_samples.append(elapsed_ms)
                        # Costs are recorded by the player itself inside
                        # answer() (real token usage, §6). The runner must not
                        # add a second zero-token read cost or call counts
                        # double up.
                    except Exception as e:
                        self._record_failure(player.name, f"answer:{question.question_id}:repeat{repeat_idx}", str(e))

                # Additional latency-only repeats (to reach 5+ for p50/p95)
                for repeat_idx in range(self.config.answer_repeats, self.config.latency_repeats):
                    try:
                        start = time.perf_counter()
                        answer = player.answer(question, self.config.players[0].config.get('context_budget', CONTROLLED_BUDGET))
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        q_result.latency_samples.append(elapsed_ms)
                        # Best-effort timing only: the player records the real
                        # cost of this call itself, the runner adds nothing.
                    except Exception:
                        pass  # Latency repeats are best-effort

                player_result.question_results[question.question_id] = q_result

            # Finalize player costs
            player_result.final_costs = player.finalize()
            player_result.finished_at = datetime.now().isoformat()
            result.player_results[player.name] = player_result

        result.finished_at = datetime.now().isoformat()
        result.failures = self._failures

        # Persist results
        self._persist_results(result)
        return result

    def _record_failure(self, player: str, stage: str, error: str) -> None:
        failure = {
            "timestamp": datetime.now().isoformat(),
            "player": player,
            "stage": stage,
            "error": error,
            "arena_class": self.config.arena_class,
        }
        self._failures.append(failure)
        # Append to failures.jsonl immediately
        with self.failures_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    def _persist_results(self, result: ArenaRunResult) -> None:
        """Write all output artifacts per §9."""
        run_dir = self.output_dir / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # manifest.json
        manifest = {
            "benchmark_version": ARENA_VERSION,
            "dataset_sha256": result.dataset_hash,
            "dataset_version": result.dataset_version,
            "arena_class": self.config.arena_class,
            "run_id": result.run_id,
            "seed": self.config.seed,
            "answer_repeats": self.config.answer_repeats,
            "latency_repeats": self.config.latency_repeats,
            "context_budget": CONTROLLED_BUDGET if self.config.arena_class == "controlled" else "native",
            "question_count": result.question_count,
            "question_ids_sha256": result.question_ids_sha256,
            "question_filter_active": self.config.question_ids is not None,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # metrics.csv (per-question aggregated)
        self._write_metrics_csv(run_dir, result)

        # latency.csv (per-question p50/p95)
        self._write_latency_csv(run_dir, result)

        # tokens.csv (per-player token breakdown)
        self._write_tokens_csv(run_dir, result)

        # raw traces
        (run_dir / "traces.jsonl").write_text(
            "\n".join(json.dumps(pr.to_dict(), ensure_ascii=False) for pr in result.player_results.values()),
            encoding="utf-8"
        )

        # environment.txt
        self._write_environment(run_dir)

        # Full result
        (run_dir / "result.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _write_metrics_csv(self, run_dir: Path, result: ArenaRunResult) -> None:
        lines = ["player,question_id,category,context_tokens,retrieval_ms,context_build_ms,answer_ms,total_latency_ms"]
        for pr in result.player_results.values():
            for qr in pr.question_results.values():
                maj = qr.majority_answer
                if maj:
                    lines.append(f"{pr.player_name},{qr.question.question_id},{qr.question.category.value},"
                                 f"{maj.context_tokens_used},{maj.retrieval_latency_ms:.1f},"
                                 f"{maj.context_build_latency_ms:.1f},{maj.answer_latency_ms:.1f},"
                                 f"{qr.avg_latency_ms:.1f}")
        (run_dir / "metrics.csv").write_text("\n".join(lines), encoding="utf-8")

    def _write_latency_csv(self, run_dir: Path, result: ArenaRunResult) -> None:
        lines = ["player,question_id,category,p50_ms,p95_ms,avg_ms"]
        for pr in result.player_results.values():
            for qr in pr.question_results.values():
                lines.append(f"{pr.player_name},{qr.question.question_id},{qr.question.category.value},"
                             f"{qr.p50_latency_ms:.1f},{qr.p95_latency_ms:.1f},{qr.avg_latency_ms:.1f}")
        (run_dir / "latency.csv").write_text("\n".join(lines), encoding="utf-8")

    def _write_tokens_csv(self, run_dir: Path, result: ArenaRunResult) -> None:
        lines = ["player,write_llm_calls,write_prompt_tokens,write_completion_tokens,"
                 "retrieval_calls,context_calls,answer_llm_calls,answer_prompt_tokens,"
                 "answer_completion_tokens,total_tokens,context_budget,context_used,budget_violations"]
        for pr in result.player_results.values():
            if pr.final_costs:
                c = pr.final_costs
                lines.append(f"{pr.player_name},{c.write_llm_calls},{c.write_llm_prompt_tokens},"
                             f"{c.write_llm_completion_tokens},{c.retrieval_calls},"
                             f"{c.context_construction_calls},{c.answer_llm_calls},"
                             f"{c.answer_llm_prompt_tokens},{c.answer_llm_completion_tokens},"
                             f"{c.total_tokens},{c.context_budget_tokens},{c.context_tokens_used},"
                             f"{c.budget_violations}")
        (run_dir / "tokens.csv").write_text("\n".join(lines), encoding="utf-8")

    def _write_environment(self, run_dir: Path) -> None:
        import platform
        import sys
        try:
            import torch
            torch_version = torch.__version__
        except ImportError:
            torch_version = "not installed"
        try:
            import transformers
            transformers_version = transformers.__version__
        except ImportError:
            transformers_version = "not installed"

        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch_version,
            "transformers": transformers_version,
            "timestamp": datetime.now().isoformat(),
        }
        (run_dir / "environment.txt").write_text(
            "\n".join(f"{k}: {v}" for k, v in env.items()), encoding="utf-8"
        )


def create_arena_manifest(
    arena_class: str,
    players: list[Player],
    dataset: ArenaDataset,
    extra: dict[str, Any] | None = None,
) -> ExperimentManifestV2:
    """Create a Phase 8 benchmark manifest from run configuration."""
    player_names = [p.name for p in players]
    manifest = create_benchmark_manifest(
        name=f"memory-arena-{arena_class}",
        benchmark_name=f"Phase 8 {arena_class.title()} Arena",
        parameters={
            "arena_class": arena_class,
            "players": player_names,
            "dataset_version": dataset.version,
            "dataset_hash": dataset.content_hash(),
            "context_budget": CONTROLLED_BUDGET if arena_class == "controlled" else "native",
            "answer_repeats": ANSWER_REPEATS,
            "latency_repeats": LATENCY_REPEATS,
            # Fixed LLM Configuration freeze (§5, Phase 8.2)
            "llm_model": FROZEN_MODEL,
            "llm_config_sha256": frozen_config_sha256(),
            **FROZEN_PROMPT_HASHES,
            **(extra or {}),
        },
        expected_metrics={},
        seed=42,
    )
    manifest.tags = ["phase8", "arena", arena_class] + player_names
    return manifest


__all__ = [
    "ArenaRunner",
    "RunConfig",
    "QuestionResult",
    "PlayerRunResult",
    "ArenaRunResult",
    "CONTROLLED_BUDGET",
    "ANSWER_REPEATS",
    "LATENCY_REPEATS",
    "create_arena_manifest",
]
