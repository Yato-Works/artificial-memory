"""Head-to-head comparison benchmark: AM v0.2.0 vs Simple RAG, MemoryBank, MemGPT, Mem0.

Runs all five Controlled players (S0-S4) plus the Hardened AM v0.2.0 player
over representative questions from all 10 Arena categories using the frozen local LLM.

Usage:
    python scripts/run_comparison_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from artificial_memory.research.benchmarks.arena import ArenaCategory, build_dataset
from artificial_memory.research.benchmarks.players import (
    AMv020Player,
    get_shared_answerer,
)
from artificial_memory.research.benchmarks.runner import ArenaRunner, RunConfig
from artificial_memory.research.benchmarks.scorer import build_aggregate, score_run_dir
from competitors import (
    Mem0Adapter,
    MemGPTAdapter,
    MemoryBankAdapter,
    SimpleRAGAdapter,
)

sys.stdout.reconfigure(encoding="utf-8")


def load_yaml(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return {}


class HardenedAMPlayer(AMv020Player):
    """AM v0.2.0 with DeepSeek SessionGate + Query-Relative Abstention Threshold."""

    def __init__(self, config=None, answerer=None):
        cfg = dict(config or {})
        cfg["retrieval_strategy"] = "deepseek_session"
        super().__init__(cfg, answerer=answerer)
        self.name = "AM v0.2.0 (Hardened)"

    def ingest(self, scenarios) -> None:
        super().ingest(scenarios)
        if self._runtime and hasattr(self._runtime.recall_engine, "gate"):
            gate = self._runtime.recall_engine.gate
            if gate is not None:
                gate.abstention_threshold = 6.15


def main() -> None:
    print("=" * 80)
    print("      MEMORY SYSTEM ARENA: HEAD-TO-HEAD COMPARISON BENCHMARK")
    print("      (Phase 1: Competitors Audited & Frozen via benchmark_config/*.yaml)")
    print("      Simple RAG | MemoryBank | MemGPT | Mem0 | AM v0.2.0")
    print("=" * 80)

    env_cfg = load_yaml(REPO_ROOT / "benchmark_config" / "arena_env.yaml")
    context_budget = env_cfg.get("evaluation", {}).get("context_budget_tokens", 2000)

    dataset = build_dataset()
    question_ids = [dataset.by_category(cat)[0].question_id for cat in ArenaCategory]

    answerer = get_shared_answerer()
    base_config = {"context_budget": context_budget}

    players = [
        SimpleRAGAdapter(base_config, answerer=answerer),
        MemoryBankAdapter(base_config, answerer=answerer),
        MemGPTAdapter(base_config, answerer=answerer),
        Mem0Adapter(base_config, answerer=answerer),
        AMv020Player(base_config, answerer=answerer),
        HardenedAMPlayer(base_config, answerer=answerer),
    ]

    print(f"\n[Configuration]")
    print(f"  Players: {[p.name for p in players]}")
    print(f"  Questions: {len(question_ids)} (1 per category across 10 categories)")
    print(f"  Context Budget: {base_config['context_budget']} tokens")
    print(f"  Frozen LLM: {answerer.model} @ {answerer.base_url}")
    print(f"\n[Questions to be evaluated]")
    for qid in question_ids:
        q = dataset.by_id(qid)
        print(f"  - [{q.category.value:<26}] {q.question}")

    out_dir = Path("benchmark/results/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        arena_class="controlled",
        players=players,
        dataset=dataset,
        answer_repeats=1,
        latency_repeats=1,
        output_dir=str(out_dir),
        question_ids=question_ids,
    )

    print("\nStarting benchmark run across all players...")
    t0 = time.perf_counter()
    result = ArenaRunner(config).run()
    elapsed_total = time.perf_counter() - t0
    print(f"\nRunner completed in {elapsed_total:.1f}s. Run ID: {result.run_id}")

    run_dir = out_dir / "controlled" / result.run_id
    scores = score_run_dir(run_dir, dataset)

    # 1. Aggregates Table
    print("\n" + "=" * 80)
    print("                        PER-PLAYER PERFORMANCE AGGREGATES")
    print("=" * 80)
    print(f"  {'Player':<24} | {'Accuracy':<8} | {'Pass':<6} | {'Part':<5} | {'Fail':<5} | {'GT Cov':<8} | {'Abst Acc':<8} | {'Violations'}")
    print("-" * 80)

    aggregates_summary = []
    for player in players:
        player_scores = [s for s in scores if s.player_name == player.name]
        agg = build_aggregate(player.name, player_scores)
        abst_str = f"{agg.abstention_accuracy:.2f}" if agg.abstention_accuracy is not None else "N/A"
        print(
            f"  {player.name:<24} | "
            f"{agg.answer_accuracy * 100:>7.1f}% | "
            f"{agg.pass_count:>2}/{agg.question_count:<2} | "
            f"{agg.partial_count:>5} | "
            f"{agg.fail_count:>5} | "
            f"{agg.mean_ground_truth_coverage * 100:>7.1f}% | "
            f"{abst_str:>8} | "
            f"{agg.forbidden_violation_rate:>10.2f}"
        )
        aggregates_summary.append({
            "player": player.name,
            "accuracy": agg.answer_accuracy,
            "pass_count": agg.pass_count,
            "partial_count": agg.partial_count,
            "fail_count": agg.fail_count,
            "gt_coverage": agg.mean_ground_truth_coverage,
            "abstention_accuracy": agg.abstention_accuracy,
            "forbidden_violation_rate": agg.forbidden_violation_rate,
        })

    # 2. Outcome Matrix Table
    print("\n" + "=" * 80)
    print("                        CATEGORY OUTCOME MATRIX")
    print("=" * 80)
    categories = [c.value for c in ArenaCategory]
    short_cats = [c[:6] for c in categories]
    header = f"  {'Player':<24} | " + " | ".join(f"{sc:>6}" for sc in short_cats)
    print(header)
    print("-" * len(header))
    for player in players:
        row = f"  {player.name:<24} | "
        outcomes = []
        for cat in categories:
            match = [s for s in scores if s.player_name == player.name and s.category == cat]
            out = match[0].outcome[:4] if match else "----"
            outcomes.append(f"{out:>6}")
        row += " | ".join(outcomes)
        print(row)

    # 3. Cost & Latency Table
    print("\n" + "=" * 80)
    print("                        COST & LATENCY BREAKDOWN")
    print("=" * 80)
    tokens_csv_path = run_dir / "tokens.csv"
    if tokens_csv_path.exists():
        print(tokens_csv_path.read_text(encoding="utf-8"))

    # Save summary artifact
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"comparison_summary_{stamp}.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "run_id": result.run_id,
        "elapsed_seconds": round(elapsed_total, 2),
        "aggregates": aggregates_summary,
        "questions": question_ids,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary JSON saved: {summary_path}")


if __name__ == "__main__":
    main()
