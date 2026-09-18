"""Phase 8.3.5 — Multi-player smoke (manual, real LLM).

All five Controlled players over the same 10 questions (one per category),
1 answer repeat each (~60 frozen-LLM calls total). Run manually:

    python tests/smoke_multi_player.py
"""

import sys
from pathlib import Path

from artificial_memory.research.benchmarks.arena import ArenaCategory, build_dataset
from artificial_memory.research.benchmarks.players import create_controlled_players
from artificial_memory.research.benchmarks.runner import ArenaRunner, RunConfig
from artificial_memory.research.benchmarks.scorer import build_aggregate, score_run_dir

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    dataset = build_dataset()
    question_ids = [dataset.by_category(cat)[0].question_id for cat in ArenaCategory]

    players = create_controlled_players({"context_budget": 2000})
    print(f"players: {[p.name for p in players]}")
    print(f"questions: {len(question_ids)} (1 per category), 1 repeat each")

    config = RunConfig(
        arena_class="controlled",
        players=players,
        dataset=dataset,
        answer_repeats=1,
        latency_repeats=1,
        output_dir="benchmark/results",
        question_ids=question_ids,
    )
    result = ArenaRunner(config).run()
    run_dir = Path("benchmark/results") / "controlled" / result.run_id
    print(f"\nrun_id: {result.run_id}")
    print(f"failures: {len(result.failures)}")
    for failure in result.failures:
        print(f"  FAILURE {failure['player']} @ {failure['stage']}: {failure['error'][:200]}")

    scores = score_run_dir(run_dir, dataset)
    print(f"\n=== per-player aggregates ({run_dir / 'scores.csv'}) ===")
    for name in [p.name for p in players]:
        player_scores = [s for s in scores if s.player_name == name]
        agg = build_aggregate(name, player_scores)
        print(
            f"  {name:<22} acc={agg.answer_accuracy:.2f} "
            f"pass={agg.pass_count}/{agg.question_count} "
            f"partial={agg.partial_count} fp={agg.false_positive_count} "
            f"forbidden={agg.forbidden_violation_rate:.2f} "
            f"abst_acc={agg.abstention_accuracy if agg.abstention_accuracy is None else round(agg.abstention_accuracy, 2)}"
        )

    print("\n=== official outcome matrix (player x category) ===")
    categories = [c.value for c in ArenaCategory]
    header = "  {:<22}".format("player") + "".join(f"{c[:10]:>12}" for c in categories)
    print(header)
    for name in [p.name for p in players]:
        row = f"  {name:<22}"
        for cat in categories:
            match = [s for s in scores if s.player_name == name and s.category == cat]
            outcome = match[0].outcome[:3] if match else "-"
            row += f"{outcome:>12}"
        print(row)

    print("\ntokens.csv:")
    print((run_dir / "tokens.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
