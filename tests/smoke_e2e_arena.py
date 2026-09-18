"""Phase 8.3.4 — Single-player E2E validation (manual, real LLM).

Runs one Controlled player over 10 questions (the first of every category,
so all ten capability axes are exercised) with 1 answer repeat, then replays
the run directory through the Scorer to produce scores.csv / aggregates.

The full output tree must be exactly the §12 Phase 8.3.4 set:

    run/
    ├── manifest.json
    ├── metrics.csv
    ├── latency.csv
    ├── tokens.csv
    ├── traces.jsonl
    └── scores.csv   <- Scorer output (plus aggregates.csv/json)

Run manually:  python tests/smoke_e2e_arena.py
"""

import sys
from pathlib import Path

from artificial_memory.research.benchmarks.arena import ArenaCategory, build_dataset
from artificial_memory.research.benchmarks.players import (
    SimpleRAGPlayer,
    get_shared_answerer,
)
from artificial_memory.research.benchmarks.runner import ArenaRunner, RunConfig
from artificial_memory.research.benchmarks.scorer import player_aggregates, score_run_dir

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    dataset = build_dataset()
    question_ids = [dataset.by_category(cat)[0].question_id for cat in ArenaCategory]
    print(f"E2E questions ({len(question_ids)}):")
    for qid in question_ids:
        q = dataset.by_id(qid)
        assert q is not None
        print(f"  [{q.category.value}] {q.question}")

    # Real benchmark runs must inject the shared frozen answerer; without it
    # players fall back to deterministic abstention (unit-test mode).
    player = SimpleRAGPlayer({"context_budget": 2000}, answerer=get_shared_answerer())
    config = RunConfig(
        arena_class="controlled",
        players=[player],
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
    print(f"\n=== scores ({len(scores)}) — {run_dir / 'scores.csv'} ===")
    for score in scores:
        print(
            f"[{score.outcome:>14}] {score.score:.2f} cov={score.ground_truth_coverage:.2f} "
            f"{score.question_id} ({score.category})\n"
            f"    answer: {score.answer_text[:160]}"
        )
        if score.forbidden_hits:
            print(f"    forbidden hits: {score.forbidden_hits}")

    print("\n=== aggregates ===")
    for agg in player_aggregates(scores):
        for key, value in agg.to_dict().items():
            print(f"  {key}: {value}")

    expected = {
        "manifest.json", "metrics.csv", "latency.csv", "tokens.csv",
        "traces.jsonl", "result.json", "environment.txt", "failures.jsonl",
        "scores.csv", "aggregates.csv", "aggregates.json",
    }
    actual = {p.name for p in run_dir.iterdir()}
    print(f"\noutput tree check: {'OK' if expected <= actual else 'MISSING: ' + str(expected - actual)}")


if __name__ == "__main__":
    main()
