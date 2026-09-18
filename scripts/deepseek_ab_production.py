"""DeepSeek Raid — production A/B run (Phase 8.9, 3,000 calls).

The Benchmark_Plan §8.3.6 production-run shape, applied to the retrieval
A/B: all 200 dataset questions × 5 answer calls (3 scored repeats + 2
latency-only repeats) × 3 retrieval arms = **3,000 calls**.

    classic  frozen BasicRecallEngine (pre-raid baseline)
    cache    CSA2 plan cache (retrieval_strategy="deepseek")
    session  SessionGate + full-store window (retrieval_strategy="deepseek_session")

Execution model: questions are processed in chunks (default 20) and each
(arm, chunk) run is persisted immediately in its own §12 artifact directory,
so a crash loses at most one chunk.  Raw per-answer values are accumulated
across chunks so the final p50/p95 are exact over all 1,000 calls per arm.

Run:
    python scripts/deepseek_ab_production.py --dry-run --max-questions 20
    python scripts/deepseek_ab_production.py --max-questions 40   # medium
    python scripts/deepseek_ab_production.py                      # full 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import deepseek_ab_small as ab
from artificial_memory.research.benchmarks.arena import ArenaCategory, build_dataset
from artificial_memory.research.benchmarks.players import AMv020Player
from artificial_memory.research.benchmarks.runner import ArenaRunner, RunConfig
from artificial_memory.research.benchmarks.scorer import player_aggregates

sys.stdout.reconfigure(encoding="utf-8")


def _pctl(values, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * pct))))
    return s[idx]


def _parse_traces(run_dir: Path) -> dict:
    """Raw per-question answer records from one chunk run dir."""
    text = (run_dir / "traces.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    out: dict = {}
    for pr in records:
        questions = pr.get("questions") or pr.get("question_results") or {}
        for qid, qr in questions.items():
            out.setdefault(qid, []).extend(qr.get("answers") or [])
    return out


class ArmAccumulator:
    """Accumulates raw evidence across chunks for one arm."""

    def __init__(self, label: str, gate_pool, candidate_window):
        self.label = label
        self.gate_pool = gate_pool
        self.candidate_window = candidate_window
        self.run_dirs: list[str] = []
        self.failures = 0
        self._score_objs: list = []           # raw QuestionScore objects
        self.scores: list[dict] = []          # per-question score rows
        self.retrieval_ms: list[float] = []
        self.ctx_tokens: list[int] = []
        self.answer_ms: list[float] = []
        self.selections: dict = {}   # first-repeat ids per qid
        self._repeat_pairs = 0
        self._repeat_stable = 0
        self.plan_stats = None
        self.gate_stats = None

    def add_chunk(self, result, run_dir: Path, dataset) -> None:
        self.run_dirs.append(str(run_dir))
        self.failures += len(result.failures)
        # Raw QuestionScore objects for exact cross-chunk aggregation, plus
        # the per-question dict rows for the evidence file.
        scores = ab.score_run_dir(run_dir, dataset)
        self._score_objs.extend(scores)
        self.scores.extend(
            {"question_id": s.question_id, "category": s.category,
             "score": round(s.score, 4), "outcome": s.outcome,
             "coverage": round(s.ground_truth_coverage, 4)}
            for s in scores
        )

        for qid, answers in _parse_traces(run_dir).items():
            ids_seq = []
            for ans in answers:
                self.retrieval_ms.append(
                    float(ans.get("retrieval_latency_ms") or 0.0))
                self.ctx_tokens.append(int(ans.get("context_tokens_used") or 0))
                self.answer_ms.append(float(ans.get("answer_latency_ms") or 0.0))
                ids_seq.append(
                    list((ans.get("metadata") or {}).get("memory_ids") or []))
            if qid not in self.selections and ids_seq:
                self.selections[qid] = ids_seq[0]
            for i in range(len(ids_seq) - 1):
                self._repeat_pairs += 1
                if ids_seq[i] == ids_seq[i + 1]:
                    self._repeat_stable += 1

    def merge_engine_stats(self, plan_stats, gate_stats) -> None:
        if plan_stats:
            if self.plan_stats is None:
                self.plan_stats = dict(plan_stats)
            else:
                for k, v in plan_stats.items():
                    self.plan_stats[k] = self.plan_stats.get(k, 0) + v
        if gate_stats:
            if self.gate_stats is None:
                self.gate_stats = dict(gate_stats)
            else:
                for k, v in gate_stats.items():
                    self.gate_stats[k] = self.gate_stats.get(k, 0) + v

    def summarise(self) -> dict:
        agg = player_aggregates(self._score_objs)
        return {
            "arm": self.label,
            "gate_pool": self.gate_pool,
            "candidate_window": self.candidate_window,
            "run_dirs": self.run_dirs,
            "chunks": len(self.run_dirs),
            "failures": self.failures,
            "calls_observed": len(self.retrieval_ms),
            "plan_stats": self.plan_stats,
            "gate_stats": self.gate_stats,
            "retrieval_latency": {
                "mean_ms": round(statistics.mean(self.retrieval_ms), 3)
                if self.retrieval_ms else 0.0,
                "p50_ms": round(_pctl(self.retrieval_ms, 0.50), 3),
                "p95_ms": round(_pctl(self.retrieval_ms, 0.95), 3),
            },
            "answer_latency": {
                "mean_ms": round(statistics.mean(self.answer_ms), 3)
                if self.answer_ms else 0.0,
                "p50_ms": round(_pctl(self.answer_ms, 0.50), 3),
                "p95_ms": round(_pctl(self.answer_ms, 0.95), 3),
            },
            "context_tokens": {
                "mean": round(statistics.mean(self.ctx_tokens), 1)
                if self.ctx_tokens else 0.0,
                "total": sum(self.ctx_tokens),
            },
            "repeat_selection_stability": (
                round(self._repeat_stable / self._repeat_pairs, 4)
                if self._repeat_pairs else 1.0
            ),
            "question_count": len(self.selections),
            "aggregates": [a.to_dict() for a in agg],
        }


def _build_chunk_config(args, dataset, player, question_ids) -> RunConfig:
    return RunConfig(
        arena_class="controlled",
        players=[player],
        dataset=dataset,
        answer_repeats=args.answer_repeats,
        latency_repeats=args.latency_repeats,
        output_dir=args.output_dir,
        question_ids=question_ids,
    )


def _fresh_player(label: str, answerer):
    """A new player instance per (arm, chunk): fresh runtime + fresh caches."""
    if label == "classic":
        return AMv020Player({"context_budget": 2000}, answerer=answerer)
    if label == "cache":
        return ab.StrategyPlayer(None, None, answerer=answerer)
    if label == "session":
        return AMv020Player(
            {"context_budget": 2000, "retrieval_strategy": "deepseek_session"},
            answerer=answerer,
        )
    raise SystemExit(f"unknown arm: {label}")


def _engine_stats(player) -> tuple:
    engine = player._runtime.recall_engine
    plan = (engine.plan_cache.stats_snapshot()
            if hasattr(engine, "plan_cache") else None)
    gate = (engine.gate.snapshot()
            if getattr(engine, "gate", None) else None)
    return plan, gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSeek Raid production A/B (3,000-call shape)")
    parser.add_argument("--dry-run", action="store_true",
                        help="unit mode: no LLM (deterministic abstention)")
    parser.add_argument("--arms", default="classic,cache,session")
    parser.add_argument("--max-questions", type=int, default=200,
                        help="cap on questions (200 = full production run)")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--answer-repeats", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--output-dir", default="benchmark/results")
    args = parser.parse_args()

    dataset = build_dataset()
    all_qids: list[str] = []
    for cat in ArenaCategory:
        all_qids.extend(q.question_id for q in dataset.by_category(cat))
    all_qids = all_qids[:args.max_questions]

    arms_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    total_calls = (len(all_qids) * args.latency_repeats * len(arms_list))
    print("=== DeepSeek Raid production A/B ===")
    print(f"  questions: {len(all_qids)}  arms: {arms_list}")
    print(f"  calls: {len(all_qids)} x {args.latency_repeats} repeats "
          f"x {len(arms_list)} arms = {total_calls}")
    print(f"  dry_run={args.dry_run} chunk={args.chunk_size}")

    answerer = None
    if not args.dry_run:
        from artificial_memory.research.benchmarks.players import get_shared_answerer
        answerer = get_shared_answerer()

    wanted = arms_list
    accs = {label: ArmAccumulator(label, None, None) for label in wanted}

    t0 = time.perf_counter()
    for label in wanted:
        chunks = [all_qids[i:i + args.chunk_size]
                  for i in range(0, len(all_qids), args.chunk_size)]
        for ci, chunk_qids in enumerate(chunks):
            player = _fresh_player(label, answerer)
            result = ArenaRunner(
                _build_chunk_config(args, dataset, player, chunk_qids)
            ).run()
            run_dir = Path(args.output_dir) / "controlled" / result.run_id
            accs[label].add_chunk(result, run_dir, dataset)
            plan, gate = _engine_stats(player)
            accs[label].merge_engine_stats(plan, gate)
            elapsed = time.perf_counter() - t0
            print(f"  [{label} chunk {ci + 1}/{len(chunks)}] "
                  f"scored={len(accs[label].scores)} "
                  f"failures={accs[label].failures} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    arm_summaries = {label: accs[label].summarise() for label in wanted}

    # Cross-arm selection agreement vs classic (first chunk, first repeat).
    arms_for_agreement = {
        label: {"run_dir": accs[label].run_dirs[0]} for label in wanted
    }
    agreement = (ab.cross_arm_agreement(arms_for_agreement)
                 if "classic" in wanted else {})

    print("\n=== PRODUCTION A/B RESULTS ===")
    print(f"  {'arm':<10} {'acc':>6} {'cov':>6} {'p50ms':>8} "
          f"{'p95ms':>8} {'tokens':>8} {'stab':>6}")
    for label, s in arm_summaries.items():
        def _pick(key):
            for a in s["aggregates"]:
                if key in a:
                    return a[key]
            return None
        acc, cov = (_pick("answer_accuracy"),
                    _pick("mean_ground_truth_coverage"))
        acc_s = f"{acc:.3f}" if isinstance(acc, (int, float)) else str(acc)
        cov_s = f"{cov:.3f}" if isinstance(cov, (int, float)) else str(cov)
        print(f"  {label:<10} {acc_s:>6} {cov_s:>6} "
              f"{s['retrieval_latency']['p50_ms']:>8.1f} "
              f"{s['retrieval_latency']['p95_ms']:>8.1f} "
              f"{s['context_tokens']['mean']:>8.1f} "
              f"{s['repeat_selection_stability']:>6.2f}")
    if agreement:
        print(f"  selection agreement vs classic: {agreement}")

    summary = {
        "experiment": "deepseek-raid-production-ab",
        "timestamp": datetime.now().isoformat(),
        "question_count": len(all_qids),
        "answer_repeats": args.answer_repeats,
        "latency_repeats": args.latency_repeats,
        "total_calls_design": total_calls,
        "dry_run": args.dry_run,
        "arms": arm_summaries,
        "cross_arm_selection_agreement": agreement,
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }
    out_dir = Path(args.output_dir) / "deepseek_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"production_{stamp}.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nevidence written: {out_path}")


if __name__ == "__main__":
    main()
