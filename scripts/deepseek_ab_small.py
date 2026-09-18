"""DeepSeek Raid — small-scale A/B E2E (Phase 8.5, manual, real LLM).

Runs the S4 player (AM v0.2.0) over the 10-question E2E subset (the first
question of every category, §12 Phase 8.3.4) in three retrieval arms:

    classic  RuntimeConfig.retrieval_strategy="classic"
             -> frozen BasicRecallEngine (the pre-raid baseline)
    cache    RuntimeConfig.retrieval_strategy="deepseek"
             -> CSA2 plan cache only (the committed runtime wiring)
    gate     "deepseek" + HierarchicalGate(pool_size=--gate-pool)
             -> plan cache + HSI lexical pre-filter (ablation arm)

All three arms share the same dataset, the same frozen answer LLM, the same
2,000 token budget and the same frozen Scorer; only the candidate pipeline
differs.  EphemeralStore is deliberately NOT in any arm (it stays outside the
Runtime), so any measured delta belongs to exactly one of the two raid
mechanisms.

Run:
    python scripts/deepseek_ab_small.py            # real frozen LLM
    python scripts/deepseek_ab_small.py --dry-run  # unit mode, no LLM
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from artificial_memory.recall.retrieval_cache import HierarchicalGate
from artificial_memory.research.benchmarks.arena import ArenaCategory, build_dataset
from artificial_memory.research.benchmarks.players import AMv020Player
from artificial_memory.research.benchmarks.runner import ArenaRunner, RunConfig
from artificial_memory.research.benchmarks.scorer import player_aggregates, score_run_dir

sys.stdout.reconfigure(encoding="utf-8")

QUESTION_IDS: list[str] = []  # filled in main()


class MeasuredGate(HierarchicalGate):
    """Measurement wrapper: records pool traffic without changing behaviour."""

    def __init__(self, pool_size: int, min_keep: int = 8):
        super().__init__(pool_size=pool_size, min_keep=min_keep)
        self.total_in = 0
        self.total_out = 0
        self.calls = 0

    def filter(self, query, memories):
        out = super().filter(query, memories)
        self.calls += 1
        self.total_in += len(memories)
        self.total_out += len(out)
        return out

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "narrowed": self.total_in - self.total_out,
        }


class StrategyPlayer(AMv020Player):
    """AMv020Player + optional post-ingest gate attachment (ablation arm).

    The gate is attached *after* ingest because the committed runtime wiring
    constructs the raid engine with ``gate=None`` (pure plan-cache mode).
    Attaching the attribute post-hoc is exactly equivalent to constructor
    injection and keeps the frozen facade untouched.
    """

    def __init__(self, gate_pool, answerer):
        self._gate_pool = gate_pool
        super().__init__(
            {"context_budget": 2000, "retrieval_strategy": "deepseek"},
            answerer=answerer,
        )

    def ingest(self, scenarios) -> None:
        super().ingest(scenarios)
        if self._gate_pool is not None and self._runtime is not None:
            from artificial_memory.recall.deepseek_engine import DeepSeekRecallEngine

            engine = self._runtime.recall_engine
            assert isinstance(engine, DeepSeekRecallEngine)
            engine.gate = MeasuredGate(pool_size=self._gate_pool)


def _pctl(values, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * pct))))
    return s[idx]


def _memory_ids(answer) -> list:
    meta = answer.metadata or {}
    return list(meta.get("memory_ids") or [])


def _collect_retrieval_evidence(result) -> dict:
    """Per-answer retrieval metrics (LLM latency excluded) from a run."""
    retrieval_ms, ctx_tokens = [], []
    repeat_stable = repeat_pairs = 0
    for pr in result.player_results.values():
        for qr in pr.question_results.values():
            ids_by_repeat = []
            for ans in qr.answers:
                retrieval_ms.append(ans.retrieval_latency_ms)
                ctx_tokens.append(ans.context_tokens_used)
                ids_by_repeat.append(_memory_ids(ans))
            for i in range(len(ids_by_repeat) - 1):
                repeat_pairs += 1
                if ids_by_repeat[i] == ids_by_repeat[i + 1]:
                    repeat_stable += 1
    return {
        "retrieval_latency": {
            "n": len(retrieval_ms),
            "mean_ms": round(statistics.mean(retrieval_ms), 3) if retrieval_ms else 0.0,
            "p50_ms": round(_pctl(retrieval_ms, 0.50), 3),
            "p95_ms": round(_pctl(retrieval_ms, 0.95), 3),
        },
        "context_tokens": {
            "mean": round(statistics.mean(ctx_tokens), 1) if ctx_tokens else 0.0,
            "total": sum(ctx_tokens),
        },
        "repeat_selection_stability": (
            round(repeat_stable / repeat_pairs, 4) if repeat_pairs else 1.0
        ),
    }


def _score_block(run_dir: Path, dataset) -> dict:
    """Frozen-Scorer output for one run directory."""
    scores = score_run_dir(run_dir, dataset)
    agg = player_aggregates(scores)
    return {
        "scores": [
            {"question_id": s.question_id, "category": s.category,
             "score": round(s.score, 4), "outcome": s.outcome,
             "coverage": round(s.ground_truth_coverage, 4)}
            for s in scores
        ],
        "aggregates": [a.to_dict() for a in agg],
    }


def _print_arm(label: str, result, summary: dict) -> None:
    print(f"\n=== arm [{label}] run_id={result.run_id} failures={len(result.failures)} ===")
    print(f"  store memories: {summary.get('store_memory_count', 'n/a')}")
    print(f"  plan stats: {summary['plan_stats']}   gate: {summary['gate_stats']}")
    lat = summary["retrieval_latency"]
    print(f"  retrieval latency mean/p50/p95: {lat['mean_ms']} / {lat['p50_ms']} / {lat['p95_ms']} ms")
    print(f"  context tokens mean: {summary['context_tokens']['mean']}")
    print(f"  repeat selection stability: {summary['repeat_selection_stability']}")
    print(f"  scorer: {json.dumps(summary['aggregates'], ensure_ascii=False)}")


def _build_config(args, dataset, player) -> RunConfig:
    return RunConfig(
        arena_class="controlled",
        players=[player],
        dataset=dataset,
        answer_repeats=args.answer_repeats,
        latency_repeats=args.latency_repeats,
        output_dir=args.output_dir,
        question_ids=QUESTION_IDS,
    )


def run_arm(label: str, gate_pool, answerer, dataset, args) -> dict:
    """Run one deepseek arm (cache-only or cache+gate) and summarise it."""
    player = StrategyPlayer(gate_pool, answerer=answerer)
    result = ArenaRunner(_build_config(args, dataset, player)).run()
    run_dir = Path(args.output_dir) / "controlled" / result.run_id

    engine = player._runtime.recall_engine
    plan_stats = (
        engine.plan_cache.stats_snapshot()
        if hasattr(engine, "plan_cache") else None
    )
    gate_stats = engine.gate.snapshot() if getattr(engine, "gate", None) else None

    summary = {
        "arm": label,
        "gate_pool": gate_pool,
        "run_id": result.run_id,
        "run_dir": str(run_dir),
        "failures": len(result.failures),
        "store_memory_count": len(player._runtime.store.get_memories()),
        "plan_stats": plan_stats,
        "gate_stats": gate_stats,
        **_collect_retrieval_evidence(result),
        **_score_block(run_dir, dataset),
    }
    _print_arm(label, result, summary)
    return summary


def run_classic(dataset, args, answerer) -> dict:
    """Classic arm via the plain AMv020Player (frozen pre-raid behaviour)."""
    player = AMv020Player({"context_budget": 2000}, answerer=answerer)
    result = ArenaRunner(_build_config(args, dataset, player)).run()
    run_dir = Path(args.output_dir) / "controlled" / result.run_id

    summary = {
        "arm": "classic",
        "gate_pool": None,
        "run_id": result.run_id,
        "run_dir": str(run_dir),
        "failures": len(result.failures),
        "plan_stats": None,
        "gate_stats": None,
        **_collect_retrieval_evidence(result),
        **_score_block(run_dir, dataset),
    }
    _print_arm("classic", result, summary)
    return summary


def cross_arm_agreement(arms: dict) -> dict:
    """Fraction of questions where an arm's first-repeat selection == classic's.

    Selections are read from the persisted traces.jsonl so the analysis is
    reproducible from the evidence locker alone.
    """
    def load_selections(run_dir: str) -> dict:
        text = (Path(run_dir) / "traces.jsonl").read_text(encoding="utf-8")
        parsed = json.loads(text)
        # traces.jsonl is "\n"-joined per-player dicts; with a single player
        # json.loads returns the dict itself, with several a list of dicts.
        records = parsed if isinstance(parsed, list) else [parsed]
        out = {}
        for pr in records:
            questions = pr.get("questions") or pr.get("question_results") or {}
            for qid, qr in questions.items():
                answers = qr.get("answers") or []
                if answers:
                    out[qid] = (answers[0].get("metadata") or {}).get("memory_ids") or []
        return out

    base_label = "classic"
    if base_label not in arms:
        return {}
    base = load_selections(arms[base_label]["run_dir"])
    agreement = {}
    for label, arm in arms.items():
        if label == base_label:
            continue
        other = load_selections(arm["run_dir"])
        pairs = same = 0
        for qid, base_ids in base.items():
            if qid in other:
                pairs += 1
                if base_ids == other[qid]:
                    same += 1
        agreement[f"{base_label}_vs_{label}"] = round(same / pairs, 4) if pairs else 0.0
    return agreement


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek Raid small-scale A/B E2E")
    parser.add_argument("--dry-run", action="store_true",
                        help="unit mode: no LLM (deterministic abstention answers)")
    parser.add_argument("--gate-pool", type=int, default=64)
    parser.add_argument("--answer-repeats", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--arms", default="classic,cache,gate",
                        help="comma-separated subset of classic/cache/gate")
    parser.add_argument("--output-dir", default="benchmark/results")
    args = parser.parse_args()

    dataset = build_dataset()
    global QUESTION_IDS
    QUESTION_IDS = [dataset.by_category(cat)[0].question_id for cat in ArenaCategory]
    print(f"DeepSeek Raid A/B small-scale E2E ({len(QUESTION_IDS)} questions, "
          f"dry-run={args.dry_run})")
    for qid in QUESTION_IDS:
        q = dataset.by_id(qid)
        assert q is not None
        print(f"  [{q.category.value}] {q.question}")

    if args.dry_run:
        answerer = None  # deterministic abstention (unit-test mode)
    else:
        from artificial_memory.research.benchmarks.players import get_shared_answerer
        answerer = get_shared_answerer()

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms: dict = {}
    for label in wanted:
        if label == "classic":
            arms[label] = run_classic(dataset, args, answerer)
        elif label == "cache":
            arms[label] = run_arm("cache", None, answerer, dataset, args)
        elif label == "gate":
            arms[label] = run_arm("gate", args.gate_pool, answerer, dataset, args)
        else:
            raise SystemExit(f"unknown arm: {label}")

    agreement = cross_arm_agreement(arms)
    if agreement:
        print("\n=== cross-arm selection agreement ===")
        for key, value in agreement.items():
            print(f"  {key}: {value}")

    summary = {
        "experiment": "deepseek-raid-ab-small",
        "timestamp": datetime.now().isoformat(),
        "question_ids": QUESTION_IDS,
        "answer_repeats": args.answer_repeats,
        "latency_repeats": args.latency_repeats,
        "dry_run": args.dry_run,
        "arms": arms,
        "cross_arm_selection_agreement": agreement,
    }
    out_dir = Path(args.output_dir) / "deepseek_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"summary_{stamp}.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nevidence written: {out_path}")


if __name__ == "__main__":
    main()

