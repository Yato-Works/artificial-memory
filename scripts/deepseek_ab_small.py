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
from contextlib import contextmanager
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

    def __init__(self, gate_pool, candidate_window, answerer):
        self._gate_pool = gate_pool
        self._candidate_window = candidate_window
        super().__init__(
            {"context_budget": 2000, "retrieval_strategy": "deepseek"},
            answerer=answerer,
        )

    def ingest(self, scenarios) -> None:
        super().ingest(scenarios)
        if self._runtime is not None:
            from artificial_memory.recall.deepseek_engine import DeepSeekRecallEngine

            engine = self._runtime.recall_engine
            assert isinstance(engine, DeepSeekRecallEngine)
            if self._gate_pool is not None:
                engine.gate = MeasuredGate(pool_size=self._gate_pool)
            if self._candidate_window is not None:
                # Post-construction window assignment: identical to the
                # constructor parameter, keeps the frozen facade untouched.
                engine.candidate_window = self._candidate_window


@contextmanager
def _nullcontext():
    """No-op context manager (kept local to avoid a version-dependent import)."""
    yield


@contextmanager
def _frozen_access_stats():
    """Freeze ``Memory.touch`` so the store ordering cannot drift.

    ``get_memories`` orders by ``updated_at DESC`` and recall touches the
    selected memories, so a *live* run re-orders its own candidate window
    (selected memories jump to the front).  A window sweep must vary the
    window alone, not the side effects of the previous question, hence this
    read-only mode.  The frozen engine is untouched: the patch lives only
    inside the audit context and is always restored.
    """
    from artificial_memory.core.models import Memory as _Memory

    original = _Memory.touch

    def _no_touch(self):  # pragma: no cover - trivial shim
        return None

    _Memory.touch = _no_touch
    try:
        yield
    finally:
        _Memory.touch = original


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


def run_arm(label: str, gate_pool, candidate_window, answerer, dataset, args) -> dict:
    """Run one deepseek arm (cache / cache+gate / cache+gate+window)."""
    player = StrategyPlayer(gate_pool, candidate_window, answerer=answerer)
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
        "candidate_window": candidate_window,
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
        "store_memory_count": len(list(player._runtime.store.get_memories(limit=10000))),
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


# ---------------------------------------------------------------------------
# A: Gate correctness audit (LLM-free ground-truth coverage probe)
# ---------------------------------------------------------------------------


def _evidence_recall(valid: list) -> dict:
    """Per-evidence-memory retention at each funnel stage.

    The binary "does this question still have any evidence" rate hides partial
    loss, which is exactly what a cheap pre-filter causes.  This metric weights
    every evidence memory equally: retained / present-in-store.
    """
    total = sum(r["evidence_count"] for r in valid)
    if not total:
        return {"store": 0.0, "candidates": 0.0, "gate": 0.0, "selected": 0.0,
                "evidence_total": 0}
    return {
        "evidence_total": total,
        "store": 1.0,
        "candidates": round(sum(r["stage_candidates"] for r in valid) / total, 4),
        "gate": round(sum(r["stage_gate"] for r in valid) / total, 4),
        "selected": round(sum(r["stage_selected"] for r in valid) / total, 4),
    }


def _gate_damage(valid: list) -> float:
    """Evidence lost *to the gate* as a fraction of the evidence it received."""
    received = sum(r["stage_candidates"] for r in valid)
    kept = sum(r["stage_gate"] for r in valid)
    return round((received - kept) / received, 4) if received else 0.0


def _probe_questions(engine, gate, runtime, store_all, questions,
                     candidate_window, recall_level, gate_pool) -> list:
    """Per-question stage-wise GT funnel probe (see ``gate_gt_audit``)."""
    from artificial_memory.core.models import RecallLevel

    import asyncio

    topic_id = runtime.conversation_manager.get_or_create_topic("Arena").id
    rows = []
    for q in questions:
        gt = q.ground_truth
        if not gt:
            rows.append({
                "question_id": q.question_id, "category": q.category.value,
                "gt_groups": 0, "note": "no ground truth (abstention)",
            })
            continue

        # Evidence set: store memories whose content carries a GT term.
        evidence = [
            m for m in store_all
            if any(term.lower() in m.content.lower() for group in gt for term in group)
        ]
        evidence_ids = {m.id for m in evidence}

        result = asyncio.run(runtime.recall(
            q.question, topic="Arena", level=recall_level, max_tokens=2000
        ))
        # Raw pool: fetch WITHOUT the gate (temporarily detached) so the
        # candidates stage measures the window, not the gate; the gate stage
        # then applies the gate explicitly to the identical pool.
        engine.gate = None
        pool = engine._get_candidates(
            q.question, topic_id, RecallLevel(recall_level)
        )
        # Frozen-path pool for the same query/level: the "reception desk" size
        # the unmodified engine would hand to ranking (the recall ceiling).
        engine.candidate_window = None
        frozen_pool = engine._get_candidates(
            q.question, topic_id, RecallLevel(recall_level)
        )
        engine.candidate_window = candidate_window
        engine.gate = gate
        gated = gate.filter(q.question, pool) if len(pool) > gate_pool else pool

        pool_ids = {m.id for m in pool}
        gate_ids = {m.id for m in gated}
        sel_ids = {m.identity.memory_id for m in result.memories}

        rows.append({
            "question_id": q.question_id,
            "category": q.category.value,
            "gt_groups": len(gt),
            "evidence_count": len(evidence),
            "stage_store": len(evidence),
            "stage_candidates": len(evidence_ids & pool_ids),
            "stage_gate": len(evidence_ids & gate_ids),
            "stage_selected": len(evidence_ids & sel_ids),
            "pool_in": len(pool),
            "pool_out": len(gated),
            "frozen_pool": len(frozen_pool),
        })
    return rows


def gate_gt_audit(dataset, gate_pool: int, sample: int,
                  candidate_window: int | None = None,
                  recall_level: int = 2,
                  static_store: bool = True) -> dict:
    """Stage-wise GT funnel: store -> candidates -> gate -> selected.

    Deterministic and LLM-free.  For each sampled question we locate every
    store memory whose content contains a ground-truth term (the *evidence
    set*) and measure where in the pipeline it is lost:

        stage_store      evidence present in the durable store
        stage_candidates evidence in the engine's candidate pool
        stage_gate       evidence survives the HierarchicalGate
        stage_selected   evidence in the final budgeted selection

    ``candidate_window`` widens the raid engine's candidate fetch (None =
    frozen per-level limits), turning the funnel into a window sweep.
    ``recall_level`` must match the level the benchmark player actually uses
    (level 2 = EPISODE) or the funnel measures the wrong reception desk.
    """
    from artificial_memory.recall.deepseek_engine import DeepSeekRecallEngine
    from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

    import asyncio

    runtime = ArtificialMemoryRuntime(RuntimeConfig(
        database_path=":memory:",
        memory_files_path=None,
        vector_index_path=None,
        retrieval_strategy="deepseek",
    ))

    # Ingest the shared history exactly like the arena player does.
    for scenario in dataset.scenarios:
        for turn in scenario.turns:
            if turn.speaker == "user":
                asyncio.run(runtime.remember(turn.content, topic="Arena"))

    topic_obj = runtime.conversation_manager.get_or_create_topic("Arena")
    engine = runtime.recall_engine
    assert isinstance(engine, DeepSeekRecallEngine)
    engine.candidate_window = candidate_window
    gate = MeasuredGate(pool_size=gate_pool)
    engine.gate = gate

    # Full store snapshot (explicit limit: the interface default is 100).
    store_all = runtime.store.get_memories(topic_id=topic_obj.id, limit=10000)

    questions = []
    for cat in ArenaCategory:
        questions.extend(dataset.by_category(cat)[:sample])

    rows = []
    questions_context = _frozen_access_stats() if static_store else _nullcontext()
    with questions_context:
        rows = _probe_questions(
            engine, gate, runtime, store_all, questions, candidate_window,
            recall_level, gate_pool,
        )

    valid = [r for r in rows if r.get("gt_groups")]
    n = len(valid)
    audit = {
        "gate_pool": gate_pool,
        "candidate_window": candidate_window,
        "recall_level": recall_level,
        "static_store": static_store,
        "store_memory_count": len(store_all),
        # Frozen-path candidate pool size (the unmodified reception desk):
        # this is the hard ceiling on any recall the engine can achieve.
        "frozen_pool_size": (
            round(statistics.mean([r["frozen_pool"] for r in valid]), 2) if n else 0.0
        ),
        "questions": len(rows),
        "funnel": {
            "stage_store": round(sum(r["stage_store"] > 0 for r in valid) / n, 4) if n else 0.0,
            "stage_candidates": round(sum(r["stage_candidates"] > 0 for r in valid) / n, 4) if n else 0.0,
            "stage_gate": round(sum(r["stage_gate"] > 0 for r in valid) / n, 4) if n else 0.0,
            "stage_selected": round(sum(r["stage_selected"] > 0 for r in valid) / n, 4) if n else 0.0,
        },
        "gate_narrowing": {
            "in": sum(r["pool_in"] for r in valid),
            "out": sum(r["pool_out"] for r in valid),
        },
        "evidence_recall": _evidence_recall(valid),
        "gate_damage": _gate_damage(valid),
        "rows": rows,
    }

    print(f"\n=== GATE GT FUNNEL AUDIT (pool={gate_pool}, window={candidate_window}, "
          f"level={recall_level}, static_store={static_store}, {len(rows)} questions, "
          f"store={len(store_all)} memories) ===")
    f = audit["funnel"]
    print(f"  frozen pool (reception desk): {audit['frozen_pool_size']} memories "
          f"of {len(store_all)} "
          f"({audit['frozen_pool_size'] / len(store_all):.1%})")
    print(f"  evidence present:  store={f['stage_store']:.0%}  "
          f"candidates={f['stage_candidates']:.0%}  "
          f"after-gate={f['stage_gate']:.0%}  "
          f"selected={f['stage_selected']:.0%}")
    print(f"  gate narrowing:    {audit['gate_narrowing']['in']} -> "
          f"{audit['gate_narrowing']['out']}")
    er = audit["evidence_recall"]
    print(f"  evidence recall (per-memory, total={er['evidence_total']}): "
          f"candidates={er['candidates']:.0%}  after-gate={er['gate']:.0%}  "
          f"selected={er['selected']:.0%}")
    print(f"  gate damage (evidence lost ON THE GATE): {audit['gate_damage']:.1%}")
    for r in rows:
        if not r.get("gt_groups"):
            print(f"  [---] {r['question_id']:<28} (no GT)")
            continue
        funnel = f"{r['stage_store']}/{r['evidence_count']} -> " \
                 f"{r['stage_candidates']}/{r['evidence_count']} -> " \
                 f"{r['stage_gate']}/{r['evidence_count']} -> " \
                 f"{r['stage_selected']}/{r['evidence_count']}"
        lost = (
            "WRITE" if r["stage_store"] == 0 else
            "POOL" if r["stage_candidates"] == 0 else
            "GATE" if r["stage_gate"] < r["stage_candidates"] else
            "SELECT" if r["stage_selected"] == 0 else "OK"
        )
        print(f"  [{lost:<6}] {r['question_id']:<28} "
              f"store->cand->gate->sel = {funnel}  "
              f"(pool {r['pool_in']}->{r['pool_out']})")
    return audit


def window_sweep(dataset, gate_pool: int, sample: int, windows: list, args) -> dict:
    """LLM-free candidate-window sweep: the Phase 8.5 decision table.

    For each window W in ``windows`` (None = frozen recent-50) run the GT
    funnel and collect one comparable row — the answer to "does widening the
    window recover recall, and does the gate keep it cheap?".
    """
    sweep = {"gate_pool": gate_pool, "recall_level": args.audit_level,
             "static_store": not args.touch_feedback, "rows": []}
    print(f"\n=== CANDIDATE WINDOW SWEEP (pool={gate_pool}, "
          f"level={args.audit_level}, static_store={not args.touch_feedback}) ===")
    print(f"  {'window':>7} | {'frozen':>6} | {'pool':>7} | {'after gate':>10} | "
          f"{'selected':>8} | {'gate in->out':>13}")
    for window in windows:
        audit = gate_gt_audit(dataset, gate_pool, sample,
                              candidate_window=window,
                              recall_level=args.audit_level,
                              static_store=not args.touch_feedback)
        f = audit["funnel"]
        row = {
            "window": window,
            "frozen_pool_size": audit["frozen_pool_size"],
            "candidates_pct": f["stage_candidates"],
            "after_gate_pct": f["stage_gate"],
            "selected_pct": f["stage_selected"],
            "gate_in": audit["gate_narrowing"]["in"],
            "gate_out": audit["gate_narrowing"]["out"],
        }
        sweep["rows"].append(row)
        print(f"  {str(window):>7} | {row['frozen_pool_size']:>6} | "
              f"{f['stage_candidates']:>6.0%} | "
              f"{f['stage_gate']:>9.0%} | {f['stage_selected']:>7.0%} | "
              f"{row['gate_in']:>6}->{row['gate_out']}")
    return sweep


def gate_sweep(dataset, sample: int, pool_sizes: list, args) -> dict:
    """LLM-free gate-pool sweep on a WIDE candidate window.

    Second half of the A1 decision table: with the reception desk widened so
    the evidence actually reaches the gate, how much can the gate narrow before
    it starts destroying evidence?  The answer is the HSI operating curve
    (pool reduction vs evidence recall) for this dataset.
    """
    window = args.wide_window
    sweep = {
        "window": window,
        "recall_level": args.audit_level,
        "static_store": not args.touch_feedback,
        "rows": [],
    }
    print(f"\n=== GATE POOL SWEEP (window={window}, level={args.audit_level}) ===")
    print(f"  {'pool':>6} | {'gate in':>8} | {'gate out':>8} | {'reduction':>9} | "
          f"{'ev recall':>9} | {'ev lost':>8} | {'selected':>8}")
    for pool in pool_sizes:
        audit = gate_gt_audit(dataset, pool, sample,
                              candidate_window=window,
                              recall_level=args.audit_level,
                              static_store=not args.touch_feedback)
        er = audit["evidence_recall"]
        g = audit["gate_narrowing"]
        reduction = 1 - (g["out"] / g["in"]) if g["in"] else 0.0
        row = {
            "gate_pool": pool,
            "gate_in": g["in"],
            "gate_out": g["out"],
            "pool_reduction": round(reduction, 4),
            "evidence_recall_candidates": er["candidates"],
            "evidence_recall_gate": er["gate"],
            "evidence_recall_selected": er["selected"],
            "gate_damage": audit["gate_damage"],
        }
        sweep["rows"].append(row)
        print(f"  {pool:>6} | {g['in']:>8} | {g['out']:>8} | {reduction:>8.0%} | "
              f"{er['gate']:>8.0%} | {audit['gate_damage']:>7.1%} | "
              f"{er['selected']:>7.0%}")
    return sweep

def grid_sweep(dataset, sample: int, windows: list, pools: list, args) -> dict:
    """Joint window × gate-pool grid (Phase 8.6 operating-point search).

    LLM-free.  Ingests the shared history **once**, then probes every
    ``(window, gate_pool)`` cell so the whole grid shares one store snapshot.
    For each cell we record:

        evidence_recall_gate : GT-evidence surviving window + gate (quality)
        gate_out_per_q       : candidates handed to ranking (cost proxy)
        pool_reduction       : gate traffic compression

    The deliverable is the Pareto frontier over (recall ↑, cost ↓) — the
    "keep coverage, cut tokens" operating points the 3,000-call benchmark
    should freeze.
    """
    from artificial_memory.recall.deepseek_engine import DeepSeekRecallEngine
    from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

    import asyncio

    runtime = ArtificialMemoryRuntime(RuntimeConfig(
        database_path=":memory:",
        memory_files_path=None,
        vector_index_path=None,
        retrieval_strategy="deepseek",
    ))
    for scenario in dataset.scenarios:
        for turn in scenario.turns:
            if turn.speaker == "user":
                asyncio.run(runtime.remember(turn.content, topic="Arena"))

    topic_obj = runtime.conversation_manager.get_or_create_topic("Arena")
    engine = runtime.recall_engine
    assert isinstance(engine, DeepSeekRecallEngine)
    store_all = runtime.store.get_memories(topic_id=topic_obj.id, limit=10000)

    questions = []
    for cat in ArenaCategory:
        questions.extend(dataset.by_category(cat)[:sample])

    static = not args.touch_feedback
    grid = {
        "recall_level": args.audit_level,
        "static_store": static,
        "store_memory_count": len(store_all),
        "sample_per_category": sample,
        "cells": [],
    }
    print(f"\n=== WINDOW x GATE GRID (level={args.audit_level}, "
          f"static_store={static}, {len(questions)} questions, "
          f"store={len(store_all)}) ===")
    print(f"  {'window':>7} {'pool':>6} | {'ev recall':>9} | "
          f"{'sel recall':>9} | {'gate out/q':>10} | {'reduction':>9} | damage")
    for window in windows:
        for pool in pools:
            gate = MeasuredGate(pool_size=pool)
            engine.candidate_window = window
            engine.gate = gate
            # The plan cache is shared across cells (one runtime): without
            # clearing it, every later cell would REUSE the first cell's
            # selection and the window/pool change would never reach the
            # selected stage.  Each cell must probe a fresh plan.
            engine.plan_cache.clear()
            ctx = _frozen_access_stats() if static else _nullcontext()
            with ctx:
                rows = _probe_questions(
                    engine, gate, runtime, store_all, questions,
                    window, args.audit_level, pool,
                )
            valid = [r for r in rows if r.get("gt_groups")]
            er = _evidence_recall(valid)
            damage = _gate_damage(valid)
            g = {"in": sum(r["pool_in"] for r in valid),
                 "out": sum(r["pool_out"] for r in valid)}
            n = len(valid)
            out_per_q = g["out"] / n if n else 0.0
            reduction = 1 - (g["out"] / g["in"]) if g["in"] else 0.0
            cell = {
                "window": window,
                "gate_pool": pool,
                "evidence_recall_candidates": er["candidates"],
                "evidence_recall_gate": er["gate"],
                "evidence_recall_selected": er["selected"],
                "gate_in": g["in"],
                "gate_out": g["out"],
                "gate_out_per_question": round(out_per_q, 2),
                "pool_reduction": round(reduction, 4),
                "gate_damage": damage,
            }
            grid["cells"].append(cell)
            print(f"  {str(window):>7} {pool:>6} | {er['gate']:>8.0%} | "
                  f"{er['selected']:>8.0%} | {out_per_q:>10.1f} | "
                  f"{reduction:>8.0%} | {damage:>5.1%}")

    # Pareto frontier over (evidence recall at gate ↑, gate out/question ↓).
    def _dominates(a, b):
        return (a["evidence_recall_gate"] >= b["evidence_recall_gate"]
                and a["gate_out_per_question"] <= b["gate_out_per_question"]
                and (a["evidence_recall_gate"] > b["evidence_recall_gate"]
                     or a["gate_out_per_question"] < b["gate_out_per_question"]))

    frontier = [
        c for c in grid["cells"]
        if not any(_dominates(o, c) for o in grid["cells"] if o is not c)
    ]
    frontier.sort(key=lambda c: (-c["evidence_recall_gate"],
                                 c["gate_out_per_question"]))
    grid["pareto_frontier"] = frontier
    print("\n  --- Pareto frontier (recall ^, cost v) ---")
    for c in frontier:
        print(f"    window={c['window']:>4} pool={c['gate_pool']:>3}  "
              f"recall={c['evidence_recall_gate']:.0%}  "
              f"out/q={c['gate_out_per_question']:.1f}  "
              f"reduction={c['pool_reduction']:.0%}")
    return grid

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek Raid small-scale A/B E2E")
    parser.add_argument("--dry-run", action="store_true",
                        help="unit mode: no LLM (deterministic abstention answers)")
    parser.add_argument("--gate-pool", type=int, default=64)
    parser.add_argument("--wide-window", type=int, default=450,
                        help="candidate window for the gate_wide arm")
    parser.add_argument("--window-sweep", default=None,
                        help="comma list of windows (e.g. 50,100,250,450); "
                             "None is always included as the frozen baseline")
    parser.add_argument("--gate-sweep", default=None,
                        help="comma list of gate pool sizes (e.g. 64,128,250,440); "
                             "swept on --wide-window to expose gate damage")
    parser.add_argument("--grid-sweep", default=None,
                        help="comma list of windows (e.g. 50,100,200,300,440); "
                             "swept jointly with --grid-pools over the full grid")
    parser.add_argument("--grid-pools", default="10,20,32,64,100",
                        help="comma list of gate pool sizes for --grid-sweep")
    parser.add_argument("--answer-repeats", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--arms", default="classic,cache,gate,gate_wide",
                        help="comma-separated subset of classic/cache/gate/gate_wide")
    parser.add_argument("--gate-audit", action="store_true",
                        help="run the LLM-free gate GT-coverage audit and exit")
    parser.add_argument("--audit-sample", type=int, default=3,
                        help="questions per category for the gate audit")
    parser.add_argument("--audit-level", type=int, default=2,
                        help="recall level the funnel audit measures; must match "
                             "the player's level (2 = EPISODE, the S4 default)")
    parser.add_argument("--touch-feedback", action="store_true",
                        help="keep the live recall touch() side effects; by "
                             "default the audit freezes access stats so the "
                             "candidate window is the only varying factor")
    parser.add_argument("--output-dir", default="benchmark/results")
    args = parser.parse_args()

    dataset = build_dataset()
    global QUESTION_IDS

    out_dir = Path(args.output_dir) / "deepseek_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.window_sweep:
        windows = [None] + [int(w) for w in args.window_sweep.split(",") if w.strip()]
        sweep = window_sweep(dataset, args.gate_pool, args.audit_sample, windows, args)
        out_path = out_dir / f"window_sweep_{stamp}.json"
        out_path.write_text(
            json.dumps(sweep, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nevidence written: {out_path}")
        return

    if args.grid_sweep:
        windows = [None] + [int(w) for w in args.grid_sweep.split(",") if w.strip()]
        pools = [int(p) for p in args.grid_pools.split(",") if p.strip()]
        grid = grid_sweep(dataset, args.audit_sample, windows, pools, args)
        out_path = out_dir / f"grid_sweep_{stamp}.json"
        out_path.write_text(
            json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nevidence written: {out_path}")
        return

    if args.gate_sweep:
        pools = [int(p) for p in args.gate_sweep.split(",") if p.strip()]
        sweep = gate_sweep(dataset, args.audit_sample, pools, args)
        out_path = out_dir / f"gate_sweep_{stamp}.json"
        out_path.write_text(
            json.dumps(sweep, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nevidence written: {out_path}")
        return

    if args.gate_audit:
        audit = gate_gt_audit(dataset, args.gate_pool, args.audit_sample,
                              recall_level=args.audit_level,
                              static_store=not args.touch_feedback)
        out_path = out_dir / f"gate_audit_{stamp}.json"
        out_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nevidence written: {out_path}")
        return

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
            arms[label] = run_arm("cache", None, None, answerer, dataset, args)
        elif label == "gate":
            arms[label] = run_arm("gate", args.gate_pool, None, answerer, dataset, args)
        elif label == "gate_wide":
            arms[label] = run_arm(
                "gate_wide", args.gate_pool, args.wide_window, answerer, dataset, args
            )
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

