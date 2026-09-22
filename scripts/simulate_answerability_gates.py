"""Simulate deterministic answerability gates on a completed LoCoMo run.

LoCoMo's adversarial questions (446 / 1986 = 22.5%) are unanswerable by design:
the question misattributes a subject ("Where did *Oscar* hide his bone?" when the
pet is Oliver) or asserts an event the evidence explicitly denies ("Nope, never
been to something like that").  The frozen scorer credits a refusal whenever the
question has no ``answer`` field, so the whole category is decided by whether the
system *knows it cannot answer*.

This harness measures candidate deterministic gates offline over the saved
answers of a run, re-scoring with the frozen scorer - no LLM calls:

  G1  a question entity is absent from the whole conversation memory -> refuse
  G2  G1 or an explicit denial in the question's evidence turns      -> refuse
  G3  G2 or no evidence speaker matches the question's subject       -> refuse

Usage: python scripts/simulate_answerability_gates.py [run_dir]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import CAT, score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(sys.argv[1] if len(sys.argv) > 1
           else "benchmark_results/locomo10_runs/selection_v3_llm")

REFUSAL_MARKERS = ["i don't know", "i dont know", "not mentioned", "unknown",
                   "no information", "unclear", "cannot determine", "not specified"]

QUESTION_STARTERS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "did",
    "does", "do", "is", "are", "was", "were", "has", "have", "had", "can", "could",
    "would", "should", "will", "the", "a", "an", "in", "on", "at", "of", "for", "to",
    "and", "or", "if", "that", "this", "these", "those", "there", "it", "its", "any",
    "some", "most", "many", "much", "more", "all", "both", "after", "before", "about",
    "yes", "no", "nope", "sure", "well", "oh", "also", "how", "time", "day", "year",
}

DENIAL_PATTERNS = [
    r"\bnope\b", r"\bnever\b", r"\bdidn't\b", r"\bdid not\b", r"\bhasn't\b",
    r"\bhaven't\b", r"\bhave not\b", r"\bno idea\b", r"\bdon't think\b",
    r"\bnot really\b", r"\bnot sure\b",
]


def question_entities(question: str) -> set[str]:
    """Capitalised tokens in the question that plausibly name a memory entity."""
    ents = set()
    for tok in re.findall(r"\b[A-Z][a-zA-Z'\-]+", question):
        low = tok.lower()
        if low in QUESTION_STARTERS:
            continue


def collect(adapter: LoCoMoAdapter, convs: list[int]) -> tuple[list[dict], dict[str, bool]]:
    rows: list[dict] = []
    gt_empty: dict[str, bool] = {}
    for conv in convs:
        turns, questions, _ = adapter.load_conversation(conv_idx=conv)
        run = json.load(open(RUN / f"conv_{conv}_results.json", encoding="utf-8"))["results"]
        by_id = {r["question_id"]: r for r in run}
        corpus_tokens = set(re.findall(r"\b[a-z0-9']+\b",
                                       " ".join(t.text for t in turns).lower()))
        turn_map = {t.dia_id: t for t in turns}
        for q in questions:
            r = by_id.get(q.question_id)
            if r is None:
                continue
            gt_empty[q.question_id] = (str(q.ground_truth).strip() == "")
            ans = str(r["predicted_answer"])
            al = ans.lower()
            ents = question_entities(q.question)
            absent = {e for e in ents if e not in corpus_tokens}
            g1 = bool(absent)
            ev = [turn_map[e] for e in q.evidence_ids if e in turn_map]
            ev_text = " ".join(t.text for t in ev).lower()
            g2 = g1 or any(re.search(p, ev_text) for p in DENIAL_PATTERNS)
            speakers = {t.speaker.lower() for t in ev}
            g3 = g2 or (bool(ents) and bool(speakers) and not (ents & speakers))
            rows.append({
                "qid": q.question_id,
                "cat": q.category,
                "ok": score(q.category, str(q.ground_truth), ans),
                "g1": g1, "g2": g2, "g3": g3,
                "refusal": any(m in al for m in REFUSAL_MARKERS) or len(al.strip()) <= 3,
            })
        print(f"  conv {conv} scanned ({len(rows)} rows)", flush=True)
    return rows, gt_empty


def simulate(rows: list[dict], gt_empty: dict[str, bool], gate: str) -> tuple[float, int, int]:
    """Return (accuracy, gained, lost) when the gate turns answers into refusals."""
    ok = gain = loss = 0
    for r in rows:
        if r["refusal"] or not r[gate]:
            ok += r["ok"]
            continue
        if gt_empty.get(r["qid", ""], False):
            ok += 1
            if not r["ok"]:
                gain += 1
        else:
            if r["ok"]:
                loss += 1
    return ok / len(rows) * 100, gain, loss


def main() -> None:
    convs = sorted(int(p.stem.split("_")[1]) for p in RUN.glob("conv_*_results.json"))
    if not convs:
        print(f"no results in {RUN}")
        return
    print(f"run: {RUN}  ({len(convs)} conversations)")
    adapter = LoCoMoAdapter()
    rows, gt_empty = collect(adapter, convs)
    n = len(rows)
    base = sum(r["ok"] for r in rows) / n * 100
    refs = [r for r in rows if r["refusal"]]
    ref_ok = sum(1 for r in refs if r["ok"])
    print(f"\nrows={n}  rescored accuracy={base:.2f}%")
    print(f"existing refusals: {len(refs)} correct={ref_ok} ({ref_ok / len(refs) * 100:.1f}%)")

    print(f"\n{'gate':<8}{'eligible':>9}{'gain':>7}{'loss':>7}{'acc':>9}{'delta':>9}")
    for gate in ("g1", "g2", "g3"):
        elig = sum(1 for r in rows if not r["refusal"] and r[gate])
        acc, gain, loss = simulate(rows, gt_empty, gate)
        print(f"{gate:<8}{elig:>9}{gain:>7}{loss:>7}{acc:>8.2f}%{acc - base:>+8.2f}pp")

    print("\nper-category (rescored -> g2 gate):")
    print(f"{'category':<14}{'n':>6}{'base':>8}{'g1':>8}{'g2':>8}{'g3':>8}")
    cats: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cats[CAT.get(r["cat"], str(r["cat"]))].append(r)
    for c, rs in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        cells = []
        for gate in ("g1", "g2", "g3"):
            gk = s = 0
            for r in rs:
                if r["refusal"] or not r[gate]:
                    gk += r["ok"]
                elif gt_empty.get(r["qid"], False):
                    gk += 1
            cells.append(gk / len(rs) * 100)
        print(f"{c:<14}{len(rs):>6}{sum(r['ok'] for r in rs) / len(rs) * 100:>7.1f}%"
              + "".join(f"{x:>7.1f}%" for x in cells))

        ents.add(low)
    return ents
