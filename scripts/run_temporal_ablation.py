"""Scientific Temporal Operator Ablation Study for AM Apex (Phase X.3).

Measures the exact causal contribution (Delta Accuracy) of isolated temporal operators:
    Config A: No Temporal Grounding (X.2 Baseline)
    Config B: + Yesterday / Tomorrow Only
    Config C: + Last Week / Last Weekend Only
    Config D: + Next Month / This Month Only
    Config E: + Two Days Ago / X Days Ago Only
    Config F: + Relative Weekdays Only (Last Friday, Last Tuesday, etc.)
    Config G: Full Universal Temporal Normalizer

Holds all other variables (retrieval evidence, context size, model) strictly constant.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 70)
print("AM APEX PHASE X.3: TEMPORAL OPERATOR SCIENTIFIC ABLATION STUDY")
print("=" * 70)

adapter = LoCoMoAdapter()
turns, questions, ir_records = adapter.load_conversation(conv_idx=0)
answerer = OllamaAnswerer()
weights = EvidenceScoreWeights()

# 50-question representative evaluation set
eval_questions = questions[:50]
print(f"Evaluating {len(eval_questions)} questions across 7 isolated temporal configurations...\n")

configs = [
    {
        "name": "Config A: No Grounding (X.2 Baseline)",
        "rules": set(),
    },
    {
        "name": "Config B: + Yesterday / Tomorrow Only",
        "rules": {"yesterday"},
    },
    {
        "name": "Config C: + Last Week / Weekend Only",
        "rules": {"last_week"},
    },
    {
        "name": "Config D: + Next/This Month Only",
        "rules": {"month"},
    },
    {
        "name": "Config E: + Two Days Ago Only",
        "rules": {"days_ago"},
    },
    {
        "name": "Config F: + Relative Weekdays Only",
        "rules": {"weekdays"},
    },
    {
        "name": "Config G: Full Temporal Normalizer",
        "rules": None,  # All enabled
    },
]

baseline_passes: set[int] = set()
ablation_records = []

for cfg_idx, cfg in enumerate(configs):
    cfg_name = cfg["name"]
    rules = cfg["rules"]

    t0 = time.perf_counter()
    passed_qids: set[int] = set()
    total_tokens = 0

    for i, q in enumerate(eval_questions):
        pcc = adapter.compiler.compile(
            q.question,
            ir_records,
            weights=weights,
            enabled_temporal_rules=rules,
        )
        total_tokens += pcc.token_cost

        # Answer Generation
        if pcc.is_abstention:
            ans_text = "I don't know."
        else:
            ans_text = answerer.answer(q.question, pcc.context_text).text

        # Scorer
        gt_lower = str(q.ground_truth).lower().strip()
        ans_lower = ans_text.lower().strip()
        is_correct = False
        if not gt_lower:
            is_correct = any(w in ans_lower for w in ["i don't know", "not mentioned", "unknown", "unclear", "no information"])
        elif gt_lower in ans_lower or ans_lower in gt_lower:
            is_correct = True
        else:
            gt_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", gt_lower) if len(w) > 2)
            ans_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", ans_lower) if len(w) > 2)
            if gt_words and ans_words:
                overlap = len(gt_words & ans_words)
                if overlap / len(gt_words) >= 0.5:
                    is_correct = True
                elif len(gt_words) <= 2 and overlap >= 1:
                    is_correct = True

        if is_correct:
            passed_qids.add(i)

    elapsed = time.perf_counter() - t0
    acc = (len(passed_qids) / len(eval_questions)) * 100
    mean_tok = total_tokens / len(eval_questions)

    if cfg_idx == 0:
        baseline_passes = passed_qids
        delta_q = 0
    else:
        unlocked = passed_qids - baseline_passes
        delta_q = len(unlocked)

    rec = {
        "name": cfg_name,
        "acc": acc,
        "passes": len(passed_qids),
        "unlocked": delta_q,
        "tokens": mean_tok,
        "time": elapsed,
        "passed_ids": list(passed_qids),
    }
    ablation_records.append(rec)

    delta_str = f"(+{delta_q} questions)" if delta_q > 0 else ""
    print(f"[{cfg_name}] -> Acc: {acc:5.1f}% ({len(passed_qids)}/50) {delta_str:<15} | Tokens/Q: {mean_tok:5.1f} | Time: {elapsed:.1f}s")

print("\n" + "=" * 70)
print("PHASE X.3 CAUSAL TEMPORAL ABLATION MATRIX:")
print(f"{'Configuration':<40} | {'Accuracy':<10} | {'Delta Acc':<11} | {'Tokens/Q':<10}")
print("-" * 75)
base_acc = ablation_records[0]["acc"]
for r in ablation_records:
    d_acc = r["acc"] - base_acc
    d_str = f"+{d_acc:4.1f}%" if d_acc > 0 else f"{d_acc:4.1f}%"
    print(f"{r['name']:<40} | {r['acc']:>8.1f}% | {d_str:>9} | {r['tokens']:>8.1f}")
print("=" * 75)
