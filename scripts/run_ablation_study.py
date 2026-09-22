"""Scientific Ablation Study for AM Apex Universal Evidence Retrieval (Phase X.1).

Evaluates the multi-dimensional evidence scoring formula:
    S(e, q) = w_l S_lexical + w_s S_semantic + w_e S_entity + w_a S_actor + w_t S_temporal + w_p S_provenance

Measures:
1. Memory Oracle Recall across 7 configurations on all 199 LoCoMo-10 questions.
2. Answer Accuracy on the best configurations using phi4-mini:3.8b.
3. Token efficiency (MSC Tokens/Q).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 70)
print("AM APEX UNIVERSAL EVIDENCE RETRIEVAL: SCIENTIFIC ABLATION STUDY")
print("=" * 70)

adapter = LoCoMoAdapter()
turns, questions, ir_records = adapter.load_conversation(conv_idx=0)
answerer = OllamaAnswerer()

print(f"Loaded {len(turns)} turns, {len(questions)} questions, {len(ir_records)} IR records.")

# Define the 7 ablation configurations
configurations = [
    {
        "name": "1. Baseline MSC (Original)",
        "weights": None,  # Original prior
        "run_llm": False,
    },
    {
        "name": "2. + Lexical Only",
        "weights": EvidenceScoreWeights(w_lexical=1.0, w_semantic=0.0, w_entity=0.0, w_actor=0.0, w_temporal=0.0, w_provenance=0.0),
        "run_llm": False,
    },
    {
        "name": "3. + Semantic Only",
        "weights": EvidenceScoreWeights(w_lexical=0.0, w_semantic=1.0, w_entity=0.0, w_actor=0.0, w_temporal=0.0, w_provenance=0.0),
        "run_llm": False,
    },
    {
        "name": "4. + Entity Only",
        "weights": EvidenceScoreWeights(w_lexical=0.0, w_semantic=0.0, w_entity=1.0, w_actor=0.0, w_temporal=0.0, w_provenance=0.0),
        "run_llm": False,
    },
    {
        "name": "5. + Actor Only",
        "weights": EvidenceScoreWeights(w_lexical=0.0, w_semantic=0.0, w_entity=0.0, w_actor=1.0, w_temporal=0.0, w_provenance=0.0),
        "run_llm": False,
    },
    {
        "name": "6. + Lexical + Entity + Actor",
        "weights": EvidenceScoreWeights(w_lexical=1.0, w_semantic=0.0, w_entity=1.5, w_actor=2.0, w_temporal=0.0, w_provenance=0.0),
        "run_llm": False,
    },
    {
        "name": "7. Full Universal Evidence Scorer",
        "weights": EvidenceScoreWeights(w_lexical=1.0, w_semantic=1.0, w_entity=1.5, w_actor=2.0, w_temporal=1.5, w_provenance=0.5),
        "run_llm": True,  # Full generation run
    },
]

ablation_results = []

for cfg in configurations:
    cfg_name = cfg["name"]
    weights = cfg["weights"]
    run_llm = cfg["run_llm"]

    t0 = time.perf_counter()
    oracle_hits = 0
    correct_answers = 0
    total_tokens = 0
    latencies = []

    for i, q in enumerate(questions):
        # 1. Compile MSC
        pcc = adapter.compiler.compile(q.question, ir_records, weights=weights)
        total_tokens += pcc.token_cost

        # 2. Oracle Recall Check
        has_oracle = False
        if q.evidence_ids:
            has_oracle = any(ev_id in pcc.context_text for ev_id in q.evidence_ids)
        else:
            has_oracle = True
        if has_oracle:
            oracle_hits += 1

        # 3. Answer Generation (if enabled)
        if run_llm:
            t_llm0 = time.perf_counter()
            if pcc.is_abstention:
                ans_text = "I don't know."
            else:
                ans = answerer.answer(q.question, pcc.context_text)
                ans_text = ans.text
            lat = (time.perf_counter() - t_llm0) * 1000
            latencies.append(lat)

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
                correct_answers += 1

    elapsed = time.perf_counter() - t0
    n = len(questions)
    oracle_recall_pct = (oracle_hits / n) * 100
    mean_tokens = total_tokens / n
    acc_pct = (correct_answers / n) * 100 if run_llm else None
    mean_lat = sum(latencies) / len(latencies) if latencies else None

    res_item = {
        "config": cfg_name,
        "oracle_recall": oracle_recall_pct,
        "accuracy": acc_pct,
        "mean_tokens": mean_tokens,
        "elapsed_sec": elapsed,
        "mean_lat_ms": mean_lat,
    }
    ablation_results.append(res_item)

    acc_str = f"Acc: {acc_pct:.1f}% | " if acc_pct is not None else ""
    lat_str = f"Lat: {mean_lat:.1f}ms | " if mean_lat is not None else ""
    print(f"[{cfg_name}] -> Oracle Recall: {oracle_recall_pct:5.1f}% | {acc_str}Tokens/Q: {mean_tokens:4.1f} | {lat_str}Time: {elapsed:.2f}s")

print("\n" + "=" * 70)
print("FINAL ABLATION STUDY SUMMARY (LoCoMo-10, 199 Questions):")
print(f"{'Configuration':<35} | {'Oracle Recall':<14} | {'Accuracy':<10} | {'Tokens/Q':<10}")
print("-" * 75)
for r in ablation_results:
    acc_val = f"{r['accuracy']:.1f}%" if r['accuracy'] is not None else "N/A"
    print(f"{r['config']:<35} | {r['oracle_recall']:>12.1f}% | {acc_val:>8} | {r['mean_tokens']:>8.1f}")
print("=" * 75)
