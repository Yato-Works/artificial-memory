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
print("AM APEX EVALUATION: LoCoMo-10 Conversation 1 (conv-30, 105 Questions)")
print("Speakers: Gina & Jon")
print("=" * 70)

adapter = LoCoMoAdapter()
turns, questions, ir_records = adapter.load_conversation(conv_idx=1)
answerer = OllamaAnswerer()
weights = EvidenceScoreWeights()

print(f"Loaded {len(turns)} turns, {len(questions)} questions, {len(ir_records)} IR records.\n")

results = []
t_start = time.perf_counter()

for i, q in enumerate(questions):
    res = adapter.evaluate_question(
        q,
        turns,
        ir_records,
        answerer,
        weights=weights,
    )
    results.append(res)

    status = "PASS" if res.is_correct else "FAIL"
    ora_status = "O" if res.oracle_recall else "X"
    print(f"[{i:03d}] [{status}] [Ora:{ora_status}] Q: {q.question[:42]} | Ans: {res.predicted_answer[:28]} | GT: {q.ground_truth[:28]} | Tok: {res.tokens_used}")

total_time = time.perf_counter() - t_start

tot_acc = sum(1 for r in results if r.is_correct) / len(results) if results else 0
tot_ora = sum(1 for r in results if r.oracle_recall) / len(results) if results else 0
mean_tok = sum(r.tokens_used for r in results) / len(results) if results else 0
mean_lat = sum(r.latency_ms for r in results) / len(results) if results else 0

# Category Breakdown
cat_stats = {}
for r in results:
    cat_name = adapter.CATEGORY_NAMES.get(r.category, f"Cat-{r.category}")
    if cat_name not in cat_stats:
        cat_stats[cat_name] = {"total": 0, "correct": 0, "oracle": 0}
    cat_stats[cat_name]["total"] += 1
    if r.is_correct:
        cat_stats[cat_name]["correct"] += 1
    if r.oracle_recall:
        cat_stats[cat_name]["oracle"] += 1

# Factual Questions Only (excluding Adversarial)
factual_results = [r for r in results if r.category != 5]
factual_acc = sum(1 for r in factual_results if r.is_correct) / len(factual_results) if factual_results else 0
factual_ora = sum(1 for r in factual_results if r.oracle_recall) / len(factual_results) if factual_results else 0

print("\n" + "=" * 70)
print(f"CONVERSATION 1 FINAL RESULTS (105 Questions):")
print(f"  - Overall Answer Accuracy:  {tot_acc * 100:.1f}% ({sum(1 for r in results if r.is_correct)}/{len(results)})")
print(f"  - Overall Oracle Recall:    {tot_ora * 100:.1f}% ({sum(1 for r in results if r.oracle_recall)}/{len(results)})")
print(f"  - Factual Oracle Recall:    {factual_ora * 100:.1f}% ({sum(1 for r in factual_results if r.oracle_recall)}/{len(factual_results)})")
print(f"  - Factual Answer Accuracy:  {factual_acc * 100:.1f}% ({sum(1 for r in factual_results if r.is_correct)}/{len(factual_results)})")
print(f"  - Mean Context Tokens:      {mean_tok:.1f} tokens/Q")
print(f"  - Mean Latency:             {mean_lat:.1f} ms")
print(f"  - Total Benchmark Elapsed:  {total_time:.1f} s")
print("=" * 70)
print("CATEGORY-BY-CATEGORY BREAKDOWN:")
for cat_name, s in cat_stats.items():
    c_acc = (s["correct"] / s["total"]) * 100 if s["total"] else 0
    c_ora = (s["oracle"] / s["total"]) * 100 if s["total"] else 0
    print(f"  * {cat_name:<15}: Acc {c_acc:5.1f}% | Oracle {c_ora:5.1f}% ({s['correct']}/{s['total']} correct)")
print("=" * 70)
