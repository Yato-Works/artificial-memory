#!/usr/bin/env python3
"""Test just open-domain questions with persona summary."""
import sys, json, time, os
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer, FROZEN_MODEL
from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
import httpx

adapter = LoCoMoAdapter()
# Longer timeout for CPU inference
answerer = OllamaAnswerer(model=FROZEN_MODEL, timeout_seconds=600.0)
weights = EvidenceScoreWeights()

turns, questions, ir_records = adapter.load_conversation(0)

# Only run open-domain (cat 3) questions
od_questions = [q for q in questions if q.category == 3]
print(f"Open-domain questions: {len(od_questions)}", flush=True)

results = []
os.makedirs('benchmark_results/quick_test', exist_ok=True)

for i, q in enumerate(od_questions):
    t0 = time.time()
    res = adapter.evaluate_question(q, turns, ir_records, answerer, weights=weights)
    elapsed = time.time() - t0
    acc = sum(1 for r in results if r.is_correct) + (1 if res.is_correct else 0)
    print(f"[{i+1}/{len(od_questions)}] Acc={acc/(i+1)*100:.0f}% | {elapsed:.1f}s | oracle={res.oracle_recall} | correct={res.is_correct} | Q: {q.question[:50]}", flush=True)
    print(f"  Pred: {res.predicted_answer[:60]}", flush=True)
    print(f"  GT:   {res.ground_truth[:60]}", flush=True)
    results.append(res)
    
    # Save after each
    with open('benchmark_results/quick_test/open_domain_10.jsonl', 'a') as f:
        f.write(json.dumps({**{k:v for k,v in res.__dict__.items()}, 'question': q.question, 'category': q.category}) + "\n")

print(f"\nDone! {sum(1 for r in results if r.is_correct)}/{len(results)} correct", flush=True)
