#!/usr/bin/env python3
"""Run LoCoMo with retries and longer timeout."""
import sys, json, time, os
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer, FROZEN_MODEL
from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
import httpx

# Use a much longer timeout 
adapter = LoCoMoAdapter()
answerer = OllamaAnswerer(model=FROZEN_MODEL, timeout_seconds=600.0)
weights = EvidenceScoreWeights()

turns, questions, ir = adapter.load_conversation(0)
results = []

os.makedirs('benchmark_results/quick_test', exist_ok=True)

for i, q in enumerate(questions[:10]):
    t0 = time.time()
    # Retry on timeout
    for attempt in range(3):
        try:
            res = adapter.evaluate_question(q, turns, ir, answerer, weights=weights)
            break
        except (httpx.ReadTimeout, Exception) as e:
            if attempt < 2:
                print(f'[{i+1}/10] Retry {attempt+1} after error: {type(e).__name__}', flush=True)
                time.sleep(5)
            else:
                raise
    elapsed = time.time() - t0
    n_correct = sum(1 for r in results if r.is_correct) + (1 if res.is_correct else 0)
    print(f'[{i+1}/10] Acc={n_correct/(i+1)*100:.0f}% | {elapsed:.1f}s | Q: {q.question[:40]}', flush=True)
    results.append(res)
    
    # Save intermediate results
    with open('benchmark_results/quick_test/conv_0_partial.json', 'w') as f:
        json.dump({'results': [r.__dict__ for r in results]}, f, indent=2)

print('Done! Saved to benchmark_results/quick_test/conv_0.json')
