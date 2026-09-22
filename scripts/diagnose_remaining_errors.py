"""Error Taxonomy Diagnostic for AM Apex (Phase X.3).

Decomposes all remaining failures into mutually exclusive diagnostic buckets:
1. Retrieval Failure (Ora: False)
2. Temporal Grounding Gaps (Relative expressions not yet captured)
3. Scorer Under-matching (Semantic synonym / phrasing divergence)
4. LLM Extraction / Reasoning Gap (Evidence present, but model failed)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 70)
print("AM APEX PHASE X.3: REMAINING ERROR TAXONOMY DIAGNOSTIC")
print("=" * 70)

adapter = LoCoMoAdapter()
turns, questions, ir_records = adapter.load_conversation(conv_idx=0)
answerer = OllamaAnswerer()
weights = EvidenceScoreWeights()

eval_questions = questions[:50]

bucket_retrieval_fail = []
bucket_temporal_gap = []
bucket_scorer_gap = []
bucket_llm_gap = []

for i, q in enumerate(eval_questions):
    pcc = adapter.compiler.compile(q.question, ir_records, weights=weights)
    has_ev = False
    if q.evidence_ids:
        has_ev = any(ev in pcc.context_text for ev in q.evidence_ids)
    else:
        has_ev = True

    if pcc.is_abstention:
        ans_text = "I don't know."
    else:
        ans_text = answerer.answer(q.question, pcc.context_text).text

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

    if not is_correct:
        item = {
            "idx": i,
            "qid": q.question_id,
            "q": q.question,
            "ans": ans_text,
            "gt": q.ground_truth,
            "has_ev": has_ev,
            "category": q.category,
        }
        # Bucket Classification
        if not has_ev:
            bucket_retrieval_fail.append(item)
        elif q.category == 2 or any(w in q.question.lower() for w in ["when", "how long"]):
            bucket_temporal_gap.append(item)
        elif any(w in ans_lower for w in ["mel", "equine", "playing games", "clay pots", "married"]):
            bucket_llm_gap.append(item)
        else:
            bucket_scorer_gap.append(item)

print(f"\nTOTAL EVALUATED: {len(eval_questions)} | PASS: {len(eval_questions) - (len(bucket_retrieval_fail) + len(bucket_temporal_gap) + len(bucket_scorer_gap) + len(bucket_llm_gap))} | FAIL: {len(bucket_retrieval_fail) + len(bucket_temporal_gap) + len(bucket_scorer_gap) + len(bucket_llm_gap)}")
print("-" * 70)
print(f"1. Retrieval Failure (Ora: False): {len(bucket_retrieval_fail)} questions (Target: Universal Retrieval / Graph)")
for it in bucket_retrieval_fail[:5]:
    print(f"   [{it['idx']:02d}] Q: {it['q'][:45]} | GT: {it['gt'][:30]}")

print(f"\n2. Temporal Grounding Gaps: {len(bucket_temporal_gap)} questions (Target: Advanced Temporal Normalizer)")
for it in bucket_temporal_gap[:5]:
    print(f"   [{it['idx']:02d}] Q: {it['q'][:45]} | Ans: {it['ans'][:25]} | GT: {it['gt'][:30]}")

print(f"\n3. Scorer / Synonym Gaps: {len(bucket_scorer_gap)} questions (Target: Semantic Evaluation Scorer)")
for it in bucket_scorer_gap[:5]:
    print(f"   [{it['idx']:02d}] Q: {it['q'][:45]} | Ans: {it['ans'][:25]} | GT: {it['gt'][:30]}")

print(f"\n4. LLM Synthesis / Knowledge Gaps: {len(bucket_llm_gap)} questions (Target: Context IR Compiler / Prompt)")
for it in bucket_llm_gap[:5]:
    print(f"   [{it['idx']:02d}] Q: {it['q'][:45]} | Ans: {it['ans'][:25]} | GT: {it['gt'][:30]}")
print("=" * 70)
