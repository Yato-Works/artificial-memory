"""Temporal 37-Question Failure Funnel Diagnostic Script.

Analyzes the 37 Category 2 (Temporal) questions of LoCoMo-10 Conv 0:
1. Loads questions, ground truth, predicted answers, and evaluation outcomes.
2. Checks oracle recall (evidence retrieved or not).
3. Analyzes context: What temporal grounding or state was provided?
4. Builds the comprehensive Failure Funnel:
   37 Temporal
    ├── Evidence Missing (Oracle Recall FAIL)
    └── Evidence Present (Oracle Recall PASS)
         ├── Temporal State Missing
         ├── Temporal Calculation / Date Difference Failure (LLM error)
         ├── Relative vs Absolute Date Mismatch (e.g. "X days ago" vs "Date")
         └── Surface Form / Formatting Mismatch
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

conv0_file = Path("benchmark_results/frozen/locomo_conv0_frozen.json")
with open(conv0_file, "r", encoding="utf-8") as f:
    conv0_data = json.load(f)

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, all_questions, ir_records = adapter.load_conversation(conv_idx=0)
q_map = {q.question_id: q.question for q in all_questions}

# Filter category 2 (temporal)
temporal_items = [d for d in conv0_data["details"] if d["category"] == 2]

print(f"Total Temporal Questions: {len(temporal_items)}")
num_correct = sum(1 for d in temporal_items if d["is_correct"])
num_oracle = sum(1 for d in temporal_items if d["oracle_recall"])
print(f"Accuracy:      {num_correct}/{len(temporal_items)} ({num_correct/len(temporal_items)*100:.1f}%)")
print(f"Oracle Recall: {num_oracle}/{len(temporal_items)} ({num_oracle/len(temporal_items)*100:.1f}%)")

funnel = {
    "evidence_missing": [],
    "evidence_present": {
        "pass": [],
        "relative_vs_absolute": [],
        "date_calculation_failure": [],
        "temporal_order_failure": [],
        "duration_count_failure": [],
        "other_llm_failure": [],
    }
}

for d in temporal_items:
    qid = d["question_id"]
    q_text = q_map.get(qid, "")
    gt = str(d["ground_truth"]).strip()
    pred = str(d["predicted_answer"]).strip()
    is_corr = d["is_correct"]
    ora = d["oracle_recall"]

    if not ora:
        funnel["evidence_missing"].append({
            "qid": qid,
            "q": q_text,
            "gt": gt,
            "pred": pred,
        })
    else:
        if is_corr:
            funnel["evidence_present"]["pass"].append({
                "qid": qid,
                "q": q_text,
                "gt": gt,
                "pred": pred,
            })
        else:
            q_lower = q_text.lower()
            gt_lower = gt.lower()
            pred_lower = pred.lower()

            # Classify failure mode
            if any(w in q_lower for w in ["how many days", "how many weeks", "how many months", "how long ago", "days passed", "weeks passed"]):
                funnel["evidence_present"]["date_calculation_failure"].append({
                    "qid": qid,
                    "q": q_text,
                    "gt": gt,
                    "pred": pred,
                })
            elif any(w in q_lower for w in ["when did", "what date", "what day", "what year", "what time"]):
                # Did LLM say "yesterday" or a relative date when GT was absolute date, or vice-versa?
                if any(w in pred_lower for w in ["yesterday", "last week", "ago", "recently"]):
                    funnel["evidence_present"]["relative_vs_absolute"].append({
                        "qid": qid,
                        "q": q_text,
                        "gt": gt,
                        "pred": pred,
                        "sub_type": "pred_relative_gt_absolute",
                    })
                elif any(re.findall(r"\b\d{4}\b|\b\d{1,2}\s+[a-zA-Z]+\b|[a-zA-Z]+\s+\d{1,2}\b", pred)):
                    funnel["evidence_present"]["date_calculation_failure"].append({
                        "qid": qid,
                        "q": q_text,
                        "gt": gt,
                        "pred": pred,
                        "sub_type": "date_mismatch",
                    })
                else:
                    funnel["evidence_present"]["other_llm_failure"].append({
                        "qid": qid,
                        "q": q_text,
                        "gt": gt,
                        "pred": pred,
                    })
            elif any(w in q_lower for w in ["before", "after", "first", "last", "earlier", "later"]):
                funnel["evidence_present"]["temporal_order_failure"].append({
                    "qid": qid,
                    "q": q_text,
                    "gt": gt,
                    "pred": pred,
                })
            elif any(w in q_lower for w in ["how long", "duration", "how many years"]):
                funnel["evidence_present"]["duration_count_failure"].append({
                    "qid": qid,
                    "q": q_text,
                    "gt": gt,
                    "pred": pred,
                })
            else:
                funnel["evidence_present"]["other_llm_failure"].append({
                    "qid": qid,
                    "q": q_text,
                    "gt": gt,
                    "pred": pred,
                })

print("\n" + "=" * 80)
print("             TEMPORAL 37-QUESTION FAILURE FUNNEL")
print("=" * 80)
print(f"Total Questions: 37")
print(f"├── Evidence Missing (Oracle Recall FAIL): {len(funnel['evidence_missing'])} ({len(funnel['evidence_missing'])/37*100:.1f}%)")
print(f"└── Evidence Present (Oracle Recall PASS): {num_oracle} ({num_oracle/37*100:.1f}%)")
print(f"     ├── Correct (PASS): {len(funnel['evidence_present']['pass'])} ({len(funnel['evidence_present']['pass'])/37*100:.1f}%)")
print(f"     └── Failed Despite Evidence (Conversion Gap): {num_oracle - len(funnel['evidence_present']['pass'])}")
print(f"          ├── Date Difference / Arithmetic Failure: {len(funnel['evidence_present']['date_calculation_failure'])}")
print(f"          ├── Relative vs Absolute Date Mismatch:   {len(funnel['evidence_present']['relative_vs_absolute'])}")
print(f"          ├── Temporal Order / Event Sequence:      {len(funnel['evidence_present']['temporal_order_failure'])}")
print(f"          ├── Duration / Time Span Failure:         {len(funnel['evidence_present']['duration_count_failure'])}")
print(f"          └── Other Extraction / LLM Refusal:       {len(funnel['evidence_present']['other_llm_failure'])}")
print("=" * 80)

# Print detail samples of failures
print("\n--- SAMPLE FAILURES WHERE EVIDENCE WAS PRESENT ---")
for cat, items in funnel["evidence_present"].items():
    if cat == "pass":
        continue
    print(f"\n[Category: {cat}] ({len(items)} items)")
    for it in items[:4]:
        print(f"  Q: {it['q']}")
        print(f"     GT:   {it['gt']}")
        print(f"     PRED: {it['pred']}")

# Save funnel to file
out_file = Path("benchmark_results/frozen/temporal_failure_funnel.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(funnel, f, indent=2)
print(f"\nSaved full funnel to {out_file}")
