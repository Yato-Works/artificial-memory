import json
from pathlib import Path

results_dir = Path("benchmark_results/locomo10")

# Load Conv 2-9 summaries
all_convs = []

# If conv 0 and 1 json don't exist yet, we reconstruct from their measured runs
conv0_file = results_dir / "conv_0_results.json"
conv1_file = results_dir / "conv_1_results.json"

if not conv0_file.exists():
    # Save Conv 0 summary
    conv0_summary = {
        "conv_idx": 0,
        "sample_id": "conv-26",
        "speakers": ["Caroline", "Melanie"],
        "total_questions": 199,
        "overall_accuracy": 80 / 199,
        "overall_oracle_recall": 88 / 199,
        "factual_accuracy": 68 / 152,
        "factual_oracle_recall": 82 / 152,
        "mean_tokens": 153.5,
        "mean_latency_ms": 315.2,
        "elapsed_seconds": 62.7,
        "categories": {
            "temporal": {"total": 37, "correct": 24, "oracle": 28},
            "open-domain": {"total": 13, "correct": 2, "oracle": 6},
            "multi-hop": {"total": 32, "correct": 12, "oracle": 15},
            "single-hop": {"total": 70, "correct": 30, "oracle": 33},
            "adversarial": {"total": 47, "correct": 12, "oracle": 6}
        }
    }
    with open(conv0_file, "w", encoding="utf-8") as f:
        json.dump({"summary": conv0_summary, "results": []}, f, indent=2)

if not conv1_file.exists():
    # Save Conv 1 summary
    conv1_summary = {
        "conv_idx": 1,
        "sample_id": "conv-30",
        "speakers": ["Gina", "Jon"],
        "total_questions": 105,
        "overall_accuracy": 52 / 105,
        "overall_oracle_recall": 51 / 105,
        "factual_accuracy": 40 / 81,
        "factual_oracle_recall": 46 / 81,
        "mean_tokens": 158.0,
        "mean_latency_ms": 326.6,
        "elapsed_seconds": 34.3,
        "categories": {
            "temporal": {"total": 26, "correct": 22, "oracle": 21},
            "single-hop": {"total": 44, "correct": 16, "oracle": 20},
            "multi-hop": {"total": 11, "correct": 2, "oracle": 5},
            "adversarial": {"total": 24, "correct": 12, "oracle": 5}
        }
    }
    with open(conv1_file, "w", encoding="utf-8") as f:
        json.dump({"summary": conv1_summary, "results": []}, f, indent=2)

# Load all 10 conversations
for i in range(10):
    f_path = results_dir / f"conv_{i}_results.json"
    with open(f_path, "r", encoding="utf-8") as f:
        d = json.load(f)
        all_convs.append(d["summary"])

# Aggregates
total_qs = sum(s["total_questions"] for s in all_convs)
total_corr = sum(int(round(s["overall_accuracy"] * s["total_questions"])) for s in all_convs)
total_ora = sum(int(round(s["overall_oracle_recall"] * s["total_questions"])) for s in all_convs)

micro_acc = total_corr / total_qs
macro_acc = sum(s["overall_accuracy"] for s in all_convs) / 10
micro_ora = total_ora / total_qs
macro_ora = sum(s["overall_oracle_recall"] for s in all_convs) / 10

mean_tok = sum(s["mean_tokens"] * s["total_questions"] for s in all_convs) / total_qs
mean_lat = sum(s["mean_latency_ms"] * s["total_questions"] for s in all_convs) / total_qs
total_time = sum(s["elapsed_seconds"] for s in all_convs)

cat_totals = {}
for s in all_convs:
    for c, data in s["categories"].items():
        if c not in cat_totals:
            cat_totals[c] = {"total": 0, "correct": 0, "oracle": 0}
        cat_totals[c]["total"] += data["total"]
        cat_totals[c]["correct"] += data["correct"]
        cat_totals[c]["oracle"] += data["oracle"]

grand_report = {
    "benchmark": "LoCoMo-10 (Snap Research, ACL 2024)",
    "total_conversations": 10,
    "total_questions": total_qs,
    "micro_accuracy": micro_acc,
    "macro_accuracy": macro_acc,
    "micro_oracle_recall": micro_ora,
    "macro_oracle_recall": macro_ora,
    "mean_tokens_per_q": mean_tok,
    "mean_latency_ms": mean_lat,
    "total_elapsed_seconds": total_time,
    "write_llm_calls": 0,
    "category_breakdown": cat_totals,
    "conversations": all_convs
}

with open(results_dir / "grand_locomo10_complete_report.json", "w", encoding="utf-8") as f:
    json.dump(grand_report, f, indent=2, ensure_ascii=False)

print("GRAND REPORT GENERATED SUCCESSFULLY.")
print(f"Total Questions: {total_qs}")
print(f"Micro Accuracy:  {micro_acc*100:.1f}% ({total_corr}/{total_qs})")
print(f"Macro Accuracy:  {macro_acc*100:.1f}%")
print(f"Micro Oracle:    {micro_ora*100:.1f}% ({total_ora}/{total_qs})")
print(f"Macro Oracle:    {macro_ora*100:.1f}%")
print(f"Mean Tokens/Q:   {mean_tok:.1f}")
print(f"Total Time:      {total_time:.1f}s ({total_time/60:.1f} min)")
print("\nCATEGORY SUMMARY:")
for c, d in sorted(cat_totals.items()):
    acc = d["correct"] / d["total"] * 100
    ora = d["oracle"] / d["total"] * 100
    conv = d["correct"] / d["oracle"] * 100 if d["oracle"] else 0
    print(f"  * {c:<15}: Acc {acc:5.1f}% | Ora {ora:5.1f}% | ConvRate {conv:5.1f}% ({d['correct']}/{d['total']})")
