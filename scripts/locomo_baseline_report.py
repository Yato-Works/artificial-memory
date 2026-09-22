"""Print per-conversation baseline summary from the frozen LoCoMo-10 report."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open("benchmark_results/locomo10/full_locomo10_report.json", encoding="utf-8"))
print(f"MICRO acc={d['micro_accuracy'] * 100:.2f}% oracle={d['micro_oracle_recall'] * 100:.2f}% "
      f"tok={d['mean_tokens']:.1f} n={d['total_questions']}")
print(f"{'conv':>5}{'acc':>9}{'oracle':>9}{'tokens':>9}{'n':>6}")
for c in d["conversations"]:
    s = c.get("summary", c)
    print(f"{s.get('conv_idx'):>5}{s.get('overall_accuracy', 0) * 100:>8.1f}%"
          f"{s.get('overall_oracle_recall', 0) * 100:>8.1f}%"
          f"{s.get('mean_tokens', 0):>9.0f}{s.get('total_questions'):>6}")
