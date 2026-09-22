"""Check the frozen temporal matcher on observed regression pairs."""
import sys

sys.stdout.reconfigure(encoding="utf-8")
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

pairs = [
    ("The sunday before 25 May 2023", "Last Saturday before 25 May 2023"),
    ("The sunday before 25 May 2023", "Saturday before 25 May 2023"),
    ("The week before 9 June 2023", "The week before 9 June 2023"),
    ("The week before 9 June 2023", "The week before 6 July 2023"),
    ("2022", "2022"),
    ("2022", "I don't know."),
    ("The week before 27 June 2023", "Last week before 27 June 2023"),
    ("The week before 27 June 2023", "June 2023"),
]
for gt, ans in pairs:
    ok = LoCoMoAdapter._temporal_answer_matches(gt.lower().strip(), ans.lower().strip())
    print(f"{ok!s:>5}  gt={gt!r:<34} ans={ans!r}")
