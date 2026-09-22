"""Simulate deterministic abstention gates on cached LoCoMo contexts (no LLM).

For every candidate rule we report, per category, how many currently-wrong
adversarial rows would become correct refusals versus how many currently-correct
rows in every category would be clobbered.  A gate is only viable when the
adversarial gain clearly exceeds the collateral loss.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
CACHE = Path(sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/locomo_context_cache.jsonl")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

STOP = {
    "what", "when", "where", "who", "whom", "which", "whose", "why", "how", "did",
    "does", "do", "is", "are", "was", "were", "has", "have", "had", "the", "a", "an",
    "and", "or", "but", "if", "then", "that", "this", "these", "those", "there",
    "their", "his", "her", "its", "our", "your", "my", "me", "you", "she", "he",
    "they", "them", "it", "of", "in", "on", "at", "to", "for", "with", "about",
    "from", "by", "as", "into", "during", "before", "after", "between", "while",
    "both", "all", "most", "least", "some", "any", "many", "much", "more", "than",
    "like", "would", "could", "should", "can", "will", "may", "might", "must",
    "get", "got", "go", "went", "going", "been", "being", "very", "just", "also",
    "not", "no", "yes", "know", "think", "want", "say", "said", "tell", "told",
    "mention", "mentioned", "talk", "talked", "discuss", "discussed", "remind",
}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"\b[a-z0-9']+\b", text.lower())
            if w not in STOP and len(w) > 2}


def parse_turns(context: str) -> list[tuple[str, str]]:
    """Split compiled context into (speaker, text) turns."""
    turns = []
    for line in context.split("\n"):
        m = re.match(r"\s*\[([^\]]+)\]\s*(.*)", line, re.DOTALL)
        if not m:
            continue
        header, body = m.group(1), m.group(2)
        speaker = "unknown"
        for cand in re.findall(r"\b([A-Z][a-z]{2,})\s*:", body):
            speaker = cand.lower()
            break
        turns.append((speaker, body.lower()))
    return turns


def main() -> None:
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[row["qid"]] = row
    print(f"cached contexts: {len(cache)}")

    rows = []
    for p in sorted(RUN.glob("conv_*_results.json")):
        d = json.load(open(p, encoding="utf-8"))
        for r in d["results"]:
            c = cache.get(r["question_id"])
            if c is None:
                continue
            rows.append((r, c["context"], c["question"]))
    print(f"joined rows: {len(rows)}")

    # ---- candidate rules -------------------------------------------------
    def rule_no_support(q: str, ctx: str, ans: str) -> bool:
        """No single turn shares at least one content word with the question."""
        qw = words(q)
        if not qw:
            return False
        for _, text in parse_turns(ctx):
            if qw & words(text):
                return True
        return False

    def rule_thin_support(q: str, ctx: str, ans: str) -> bool:
        """Best turn covers < 20% of the question's content words."""
        qw = words(q)
        if not qw:
            return False
        best = 0.0
        for _, text in parse_turns(ctx):
            if not text:
                continue
            share = len(qw & words(text)) / len(qw)
            best = max(best, share)
        return best < 0.2

    def rule_entity_absent(q: str, ctx: str, ans: str) -> bool:
        """A capitalised question entity (not a speaker) is absent from context."""
        speakers = {s for s, _ in parse_turns(ctx)}
        ctx_low = ctx.lower()
        toks = re.findall(r"\b[A-Z][a-z]{2,}\b", q)
        for tok in toks[1:]:
            low = tok.lower()
            if low in speakers or low in STOP:
                continue
            if low not in ctx_low:
                return True
        return False

    rules = {
        "NO_SUPPORT_TURN": rule_no_support,
        "THIN_SUPPORT_<0.2": rule_thin_support,
        "ENTITY_ABSENT": rule_entity_absent,
    }

    for name, rule in rules.items():
        stats = defaultdict(lambda: {"n": 0, "fire": 0, "fire_wrong": 0, "fire_right": 0})
        for r, ctx, q in rows:
            cat = CAT.get(r["category"], str(r["category"]))
            s = stats[cat]
            s["n"] += 1
            try:
                fires = rule(q, ctx, r["predicted_answer"] or "")
            except Exception:
                fires = False
            if fires:
                s["fire"] += 1
                if r["is_correct"]:
                    s["fire_right"] += 1
                else:
                    s["fire_wrong"] += 1
        print(f"\n### {name}")
        print(f"{'category':<14}{'n':>6}{'fires':>7}{'on wrong':>10}{'on correct':>12}")
        gain = loss = 0
        for cat in sorted(stats, key=lambda c: -stats[c]["n"]):
            s = stats[cat]
            print(f"{cat:<14}{s['n']:>6}{s['fire']:>7}{s['fire_wrong']:>10}{s['fire_right']:>12}")
            gain += s["fire_wrong"]
            loss += s["fire_right"]
        print(f"{'NET':<14}{'':>6}{'':>7}{'GAIN ' + str(gain):>10}{'LOSS ' + str(loss):>12}"
              f"   -> net {gain - loss:+d}")


if __name__ == "__main__":
    main()
