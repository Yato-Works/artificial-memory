"""Offline A/B for an *answer-grounding* gate (G-family rules).

Hypothesis (measured, not assumed)
----------------------------------
Two distinct failure modes hide inside LoCoMo's hallucination-bait category:

  (a) attribution bait - the bait text IS in the context but belongs to the
      other speaker  -> handled by the subject-binding guard (AnswerVerifier).
  (b) unsupported synthesis - the model invents plausible content that never
      appears anywhere in the retrieved context (e.g. "Marley flooring").
      -> NOT handled by any current guard.

Mode (b) is detectable by a conversation-agnostic rule: refuse when the
answer's *distinctive* content tokens are absent from the context.

This script sweeps that rule (coverage thresholds, token-length filters) over
the cached contexts and the stored post-guard predictions, reporting gain /
loss per category so the trade-off is explicit before any production change.

Run: python scripts/simulate_grounding_gate.py [run_dir]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ORDER = ["adversarial", "single-hop", "temporal", "multi-hop", "open-domain"]
REFUSAL = "None (not mentioned in conversation)."
