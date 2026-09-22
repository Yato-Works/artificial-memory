"""Evidence Evaluator for AM Apex Steroid Phase (Steroid #3).

Directly measures Evidence Recall across expansion hops independently of LLM generation:
    R_k = |Required Evidence in E_k| / |Required Evidence|

Generates the quantitative Evidence Recall Curve:
    k = 0 (Wide Slicing Baseline)
    k = 1 (1-hop expansion)
    k = 2 (2-hop expansion)
    k = 3 (3-hop expansion)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.steroid.adaptive_graph_expander import AdaptiveGraphExpander
from artificial_memory.steroid.wide_slicer import WideSlicer


@dataclass
class EvidenceRecallPoint:
    hop: int
    recall: float
    total_evaluated: int
    found_count: int


@dataclass
class EvidenceRecallCurve:
    points: list[EvidenceRecallPoint]
    delta_recall: float  # R_max - R_0


class EvidenceEvaluator:
    """Evaluates evidence recall curves on benchmark items."""

    def __init__(self) -> None:
        self.slicer = WideSlicer()
        self.expander = AdaptiveGraphExpander()

    def evaluate_locomo_curve(
        self,
        questions: list,
        corpus_records: Sequence[StructuredIR],
        max_hops: int = 3,
    ) -> EvidenceRecallCurve:
        """Measure R_0 through R_max on LoCoMo questions with evidence_ids."""
        hop_hits = [0] * (max_hops + 1)
        valid_qs = [q for q in questions if q.evidence_ids]
        total_q = len(valid_qs)

        if total_q == 0:
            return EvidenceRecallCurve(points=[], delta_recall=0.0)

        for q in valid_qs:
            ev_ids = q.evidence_ids
            # Hop 0: Wide Slicing
            slice_res = self.slicer.slice(q.question, corpus_records)
            e0_records = slice_res.candidate_records

            # Check Hop 0
            e0_text = " ".join(r.raw_content for r in e0_records)
            if any(eid in e0_text for eid in ev_ids):
                hop_hits[0] += 1

            # Successive Hops
            current_pool = list(e0_records)
            seen_keys = {r.raw_content for r in current_pool if r.raw_content}

            for k in range(1, max_hops + 1):
                # Expand
                exp_res = self.expander.expand(q.question, current_pool, corpus_records)
                current_pool = exp_res.evidence_pool
                pool_text = " ".join(r.raw_content for r in current_pool)
                if any(eid in pool_text for eid in ev_ids):
                    hop_hits[k] += 1

        points = []
        for k in range(max_hops + 1):
            r_k = hop_hits[k] / total_q * 100.0
            points.append(EvidenceRecallPoint(hop=k, recall=r_k, total_evaluated=total_q, found_count=hop_hits[k]))

        delta = points[-1].recall - points[0].recall
        return EvidenceRecallCurve(points=points, delta_recall=delta)
