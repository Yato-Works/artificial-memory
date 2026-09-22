"""Decoupled Evaluation (Phase 4).

Evaluates the intrinsic capability of the memory system independently from
downstream LLM reasoning (Test A: Retrieval-only evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from artificial_memory.recall.ir_resolver import ResolvedContext
from artificial_memory.research.benchmarks.arena import ArenaQuestion
from artificial_memory.research.benchmarks.scorer import term_present


@dataclass(frozen=True)
class RetrievalOnlyScore:
    question_id: str
    category: str
    ground_truth_present: bool
    forbidden_present: bool
    abstention_accurate: bool
    passed: bool
    score: float
    retrieved_tokens: int


def evaluate_retrieval_only(
    question: ArenaQuestion,
    resolved: ResolvedContext,
) -> RetrievalOnlyScore:
    """Evaluate whether AM retrieved and resolved the correct truth without calling an LLM.

    - For standard questions: passed if all ground truth groups are present in context_text
      and no forbidden terms are present.
    - For abstention questions: passed if is_abstention is True and no forbidden terms are present.
    """
    ctx_lower = resolved.context_text.lower()
    tokens = len(resolved.context_text) // 4

    # 1. Check forbidden terms
    forbidden_hits = [
        term for term in question.forbidden
        if term_present(term, ctx_lower)
    ]
    forbidden_present = bool(forbidden_hits)

    # 2. Check Abstention
    if question.expected_abstention:
        abstention_accurate = resolved.is_abstention
        passed = abstention_accurate and not forbidden_present
        score = 1.0 if passed else 0.0
        return RetrievalOnlyScore(
            question_id=question.question_id,
            category=question.category.value,
            ground_truth_present=True,
            forbidden_present=forbidden_present,
            abstention_accurate=abstention_accurate,
            passed=passed,
            score=score,
            retrieved_tokens=tokens,
        )

    # 3. Check Ground Truth groups
    covered_groups = sum(
        1 for group in question.ground_truth
        if any(term_present(term, ctx_lower) for term in group)
    )
    all_gt_present = (covered_groups == len(question.ground_truth))

    passed = all_gt_present and not forbidden_present
    score = 1.0 if passed else (covered_groups / len(question.ground_truth) if question.ground_truth else 0.0)

    return RetrievalOnlyScore(
        question_id=question.question_id,
        category=question.category.value,
        ground_truth_present=all_gt_present,
        forbidden_present=forbidden_present,
        abstention_accurate=True,
        passed=passed,
        score=score,
        retrieved_tokens=tokens,
    )
