"""Failure Taxonomy v2 for AM Apex Overdrive Core.

Classifies failure modes into 8 orthogonal causal categories:
1. RETRIEVAL_FAILURE: Evidence not retrieved (Oracle Recall = False).
2. ENTITY_FAILURE: Entity mismatch, unresolved referent, or alias failure.
3. TEMPORAL_FAILURE: Date calculation, duration delta, or chronological order error.
4. STATE_FAILURE: Stale / superseded state chosen instead of active state (or vice-versa).
5. COMPOSITION_FAILURE: Multi-hop reasoning failure (hop 1 found, hop 2 missing).
6. GROUNDING_FAILURE: Context contained evidence, but LLM failed to convert.
7. VERIFICATION_FAILURE: Hallucination not pruned by AnswerVerifier.
8. ABSTENTION_FAILURE: False positive on adversarial/unanswerable query, or false refusal on valid query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class FailureCategoryV2(StrEnum):
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    ENTITY_FAILURE = "ENTITY_FAILURE"
    TEMPORAL_FAILURE = "TEMPORAL_FAILURE"
    STATE_FAILURE = "STATE_FAILURE"
    COMPOSITION_FAILURE = "COMPOSITION_FAILURE"
    GROUNDING_FAILURE = "GROUNDING_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    ABSTENTION_FAILURE = "ABSTENTION_FAILURE"


@dataclass
class FailureDiagnosisV2:
    category: FailureCategoryV2
    explanation: str
    is_failure: bool = True


class FailureClassifierV2:
    """Deterministic failure mode classifier."""

    def classify(
        self,
        question: str,
        ground_truth: str,
        predicted_answer: str,
        context: str,
        oracle_recall: bool,
        is_correct: bool,
        question_type: str = "general",
    ) -> Optional[FailureDiagnosisV2]:
        """Classify a question outcome into Failure Taxonomy v2."""
        if is_correct:
            return None

        q_lower = question.lower()
        gt_lower = str(ground_truth).lower().strip()
        ans_lower = predicted_answer.lower().strip()
        ctx_lower = context.lower()

        # 1. ABSTENTION_FAILURE (Adversarial / unanswerable false positive, or false refusal on known fact)
        if not gt_lower or gt_lower == "none":
            if not any(w in ans_lower for w in ["none", "not mentioned", "unknown", "i don't know"]):
                return FailureDiagnosisV2(
                    category=FailureCategoryV2.ABSTENTION_FAILURE,
                    explanation="False positive: Model hallucinated an answer for an unanswerable/adversarial question.",
                )
        elif any(w in ans_lower for w in ["i don't know", "not mentioned"]) and oracle_recall:
            return FailureDiagnosisV2(
                category=FailureCategoryV2.ABSTENTION_FAILURE,
                explanation="False refusal: Evidence was present in context, but model abstained.",
            )

        # 2. TEMPORAL_FAILURE (Calendar arithmetic, duration, ordering)
        if question_type in ["temporal", "temporal-reasoning"] or any(w in q_lower for w in ["how many days", "how many weeks", "how many months", "order from first to last", "which event happened first"]):
            return FailureDiagnosisV2(
                category=FailureCategoryV2.TEMPORAL_FAILURE,
                explanation="Temporal failure: Inaccurate date arithmetic, duration calculation, or chronological ordering.",
            )

        # 3. STATE_FAILURE (Knowledge update / supersession)
        if question_type == "knowledge-update" or any(w in q_lower for w in ["currently", "now", "previously", "used to", "updated"]):
            return FailureDiagnosisV2(
                category=FailureCategoryV2.STATE_FAILURE,
                explanation="State failure: Stale or superseded state selected instead of current state (or vice-versa).",
            )

        # 4. COMPOSITION_FAILURE (Multi-hop)
        if question_type in ["multi-hop", "multi-session"] or any(w in q_lower for w in ["where did", "why did", "what country", "who is", "how did"]):
            if oracle_recall:
                return FailureDiagnosisV2(
                    category=FailureCategoryV2.COMPOSITION_FAILURE,
                    explanation="Composition failure: Partial evidence retrieved, but multi-hop link to target slot broke.",
                )

        # 5. RETRIEVAL_FAILURE (Evidence missing from context)
        if not oracle_recall:
            return FailureDiagnosisV2(
                category=FailureCategoryV2.RETRIEVAL_FAILURE,
                explanation="Retrieval failure: Ground-truth turn/evidence was not retrieved into context.",
            )

        # 6. ENTITY_FAILURE (Entity / pronoun / alias mismatch)
        if any(w in ans_lower for w in ["her home country", "her dog", "the country", "his"]):
            return FailureDiagnosisV2(
                category=FailureCategoryV2.ENTITY_FAILURE,
                explanation="Entity failure: Unresolved pronoun or generic hypernym in generated answer.",
            )

        # 7. VERIFICATION_FAILURE (Hallucination uncaught)
        if len(ans_lower.split()) > 2 and not any(w in ctx_lower for w in ans_lower.split() if len(w) > 4):
            return FailureDiagnosisV2(
                category=FailureCategoryV2.VERIFICATION_FAILURE,
                explanation="Verification failure: Extraneous hallucinated claim not grounded in context.",
            )

        # 8. GROUNDING_FAILURE (Default when evidence is present but LLM conversion failed)
        return FailureDiagnosisV2(
            category=FailureCategoryV2.GROUNDING_FAILURE,
            explanation="Grounding failure: Evidence was present in context, but downstream LLM failed to extract exact answer.",
        )
