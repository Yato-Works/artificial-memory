"""Failure Attribution Engine (Phase 4).

Systematically diagnoses and attributes benchmark failures into four root causes:
1. AM Retrieval Failure: The relevant ground-truth memory was not retrieved.
2. AM Resolution Failure: The memory was retrieved, but AM's IR/temporal/entity/conflict resolution selected the wrong value or state.
3. LLM Reasoning Failure: AM provided the correct evidence in Context IR, but the downstream LLM failed to infer the correct answer.
4. Evaluator Failure: The LLM answer was semantically correct, but the strict deterministic scorer failed to match the declared lexicon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Sequence

from artificial_memory.core.ir import StructuredIR
from artificial_memory.recall.ir_resolver import ResolvedContext
from artificial_memory.research.benchmarks.arena import ArenaQuestion
from artificial_memory.research.benchmarks.scorer import QuestionScore, term_present


class FailureOrigin(StrEnum):
    AM_RETRIEVAL_FAILURE = "AM_Retrieval_Failure"
    AM_RESOLUTION_FAILURE = "AM_Resolution_Failure"
    LLM_REASONING_FAILURE = "LLM_Reasoning_Failure"
    EVALUATOR_FAILURE = "Evaluator_Failure"
    DATASET_ERROR = "Dataset_Error"
    NONE = "None"  # Passed


@dataclass
class FailureTrace:
    """Complete diagnostic audit trail for an evaluated question (8 core items)."""
    question_id: str
    category: str
    question_text: str

    # 1. Ground Truth
    ground_truth: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...]

    # 2. Expected Answer / Abstention status
    expected_abstention: bool

    # 3. AM Retrieved Memories
    retrieved_memories: list[str]

    # 4. AM Context IR
    context_ir_text: str
    ir_records_count: int

    # 5. LLM Input (Full prompt including context)
    llm_input_prompt: str

    # 6. LLM Output (Generated response)
    llm_output_text: str

    # 7. Evaluator Decision
    outcome: str
    score: float
    forbidden_violation: bool

    # 8. Failure Origin (Diagnosed cause)
    failure_origin: FailureOrigin
    diagnostic_reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "question_text": self.question_text,
            "ground_truth": [list(g) for g in self.ground_truth],
            "forbidden": list(self.forbidden),
            "expected_abstention": self.expected_abstention,
            "retrieved_memories": self.retrieved_memories,
            "context_ir_text": self.context_ir_text,
            "ir_records_count": self.ir_records_count,
            "llm_input_prompt": self.llm_input_prompt,
            "llm_output_text": self.llm_output_text,
            "outcome": self.outcome,
            "score": self.score,
            "forbidden_violation": self.forbidden_violation,
            "failure_origin": self.failure_origin.value,
            "diagnostic_reasoning": self.diagnostic_reasoning,
        }


class FailureAttributor:
    """Diagnoses the precise origin of failure for any evaluated question."""

    def diagnose(
        self,
        question: ArenaQuestion,
        resolved: ResolvedContext,
        llm_input: str,
        llm_output: str,
        score: QuestionScore,
        raw_corpus: Sequence[str] | None = None,
    ) -> FailureTrace:
        """Diagnose failure origin using a deterministic decision tree."""
        if score.outcome == "pass":
            return FailureTrace(
                question_id=question.question_id,
                category=question.category.value,
                question_text=question.question,
                ground_truth=question.ground_truth,
                forbidden=question.forbidden,
                expected_abstention=question.expected_abstention,
                retrieved_memories=[r.raw_content for r in resolved.matched_records],
                context_ir_text=resolved.context_text,
                ir_records_count=len(resolved.matched_records),
                llm_input_prompt=llm_input,
                llm_output_text=llm_output,
                outcome=score.outcome,
                score=score.score,
                forbidden_violation=score.forbidden_violation,
                failure_origin=FailureOrigin.NONE,
                diagnostic_reasoning="Passed all criteria.",
            )

        # Step 1: Check if Ground Truth was even present in the raw corpus
        gt_in_corpus = True
        if raw_corpus and question.ground_truth and not question.expected_abstention:
            gt_in_corpus = any(
                any(term.lower() in c.lower() for c in raw_corpus)
                for group in question.ground_truth
                for term in group
            )
            if not gt_in_corpus:
                return self._create_trace(
                    question, resolved, llm_input, llm_output, score,
                    FailureOrigin.DATASET_ERROR,
                    "Ground truth term is absent from the entire conversation history."
                )

        # Step 2: Check if AM's Context IR contained the Ground Truth
        gt_in_context = True
        if question.ground_truth and not question.expected_abstention:
            # Does the context_text satisfy all ground truth groups?
            norm_ctx = resolved.context_text.lower()
            groups_covered = sum(
                1 for group in question.ground_truth
                if any(term_present(term, norm_ctx) for term in group)
            )
            gt_in_context = (groups_covered == len(question.ground_truth))

        if not gt_in_context and not question.expected_abstention:
            # Did AM retrieve the memory containing GT, but filtered/superseded it during resolution?
            retrieved_has_gt = any(
                any(term_present(term, r.raw_content.lower()) for term in group)
                for r in resolved.matched_records
                for group in question.ground_truth
            )
            if retrieved_has_gt:
                return self._create_trace(
                    question, resolved, llm_input, llm_output, score,
                    FailureOrigin.AM_RESOLUTION_FAILURE,
                    "AM retrieved the relevant memory, but resolution logic (temporal/update/filter) excluded or superseded the ground truth in the final context."
                )
            else:
                return self._create_trace(
                    question, resolved, llm_input, llm_output, score,
                    FailureOrigin.AM_RETRIEVAL_FAILURE,
                    "AM failed to retrieve the memory containing the ground truth evidence."
                )

        # Step 3: Check if Context contained the answer, but LLM failed to infer it
        # If ground truth was in context, but LLM output lacked it:
        if gt_in_context and not question.expected_abstention:
            norm_llm = llm_output.lower()
            llm_has_gt = any(
                any(term_present(term, norm_llm) for term in group)
                for group in question.ground_truth
            )
            if not llm_has_gt:
                return self._create_trace(
                    question, resolved, llm_input, llm_output, score,
                    FailureOrigin.LLM_REASONING_FAILURE,
                    "AM provided the complete ground truth in Context IR, but the downstream LLM failed to extract/state it in its final response."
                )

        # Step 4: Abstention failure check
        if question.expected_abstention:
            if resolved.is_abstention:
                # AM knew it should abstain, but LLM output failed scorer
                return self._create_trace(
                    question, resolved, llm_input, llm_output, score,
                    FailureOrigin.LLM_REASONING_FAILURE,
                    "AM signaled abstention, but downstream LLM output failed to match the scorer's abstention lexicon."
                )
            else:
                return self._create_trace(
                    question, resolved, llm_input, llm_output, score,
                    FailureOrigin.AM_RESOLUTION_FAILURE,
                    "AM failed to recognize that the queried property was absent/never configured, passing spurious context."
                )

        # Step 5: Evaluator Failure (LLM answer is semantically right, but rejected by scorer)
        return self._create_trace(
            question, resolved, llm_input, llm_output, score,
            FailureOrigin.EVALUATOR_FAILURE,
            "LLM output conveys the correct information but was rejected by the strict term-matching evaluator."
        )

    def _create_trace(
        self,
        question: ArenaQuestion,
        resolved: ResolvedContext,
        llm_input: str,
        llm_output: str,
        score: QuestionScore,
        origin: FailureOrigin,
        reason: str,
    ) -> FailureTrace:
        return FailureTrace(
            question_id=question.question_id,
            category=question.category.value,
            question_text=question.question,
            ground_truth=question.ground_truth,
            forbidden=question.forbidden,
            expected_abstention=question.expected_abstention,
            retrieved_memories=[r.raw_content for r in resolved.matched_records],
            context_ir_text=resolved.context_text,
            ir_records_count=len(resolved.matched_records),
            llm_input_prompt=llm_input,
            llm_output_text=llm_output,
            outcome=score.outcome,
            score=score.score,
            forbidden_violation=score.forbidden_violation,
            failure_origin=origin,
            diagnostic_reasoning=reason,
        )
