"""The adapter must build LongMemEval prompts through the shared module.

``lme_prompts`` is the single source of truth for the prompt structure, and the
measured failure-loop deltas (+21/0 on the 110-question decision set, 10 of 17
temporal refusals recovered) were obtained with it.  If the adapter ever goes
back to building prompts inline, those numbers stop describing production, so
this test locks the wiring.
"""

from artificial_memory.research.benchmarks.external import lme_prompts
from artificial_memory.research.benchmarks.external.longmemeval_adapter import (
    LongMemEvalAdapter,
    LongMemEvalItem,
)


class _CapturingAnswerer:
    """Stands in for OllamaAnswerer and records the prompt it receives."""

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    def answer(self, question_text: str, context: str):
        self.prompts.append((question_text, context))

        class _A:
            text = "a guitar"

        return _A()


def _item(qtype: str) -> LongMemEvalItem:
    return LongMemEvalItem(
        question_id="q1",
        question_type=qtype,
        question="What instrument did the user buy?",
        question_date="2023/01/01",
        answer="a guitar",
        answer_session_ids=["s1"],
        haystack_sessions=[[{"role": "user", "content": "I bought a guitar today."}]],
        haystack_dates=["2023/01/01"],
        haystack_session_ids=["s1"],
    )


ALL_TYPES = (
    "single-session-user",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-assistant",
    "single-session-preference",
)


def test_adapter_builds_prompts_through_shared_module() -> None:
    adapter = LongMemEvalAdapter()
    for qtype in ALL_TYPES:
        answerer = _CapturingAnswerer()
        adapter.evaluate_item(_item(qtype), answerer)
        assert answerer.prompts, f"{qtype}: the reader must always be called"
        prompt = answerer.prompts[0][1]
        if qtype == "single-session-preference":
            # Measured: the evidence-first preamble loses the preference grounding.
            assert "USER PROFILE & PREFERENCES" in prompt
            assert lme_prompts.EVIDENCE_FIRST not in prompt
        else:
            assert lme_prompts.EVIDENCE_FIRST in prompt, qtype


def test_single_session_user_gets_an_instruction_branch() -> None:
    """Historically this type had no branch at all, so the reader refused."""
    adapter = LongMemEvalAdapter()
    answerer = _CapturingAnswerer()
    adapter.evaluate_item(_item("single-session-user"), answerer)
    prompt = answerer.prompts[0][1]
    assert lme_prompts.SS_USER_INSTRUCTION in prompt


def test_temporal_no_longer_bypasses_the_reader() -> None:
    """The resolver's abstention verdict used to answer without any LLM call."""
    adapter = LongMemEvalAdapter()
    answerer = _CapturingAnswerer()
    result = adapter.evaluate_item(_item("temporal-reasoning"), answerer)
    # The fake reader answered "a guitar", so the run must not have been replaced
    # by the old hard-coded "not enough to answer this question" string.
    assert "not enough to answer this question" not in str(result.predicted_answer)
    assert answerer.prompts, "temporal questions must reach the reader"


def test_production_fixes_are_the_adopted_ones() -> None:
    adopted = lme_prompts.ADOPTED
    assert adopted.evidence_first is True
    assert adopted.fix_ss_user is True
    assert adopted.fix_multi is True
    assert adopted.fix_temporal is True
    # Measured net-zero with added risk / tokens: kept off, not deleted.
    assert adopted.fix_ku is False
    assert adopted.fix_ss_assist is False
