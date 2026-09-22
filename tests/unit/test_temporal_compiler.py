"""Regression tests for deterministic Chronos calendar anchoring."""

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall.answer_verifier import AnswerVerifier
from artificial_memory.temporal.temporal_compiler import TemporalCompiler


def _record(text: str, date: str, turn: str = "D1:1") -> StructuredIR:
    return StructuredIR(
        entity="melanie",
        property="statement",
        value=text,
        raw_content=f"[{turn} on {date}] Melanie: {text}",
        time_scope=date,
    )


def _state_for(question: str, record: StructuredIR) -> str:
    states = TemporalCompiler().compile_temporal_states(question, [record])
    assert len(states) == 1
    return states[0].format_state()


def test_chronos_anchors_abbreviated_weekday() -> None:
    state = _state_for(
        "When did Melanie go to the pottery workshop?",
        _record("Last Fri I took my kids to a pottery workshop.", "15 July 2023"),
    )
    assert "The Friday before 15 July 2023" in state


def test_chronos_anchors_relative_quantity_and_continuing_duration() -> None:
    conference = _state_for(
        "When did Melanie go to the conference?",
        _record("I went to a conference two days ago.", "12 July 2023"),
    )
    art = _state_for(
        "How long has Melanie been practicing art?",
        _record("Seven years now, I have been practicing art.", "20 September 2023"),
    )
    assert "10 July 2023" in conference
    assert "Since 2016" in art


def test_chronos_anchors_year_and_last_night() -> None:
    last_year = _state_for(
        "When did Melanie read the book?",
        _record("This book I read last year changed my life.", "15 July 2023"),
    )
    birthday = _state_for(
        "When was Melanie's daughter's birthday?",
        _record("Last night we celebrated my daughter's birthday.", "14 August 2023"),
    )
    assert "in 2022" in last_year
    assert "on 13 August 2023" in birthday


def test_chronos_prefers_inflected_event_match_over_unrelated_temporal_turn() -> None:
    records = [
        _record("Last week I went camping with my family.", "27 June 2023", "D4:7"),
        _record("Last year I painted a sunrise.", "27 June 2023", "D4:8"),
    ]
    state = _state_for("When did Melanie go camping?", records[0])
    assert "The week before 27 June 2023" in state


def test_answer_verifier_recovers_new_relative_forms() -> None:
    context = "[TEMPORAL STATE] Melanie read the book in 2022. (supported_by: [D7:8])"
    result = AnswerVerifier().verify(
        question="When did Melanie read the book?",
        predicted_answer="Last year",
        context=context,
        propositions=[],
    )
    assert result.verified_answer == "2022"
