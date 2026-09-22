"""Guardrails against inflated LoCoMo temporal benchmark scores."""

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter


def test_temporal_scorer_rejects_year_only_date_match() -> None:
    assert not LoCoMoAdapter._temporal_answer_matches("24 August 2023", "2 July 2023")
    assert not LoCoMoAdapter._temporal_answer_matches("7 May 2023", "8 May 2023")


def test_temporal_scorer_rejects_wrong_weekday_but_accepts_formatting_variant() -> None:
    assert not LoCoMoAdapter._temporal_answer_matches(
        "The Sunday before 25 May 2023", "The Saturday before 25 May 2023"
    )
    assert LoCoMoAdapter._temporal_answer_matches("13 August", "13 Aug 2023")
    assert LoCoMoAdapter._temporal_answer_matches(
        "The week before 9 June 2023", "It was the week before 9 June, 2023."
    )


def test_temporal_scorer_tokenises_glued_official_dates() -> None:
    """Official GTs glue day+month ("10July"); the answer is still correct."""
    assert LoCoMoAdapter._temporal_answer_matches(
        "The Friday before 10July, 2022.", "Friday, before 10 July 2022"
    )
    assert LoCoMoAdapter._temporal_answer_matches(
        "The week before 22August, 2022.", "The week before 22 August 2022"
    )
    assert LoCoMoAdapter._temporal_answer_matches(
        "The Saturday before 7November, 2022", "Last Saturday before 7 November 2022"
    )


def test_temporal_scorer_strips_ordinal_suffix() -> None:
    assert LoCoMoAdapter._temporal_answer_matches(
        "The week before October 13th, 2023.", "The week before October 13, 2023"
    )
    assert LoCoMoAdapter._temporal_answer_matches("Aug 15th", "15 August 2023")


def test_temporal_scorer_keeps_incomplete_and_wrong_dates_failing() -> None:
    """Tokenisation repair must not weaken the calendar check itself."""
    # "the weekend of X" is not satisfied by the bare anchor date.
    assert not LoCoMoAdapter._temporal_answer_matches(
        "The weekend of 22August, 2022.", "22 August, 2022"
    )
    # Glued token must not rescue a wrong weekday / wrong anchor.
    assert not LoCoMoAdapter._temporal_answer_matches(
        "The Friday before 14September, 2022", "Thursday before 14 September 2022"
    )
    assert not LoCoMoAdapter._temporal_answer_matches(
        "The Friday before 9October, 2022.", "Last Friday before 9 October 2023"
    )


def test_refusal_detection_is_word_bounded() -> None:
    """Hallucinated answers must not be paid for containing "no" as a substring."""
    for hallucination in (
        "Fantasy novels",
        "Learning how to play the piano",
        "He learned about economic systems",
        "Snowshoeing",
    ):
        assert not LoCoMoAdapter.is_refusal_shaped(hallucination), hallucination


def test_refusal_detection_accepts_real_abstentions() -> None:
    for refusal in (
        "",
        "No",
        "No.",
        "None",
        "None (not mentioned in conversation).",
        "I don't know.",
        "The context does not contain this information; no information available.",
        "Not mentioned",
        "unknown",
    ):
        assert LoCoMoAdapter.is_refusal_shaped(refusal), refusal


def test_normalize_official_abstention_matches_official_rule() -> None:
    """Only real refusals are rewritten, onto the phrasing the official harness credits."""
    for refusal in ("No", "None", "I don't know.", "unknown"):
        out = LoCoMoAdapter.normalize_official_abstention(refusal).lower()
        assert "no information available" in out or "not mentioned" in out, refusal
    # An answer that already satisfies the official rule keeps the model's wording.
    keep = "No information available (not mentioned in the conversation)."
    assert LoCoMoAdapter.normalize_official_abstention(keep) == keep
    # A genuine (wrong) answer is never turned into an abstention here.
    for content in ("Fantasy novels", "piano", "economic systems", "24 August 2023"):
        assert LoCoMoAdapter.normalize_official_abstention(content) == content, content
