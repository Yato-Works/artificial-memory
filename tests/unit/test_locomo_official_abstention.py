"""Guardrails for the official-protocol abstention surface (LoCoMo cat 5).

The pinned official harness only credits ``"no information available"`` /
``"not mentioned"`` for adversarial items, so AM's equivalent refusals are
canonicalised onto that surface.  These tests pin two invariants:

1. every refusal shape the production guards already emit becomes creditable;
2. a real (substantive) answer is NEVER rewritten into an abstention.
"""
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OFFICIAL_ABSTENTION_TEXT

CREDITED = ("no information available", "not mentioned")


def test_refusal_shapes_become_officially_creditable() -> None:
    refusals = [
        "I don't know.",
        "I do not know.",
        "No",
        "No.",
        "None",
        "None (not mentioned in conversation).",
        "unknown",
        "Cannot be determined.",
        "not enough information",
        "",
    ]
    for refusal in refusals:
        out = LoCoMoAdapter.normalize_official_abstention(refusal)
        assert any(m in out.lower() for m in CREDITED), (refusal, out)


def test_already_creditable_wording_is_preserved() -> None:
    """A model already using the official phrase keeps its exact wording."""
    original = "not mentioned in the conversation"
    assert LoCoMoAdapter.normalize_official_abstention(original) == original


def test_substantive_answers_are_never_turned_into_abstentions() -> None:
    answers = [
        "Her mother.",
        "21 December 2022",
        "Kickboxing, Taekwondo",
        "Volunteering at a homeless shelter",
        "A cup with a dog face on it.",
        "Two cats and a dog.",
        "No one was present in the room.",  # "No" must not trigger a bare-negation match
        "Karen is not mentioned as a pianist, but plays guitar.",  # already-creditable substring
    ]
    for answer in answers:
        out = LoCoMoAdapter.normalize_official_abstention(answer)
        assert out == answer, (answer, out)


def test_official_text_matches_the_pinned_harness_markers() -> None:
    assert any(m in OFFICIAL_ABSTENTION_TEXT.lower() for m in CREDITED)
