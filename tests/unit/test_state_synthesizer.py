"""Unit tests for StateSynthesizer in AM Apex Protein Phase."""

from __future__ import annotations

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.protein.state_synthesizer import StateSynthesizer


def test_state_synthesizer_sweden():
    synth = StateSynthesizer()
    records = [
        StructuredIR(
            entity="Caroline",
            property="roots",
            value="Sweden",
            raw_content="[D4:3] Caroline: This necklace is a gift from my grandma in my home country, Sweden.",
        ),
        StructuredIR(
            entity="Caroline",
            property="residence",
            value="4 years",
            raw_content="[D3:13] Caroline: I've known these friends for 4 years, since I moved from my home country.",
        ),
    ]

    states = synth.synthesize("Where did Caroline move from 4 years ago?", records)
    assert len(states) == 1
    assert states[0].entity.lower() == "caroline"
    assert states[0].target_property == "home country"
    assert states[0].value == "Sweden"
    assert states[0].format_state() == "[STATE] Caroline's home country is Sweden."


def test_state_synthesizer_relationship():
    synth = StateSynthesizer()
    records = [
        StructuredIR(
            entity="Caroline",
            property="family",
            value="single parent",
            raw_content="[D2:14] Caroline: It'll be tough as a single parent, but I'm up for the challenge!",
        ),
    ]

    states = synth.synthesize("What is Caroline's relationship status?", records)
    assert len(states) == 1
    assert states[0].value == "single"
    assert states[0].format_state() == "[STATE] Caroline's relationship status is single."


def test_state_synthesizer_no_match():
    synth = StateSynthesizer()
    records = [
        StructuredIR(
            entity="Melanie",
            property="weather",
            value="sunny",
            raw_content="[D1:1] Melanie: It was very sunny today.",
        ),
    ]

    states = synth.synthesize("What is Caroline's favorite color?", records)
    assert len(states) == 0
