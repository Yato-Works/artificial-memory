"""Unit tests for StateTrigger in AM Apex Protein Phase."""

from __future__ import annotations

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.protein.state_trigger import StateTrigger


def test_state_trigger_multihop_positive():
    trigger = StateTrigger()
    records = [
        StructuredIR(entity="Caroline", property="origin", value="Sweden", raw_content="[D4:3] Sweden"),
    ]

    dec = trigger.evaluate("Where did Caroline move from 4 years ago?", records)
    assert dec.should_synthesize is True
    assert "composition pattern" in dec.trigger_reason


def test_state_trigger_single_hop_negative():
    trigger = StateTrigger()
    records = [
        StructuredIR(entity="Melanie", property="color", value="blue", raw_content="[D1:1] My favorite color is blue"),
    ]

    dec = trigger.evaluate("What is Melanie's favorite color?", records)
    assert dec.should_synthesize is False
    assert "Simple single-turn" in dec.trigger_reason
