"""Unit tests for StateCompiler in AM Apex Protein Phase (Phase C5 & C6)."""

from __future__ import annotations

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.protein.state_compiler import StateCompiler


def test_state_compiler_origin():
    compiler = StateCompiler()
    records = [
        StructuredIR(
            entity="Caroline",
            property="residence",
            value="Sweden",
            raw_content="[D4:3] Caroline: Gift from grandma in Sweden.",
        ),
    ]

    states = compiler.compile_states("Where did Caroline move from 4 years ago?", records)
    assert len(states) == 1
    st = states[0]
    assert st.subject == "Caroline"
    assert st.property_name == "home country"
    assert st.value == "Sweden"
    assert "supported_by: [D4:3]" in st.format_state()
    assert "[STATE] Caroline's home country is Sweden." in st.format_state()


def test_state_compiler_multi_item():
    compiler = StateCompiler()
    records = [
        StructuredIR(entity="Melanie", property="item", value="figurines", raw_content="[D1:2] Melanie bought figurines"),
        StructuredIR(entity="Melanie", property="item", value="shoes", raw_content="[D1:4] Melanie bought shoes"),
    ]

    states = compiler.compile_states("What items has Melanie bought?", records)
    assert len(states) == 1
    st = states[0]
    assert "figurines" in st.value
    assert "shoes" in st.value
    assert "include figurines, shoes" in st.format_state()


def test_state_compiler_count():
    compiler = StateCompiler()
    records = [
        StructuredIR(entity="Melanie", property="kids", value="3 kids", raw_content="[D2:1] Melanie has 3 children"),
    ]

    states = compiler.compile_states("How many children does Melanie have?", records)
    assert len(states) == 1
    st = states[0]
    assert st.value == "3"
    assert "[STATE] Melanie's children count is 3." in st.format_state()
