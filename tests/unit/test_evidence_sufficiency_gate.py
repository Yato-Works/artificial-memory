"""Unit tests for Evidence Sufficiency Gate and Associative Graph Navigation (Phase X.6)."""

import pytest

from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.core.ir.memory_types import ApexMemoryUnit, MemoryRole
from artificial_memory.recall.evidence_graph import EvidenceGraph, extract_informative_phrases
from artificial_memory.recall.evidence_sufficiency_gate import EvidenceSufficiencyGate
from artificial_memory.recall.state_reconstructor import StateReconstructor


def test_informative_phrase_extraction() -> None:
    """Verify that stopwords and conversational filler are stripped from phrases."""
    text1 = "Caroline: Yeah, I'm really lucky to have them. I moved here from my home country 4 years ago."
    text2 = "Caroline: This necklace is a gift from my grandma in my home country, Sweden."

    p1 = extract_informative_phrases(text1)
    p2 = extract_informative_phrases(text2)

    assert "home country" in p1
    assert "home country" in p2
    assert "home country" in (p1 & p2)

    # Filler bigrams must not be extracted
    assert "yeah i" not in p1
    assert "to have" not in p1
    assert "is a" not in p2


def test_evidence_sufficiency_gate_triggers_on_unresolved_referent() -> None:
    """Verify that generic hypernyms trigger the gate when unresolved."""
    gate = EvidenceSufficiencyGate()
    extractor = UniversalIRExtractor()

    records = extractor.extract(
        "[D3:13 on 2023-06-09] Caroline: I've known these friends since I moved here from my home country 4 years ago.",
        default_source="caroline",
    )
    units = [
        ApexMemoryUnit(ir=r, role=MemoryRole.EVIDENCE)
        for r in records
    ]

    # Query asking for country
    query = "Where did Caroline move from 4 years ago?"
    decision = gate.check_sufficiency(query, units)

    assert decision.is_sufficient is False
    assert decision.unresolved_phrase == "home country"
    assert decision.expected_category == "country"


def test_state_reconstructor_pulls_second_hop_via_sufficiency_gate() -> None:
    """Verify that StateReconstructor pulls the 2nd hop node into top ranks when gate triggers."""
    reconstructor = StateReconstructor()
    extractor = UniversalIRExtractor()

    dialogue = [
        "[D1:1 on 2023-05-08] Caroline: Just checking in about our schedule.",
        "[D2:1 on 2023-05-25] Caroline: I had a busy week at work.",
        "[D3:13 on 2023-06-09] Caroline: I've known these friends since I moved here from my home country 4 years ago.",
        "[D4:1 on 2023-06-27] Caroline: Good morning Melanie!",
        "[D4:3 on 2023-06-27] Caroline: This necklace is a gift from my grandma in my home country, Sweden.",
        "[D5:1 on 2023-07-03] Caroline: Hope you have a wonderful weekend.",
    ]

    records = []
    for line in dialogue:
        records.extend(extractor.extract(line, default_source="caroline"))

    query = "Where did Caroline move from 4 years ago?"
    intent, candidate_units = reconstructor.reconstruct_world(query, records)

    top2_text = " ".join(u.ir.raw_content for u in candidate_units[:2])
    assert "D3:13" in top2_text
    assert "D4:3" in top2_text
    assert "sweden" in top2_text.lower()
