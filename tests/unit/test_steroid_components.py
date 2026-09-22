"""Unit tests for AM Apex Steroid Phase components."""

import pytest
from artificial_memory.core.ir.structured import StructuredIR, IRStatus, IRRelation
from artificial_memory.steroid.wide_slicer import WideSlicer
from artificial_memory.steroid.adaptive_graph_expander import AdaptiveGraphExpander
from artificial_memory.steroid.evidence_evaluator import EvidenceEvaluator
from artificial_memory.steroid.steroid_compiler import SteroidContextCompiler


def test_wide_slicer_multi_channel():
    slicer = WideSlicer(per_channel_budget=5)
    records = [
        StructuredIR(entity="Alice", property="location", value="Tokyo", raw_content="[D1] Alice: I arrived in Tokyo today.", time_scope="2023-01-10"),
        StructuredIR(entity="Bob", property="camera", value="Sony A7R", raw_content="[D2] Bob: I use a Sony A7R camera.", time_scope="2023-01-11"),
        StructuredIR(entity="Caroline", property="trip", value="Sweden", raw_content="[D3] Caroline: I visited Sweden last month.", time_scope="2023-01-12"),
        StructuredIR(entity="Dave", property="project", value="Atlas", raw_content="[D4] Dave: Working on project Atlas.", time_scope="2023-01-13"),
    ]

    res = slicer.slice("What camera does Bob use in Tokyo?", records)
    assert res.total_unioned >= 2
    contents = " ".join(r.raw_content for r in res.candidate_records)
    assert "Sony A7R" in contents
    assert "Tokyo" in contents


def test_adaptive_graph_expander_early_exit():
    expander = AdaptiveGraphExpander(max_hops=3)
    records = [
        StructuredIR(entity="Alice", property="job", value="Engineer", raw_content="Alice is a senior engineer at Google."),
        StructuredIR(entity="Alice", property="degree", value="CS", raw_content="Alice graduated with a CS degree."),
        StructuredIR(entity="Alice", property="city", value="Seattle", raw_content="Alice lives in Seattle."),
        StructuredIR(entity="Alice", property="team", value="AI", raw_content="Alice works in the AI team."),
        StructuredIR(entity="Alice", property="laptop", value="MacBook", raw_content="Alice uses a MacBook."),
        StructuredIR(entity="Alice", property="hobby", value="Hiking", raw_content="Alice likes hiking."),
    ]
    # Complete evidence -> early exit
    res = expander.expand("What is Alice's job?", records[:5], records)
    assert res.hops_performed == 0


def test_adaptive_graph_expander_with_unresolved_referent():
    expander = AdaptiveGraphExpander(max_hops=3)
    initial = [
        StructuredIR(entity="Caroline", property="origin", value="home country", raw_content="Caroline moved from her home country.")
    ]
    corpus = [
        StructuredIR(entity="Caroline", property="origin", value="home country", raw_content="Caroline moved from her home country."),
        StructuredIR(entity="Caroline", property="country", value="Sweden", raw_content="Caroline grew up in Sweden, which is her home country."),
    ]
    res = expander.expand("Where was Caroline born?", initial, corpus)
    assert res.hops_performed >= 1
    contents = " ".join(r.raw_content for r in res.evidence_pool)
    assert "Sweden" in contents


def test_steroid_compiler_compilation():
    compiler = SteroidContextCompiler()
    records = [
        StructuredIR(entity="Melanie", property="trip", value="Kyoto", raw_content="[D1:1] Melanie: I went to Kyoto with my sister.", time_scope="2023-05-01"),
        StructuredIR(entity="Melanie", property="sister", value="Chloe", raw_content="[D1:2] Melanie: My sister Chloe loved the temples.", time_scope="2023-05-02"),
    ]
    pcc = compiler.compile("Who is Melanie's sister?", records)
    assert not pcc.is_abstention
    assert "Chloe" in pcc.context_text
