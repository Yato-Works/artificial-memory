"""Unit tests for AM Apex Protein Phase components."""

import pytest
from artificial_memory.core.ir.structured import StructuredIR, IRStatus, IRRelation
from artificial_memory.protein.evidence_fuser import EvidenceFuser
from artificial_memory.protein.temporal_supersession_protein import TemporalSupersessionProtein
from artificial_memory.protein.evidence_ranker import EvidenceRanker
from artificial_memory.protein.context_ir_compressor import ContextIRCompressor
from artificial_memory.protein.provenance_tracker import ProvenanceTracker
from artificial_memory.protein.protein_compiler import ProteinContextCompiler


def test_evidence_fuser_deduplication():
    fuser = EvidenceFuser()
    records = [
        StructuredIR(entity="Alice", property="city", value="Osaka", raw_content="Alice lives in Osaka."),
        StructuredIR(entity="Alice", property="city", value="Osaka", raw_content="Alice lives in Osaka."),  # Duplicate
        StructuredIR(entity="Mel", property="hobby", value="pottery", raw_content="Mel likes pottery."),
        StructuredIR(entity="Melanie", property="hobby", value="pottery", raw_content="Melanie likes pottery."),  # Alias duplicate
    ]
    fused = fuser.fuse("Where does Alice live?", records)
    assert len(fused) == 2
    contents = " ".join(r.raw_content for r in fused)
    assert "Osaka" in contents
    assert "pottery" in contents


def test_temporal_supersession_ordering():
    supersession = TemporalSupersessionProtein()
    records = [
        StructuredIR(entity="Alice", property="city", value="Tokyo", raw_content="Alice lives in Tokyo.", time_scope="2024-01-01"),
        StructuredIR(entity="Alice", property="city", value="Osaka", raw_content="Alice moved to Osaka.", time_scope="2025-04-10"),
        StructuredIR(entity="Alice", property="city", value="Okinawa", raw_content="Alice now lives in Okinawa.", time_scope="2026-06-01"),
    ]
    active, groups = supersession.resolve("Where does Alice live now?", records)
    assert len(active) == 3
    assert active[-1].value == "Okinawa"
    assert active[-1].status == IRStatus.ACTIVE
    assert len(groups) == 1
    assert len(groups[0].superseded_records) == 2
    assert groups[0].superseded_records[0].value == "Tokyo"
    assert groups[0].superseded_records[1].value == "Osaka"


def test_evidence_ranker_multi_objective():
    ranker = EvidenceRanker(top_k=2)
    records = [
        StructuredIR(entity="Bob", property="camera", value="Sony", raw_content="Bob bought a Sony camera yesterday.", time_scope="2023-01-05"),
        StructuredIR(entity="Alice", property="lunch", value="sandwich", raw_content="Alice ate a sandwich.", time_scope="2023-01-02"),
        StructuredIR(entity="Bob", property="trip", value="Kyoto", raw_content="Bob traveled to Kyoto.", time_scope="2023-01-04"),
    ]
    ranked = ranker.rank("What camera did Bob buy?", records, target_entity="bob")
    assert len(ranked) == 2
    # Bob's camera should be ranked #1
    assert ranked[0].record.property == "camera"
    assert ranked[0].record.value == "Sony"


def test_context_ir_compressor():
    compressor = ContextIRCompressor()
    records = [
        StructuredIR(entity="Caroline", property="sister", value="Chloe", raw_content="Yeah well haha my sister is Chloe."),
        StructuredIR(entity="Caroline", property="pet", value="dog", raw_content="I have a pet dog named Max."),
    ]
    compressed = compressor.compress(records, target_token_budget=100)
    assert "[FACT] Caroline -> sister: Chloe" in compressed
    assert "[FACT] Caroline -> pet: dog" in compressed
    assert "haha" not in compressed  # Filler stripped


def test_provenance_tracker():
    tracker = ProvenanceTracker()
    records = [
        StructuredIR(entity="Melanie", property="gift", value="necklace", raw_content="[D1:5] Melanie: My grandma gave me a necklace.", time_scope="2023-05-10")
    ]
    prov = tracker.track("Melanie's necklace", records, hop_distance=1)
    assert prov.claim_id.startswith("claim-")
    assert "D1:5" in prov.source_turns
    assert prov.hop_distance == 1
    assert prov.confidence == 0.85


def test_protein_compiler_integration():
    compiler = ProteinContextCompiler()
    records = [
        StructuredIR(entity="Evan", property="trip", value="Jasper", raw_content="[D1] Evan traveled to Jasper with Sam.", time_scope="2023-03-01"),
        StructuredIR(entity="Evan", property="city", value="Calgary", raw_content="[D2] Evan lives in Calgary.", time_scope="2023-03-02"),
    ]
    pcc = compiler.compile("Where did Evan travel?", records)
    assert not pcc.is_abstention
    assert "Jasper" in pcc.context_text
    assert pcc.token_cost <= 150


def test_adaptive_context_compression():
    from artificial_memory.protein.protein_compiler import ContextPolicy
    compressor = ContextIRCompressor()
    records = [
        StructuredIR(entity="Evan", property="trip", value="Jasper", raw_content="[D1] Evan traveled to Jasper with Sam.", time_scope="2023-03-01"),
        StructuredIR(entity="Evan", property="city", value="Calgary", raw_content="[D2] Yeah well Evan lives in Calgary.", time_scope="2023-03-02"),
    ]
    adaptive_out = compressor.compress_adaptive(records, target_token_budget=100)
    assert "[D1] Evan traveled to Jasper with Sam." in adaptive_out

    # Test compiler with ADAPTIVE and COMPACT policies
    compiler_adaptive = ProteinContextCompiler(policy=ContextPolicy.ADAPTIVE)
    pcc_adapt = compiler_adaptive.compile("Where did Evan travel?", records)
    assert "Jasper" in pcc_adapt.context_text

    compiler_compact = ProteinContextCompiler(policy=ContextPolicy.COMPACT)
    pcc_comp = compiler_compact.compile("Where did Evan travel?", records)
    assert "Jasper" in pcc_comp.context_text


def test_graph_chain_retention():
    from artificial_memory.protein.chain_retention import GraphChainRetainer

    retainer = GraphChainRetainer(max_chain_reserve=2)
    # Hop 1: Caroline researched adoption agencies
    # Hop 2: The adoption agency is named St. Jude
    r1 = StructuredIR(entity="Caroline", property="research", value="Adoption Agency", raw_content="[D1] Caroline researched an Adoption Agency.")
    r2 = StructuredIR(entity="Adoption Agency", property="name", value="St. Jude", raw_content="[D2] The Adoption Agency is named St. Jude.")
    r3 = StructuredIR(entity="Bob", property="car", value="Tesla", raw_content="[D3] Bob bought a Tesla.")

    # r1 is in top_records, r2 is a candidate outside top_records
    top_records = [r1]
    candidate_records = [r1, r2, r3]

    retained = retainer.retain_chains("What is the name of the agency Caroline researched?", top_records, candidate_records)
    # r2 should be retained via forward hop (Adoption Agency in r1.value matches r2.entity)
    assert len(retained) == 2
    assert r2 in retained
    assert r3 not in retained


