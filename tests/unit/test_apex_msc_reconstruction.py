"""Unit tests for Apex Memory Runtime: MSC & State Reconstruction (Apex Phase A & B)."""

import pytest

from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.core.ir.memory_types import QueryIntent
from artificial_memory.recall.state_reconstructor import StateReconstructor


def test_state_reconstruction_intent_slicing() -> None:
    """Verify that StateReconstructor slices the world correctly based on query intent."""
    reconstructor = StateReconstructor()
    extractor = UniversalIRExtractor()

    dialogue = [
        "In yesterday's meeting, the team had a long discussion about Redis.",
        "Alice proposed: 'We must use Redis for the queue.'",
        "The proposed Redis migration was officially abandoned due to memory overhead.",
        "For project Orion, the active database is PostgreSQL.",
        "For project Orion, the user always reverts the build configuration to the last known good state.",
    ]

    records = []
    for line in dialogue:
        records.extend(extractor.extract(line, default_source="user"))

    # 1. State Query: Should pick active PostgreSQL and discard abandoned Redis
    intent, state_slice = reconstructor.reconstruct_world("What is the current active database for project Orion?", records)
    assert intent == QueryIntent.STATE_QUERY
    state_text = " ".join(u.ir.raw_content for u in state_slice).lower()
    assert "postgresql" in state_text
    assert "abandoned" not in state_text

    # 2. Evidence Query: Should pick Alice's proposal
    intent, ev_slice = reconstructor.reconstruct_world("Did Alice propose using Redis for the queue at any point?", records)
    assert intent == QueryIntent.EVIDENCE_QUERY
    ev_text = " ".join(u.ir.raw_content for u in ev_slice).lower()
    assert "alice proposed" in ev_text

    # 3. Reflection Query: Should pick behavioral tendency
    intent, refl_slice = reconstructor.reconstruct_world("Considering everything, what recurring preference does the user show?", records)
    assert intent == QueryIntent.REFLECTION_QUERY
    refl_text = " ".join(u.ir.raw_content for u in refl_slice).lower()
    assert "reverts" in refl_text


def test_msc_coverage_and_recovery() -> None:
    """Verify that MSC Compiler automatically recovers omitted evidence when coverage fails."""
    compiler = MinimumSufficientContextCompiler()
    extractor = UniversalIRExtractor()

    dialogue = [
        "For project Chrono, the team discussed general scheduling.",
        "For project Chrono, the team reviewed the backlog.",
        "For project Chrono, lunch was ordered at noon.",
        "On 2026-04-10, for project Chrono the active database was ClickHouse.",
        "On 2026-06-01, for project Chrono the database was migrated to PostgreSQL.",
    ]

    records = []
    for line in dialogue:
        records.extend(extractor.extract(line, default_source="user"))

    # Query with strict temporal constraint: 2026-04-10
    query = "Which database was active on 2026-04-10 for project Chrono?"
    pcc = compiler.compile(query, records)

    # Verify coverage check and recovery
    assert pcc.certificate.is_sufficient is True
    assert pcc.certificate.temporal_coverage is True
    assert "clickhouse" in pcc.context_text.lower()


def test_proof_carrying_context_certificate() -> None:
    """Verify Proof-Carrying Context certificate formatting."""
    compiler = MinimumSufficientContextCompiler()
    extractor = UniversalIRExtractor()

    dialogue = [
        "For project Atlas, the team selected ClickHouse as the primary database.",
    ]
    records = []
    for line in dialogue:
        records.extend(extractor.extract(line, default_source="user"))

    pcc = compiler.compile("Which database was selected for Project Atlas?", records)
    assert pcc.certificate.is_sufficient is True
    cert_str = pcc.certificate.format_certificate()

    assert "[COVERAGE CERTIFICATE: VERIFIED_SUFFICIENT]" in cert_str
    assert "Entity Scope    : MATCHED" in cert_str
    assert "Property/Aspect : MATCHED" in cert_str
