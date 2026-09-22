"""Scale and locality regressions for bounded evidence retrieval."""

from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall.hierarchical_evidence_index import HierarchicalEvidenceIndex


def _record(session: str, text: str) -> StructuredIR:
    return StructuredIR(
        entity="user",
        property="statement",
        value=text,
        raw_content=f"[{session} on 2026-09-21] user: {text}",
        time_scope="2026-09-21",
        source="user",
    )


def _large_corpus() -> list[StructuredIR]:
    records = []
    for idx in range(10_000):
        records.append(_record(f"session-{idx}", f"Routine journal note number {idx}. მშვიდ"))
    # The preceding turn supplies the noun; the target response uses ``it``.
    records[4_200] = _record("target-session", "I bought the sapphire ledger at the stationery store.")
    records[4_201] = _record("target-session", "I put it in the locked desk drawer.")
    return records


def test_index_bounds_work_and_restores_adjacent_turns() -> None:
    records = _large_corpus()
    index = HierarchicalEvidenceIndex(shard_size=64)
    index.build(records)

    found = index.retrieve("Where did I put the sapphire ledger?", max_shards=8, max_records=12)

    text = " ".join(record.raw_content for record in found).lower()
    assert "sapphire ledger" in text
    assert "locked desk drawer" in text
    assert index.last_trace is not None
    assert index.last_trace.visited_shards <= 8
    assert index.last_trace.returned_records <= 12


def test_msc_uses_prebuilt_index_without_full_corpus_scan() -> None:
    records = _large_corpus()
    compiler = MinimumSufficientContextCompiler(
        scalable_retrieval_threshold=10,
        scalable_candidate_budget=24,
    )
    compiler.prepare_scalable_retrieval(records)

    pcc = compiler.compile("Where did I put the sapphire ledger?", records)

    assert "locked desk drawer" in pcc.context_text.lower()
    assert compiler.evidence_index.last_trace is not None
    assert compiler.evidence_index.last_trace.returned_records <= 24


def test_index_caps_a_common_term_posting_list() -> None:
    records = [
        _record(f"session-{idx}", f"meeting note {idx}")
        for idx in range(200)
    ]
    index = HierarchicalEvidenceIndex(shard_size=8, max_posting_shards=5)
    index.build(records)

    index.retrieve("Where was the meeting?", max_shards=3, max_records=8)

    assert index.last_trace is not None
    assert index.last_trace.matched_shards <= 5
    assert index.last_trace.visited_shards <= 3


def test_index_rebuilds_when_a_different_same_length_corpus_arrives() -> None:
    """id() reuse must not let a stale index answer a new question."""
    index = HierarchicalEvidenceIndex(shard_size=64)

    corpus_a = _large_corpus()
    index.build(corpus_a)
    assert index.matches(corpus_a)

    # Same length, different content: the previous id()-based fingerprint
    # could alias this to corpus_a after garbage collection.
    corpus_b = _large_corpus()
    corpus_b[4_200] = _record("target-session", "I filed the amber manifest at the harbor office.")
    corpus_b[4_201] = _record("target-session", "I sent it to the customs agent.")
    assert not index.matches(corpus_b)

    index.build(corpus_b)
    found = index.retrieve("Where did I send the amber manifest?", max_shards=8, max_records=12)
    text = " ".join(record.raw_content for record in found).lower()
    assert "customs agent" in text
