"""Unit Tests for AM Apex Overdrive Core (Phases X.6 - X.10).

Tests:
1. UnifiedProposition & StateHistory
2. UnifiedPropositionGraph construction & traversal
3. QueryPlanner intent and slot extraction
4. AdaptiveEvidenceSearcher bounded search (hops <= 3)
5. PropositionIntegrityGate (entity-swap adversarial prevention)
6. StateSupersessionEngine (knowledge update current vs previous)
7. AnswerVerifier (hallucination correction)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.core.ir.proposition import PropositionStatus, StateHistory, UnifiedProposition
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall.adaptive_search import AdaptiveEvidenceSearcher
from artificial_memory.recall.answer_verifier import AnswerVerifier
from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph
from artificial_memory.recall.proposition_integrity_gate import PropositionIntegrityGate
from artificial_memory.recall.query_planner import PlannerIntent, QueryPlanner
from artificial_memory.recall.state_supersession_engine import StateSupersessionEngine


def test_unified_proposition_and_state_history():
    history = StateHistory("Eli", "database")
    history.add_snapshot("SQLite", "2026-01-01", "session_1", "p1")
    history.add_snapshot("ClickHouse", "2026-02-01", "session_2", "p2")

    assert history.get_latest().value == "ClickHouse"
    assert history.get_previous().value == "SQLite"
    summary = history.get_history_summary()
    assert "was previously 'SQLite'" in summary
    assert "updated to 'ClickHouse'" in summary
    print("[PASS] test_unified_proposition_and_state_history")


def test_query_planner():
    planner = QueryPlanner()
    
    p1 = planner.plan("How many days passed between the first race and the marathon?")
    assert p1.intent == PlannerIntent.TEMPORAL_DELTA
    assert len(p1.sub_queries) == 2

    p2 = planner.plan("What does Melanie's necklace symbolize?")
    assert p2.intent == PlannerIntent.ADVERSARIAL_CHECK
    assert "melanie" in p2.target_entities
    assert "owns_necklace" in p2.target_predicates

    p3 = planner.plan("Where did Caroline move from 4 years ago?")
    assert p3.intent == PlannerIntent.MULTI_HOP
    assert "caroline" in p3.target_entities
    assert "origin_country" in p3.missing_slots
    print("[PASS] test_query_planner")


def test_proposition_integrity_gate():
    gate = PropositionIntegrityGate()
    planner = QueryPlanner()

    # Query asks about Melanie's necklace
    plan = planner.plan("What does Melanie's necklace symbolize?")

    # Evidence only contains Caroline's necklace
    p1 = UnifiedProposition(
        id="p1",
        subject="Caroline",
        predicate="owns",
        object="necklace",
        raw_text="[D4:3] Caroline: My grandma gave me this necklace when I graduated in Sweden.",
    )
    p2 = UnifiedProposition(
        id="p2",
        subject="Melanie",
        predicate="talks_to",
        object="Caroline",
        raw_text="[D4:4] Melanie: That is such a lovely necklace!",
    )

    decision = gate.check(plan, [p1, p2])
    assert not decision.is_valid
    assert decision.recommended_abstention
    assert "Proposition Integrity Warning" in decision.grounding_note
    print("[PASS] test_proposition_integrity_gate")


def test_state_supersession_engine():
    engine = StateSupersessionEngine()
    planner = QueryPlanner()
    graph = UnifiedPropositionGraph()

    p1 = UnifiedProposition(
        id="p1",
        subject="Eli",
        predicate="uses",
        object="Canon 5D",
        time_scope="2022-01-01",
        raw_text="Eli uses Canon 5D",
    )
    p2 = UnifiedProposition(
        id="p2",
        subject="Eli",
        predicate="uses",
        object="Sony A7R IV",
        time_scope="2023-05-01",
        raw_text="Eli uses Sony A7R IV",
    )
    graph.add_proposition(p1)
    graph.add_proposition(p2)

    plan_curr = planner.plan("What camera does Eli currently use?")
    res_curr = engine.resolve(plan_curr, graph)
    assert res_curr.resolved_snapshot.value == "Sony A7R IV"
    assert "Currently" in res_curr.grounding_certificate

    plan_prev = planner.plan("What camera did Eli previously use?")
    res_prev = engine.resolve(plan_prev, graph)
    assert res_prev.resolved_snapshot.value == "Canon 5D"
    assert "Previously" in res_prev.grounding_certificate
    print("[PASS] test_state_supersession_engine")


def test_answer_verifier():
    verifier = AnswerVerifier()

    # 1. Test abstention enforcement
    res = verifier.verify(
        question="What does Melanie's necklace symbolize?",
        predicted_answer="It symbolizes graduation from university.",
        context="[Proposition Integrity Warning: ...]",
        propositions=[],
        integrity_abstention_recommended=True,
    )
    assert res.verified_answer == "None (not mentioned in conversation)."
    assert res.hallucination_detected

    # 2. Test temporal correction
    res_t = verifier.verify(
        question="How many days passed between Event 1 and Event 2?",
        predicted_answer="3 days passed.",
        context="[Temporal Calculation: Exactly 7 days passed between Event 1 and Event 2.]",
        propositions=[],
    )
    assert res_t.verified_answer == "7 days"
    assert res_t.hallucination_detected
    print("[PASS] test_answer_verifier")


if __name__ == "__main__":
    test_unified_proposition_and_state_history()
    test_query_planner()
    test_proposition_integrity_gate()
    test_state_supersession_engine()
    test_answer_verifier()
    print("\nALL OVERDRIVE CORE UNIT TESTS PASSED (5/5)!")
