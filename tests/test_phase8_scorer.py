"""Scorer unit + replay tests (Phase 8.3.2/8.3.3, Benchmark_Plan.txt §10).

§10 requires the Scorer to be fixed with synthetic cases *before* any
production run, so a scorer bug can never silently distort a player's result:
all required groups -> pass, one of two -> partial / 0.5, forbidden hit ->
fail / 0.0, assertion without evidence -> false_positive, abstention ->
pass, case differences -> pass via normalization, compound terms are
order-fixed, ``trust`` must not match ``rust``, synonym groups count.
"""

import csv
import json

import pytest

from artificial_memory.research.benchmarks.arena import (
    ArenaCategory,
    ArenaQuestion,
    build_dataset,
)
from artificial_memory.research.benchmarks.player import Answer, ControlledPlayer
from artificial_memory.research.benchmarks.runner import ArenaRunner, RunConfig
from artificial_memory.research.benchmarks.scorer import (
    OUTCOME_FAIL,
    OUTCOME_FALSE_POSITIVE,
    OUTCOME_PARTIAL,
    OUTCOME_PASS,
    SCORE_CSV_COLUMNS,
    QuestionScore,
    build_aggregate,
    detect_abstention,
    normalize_text,
    player_aggregates,
    score_answer,
    score_answer_repeats,
    score_run_dir,
    term_present,
)


def make_question(**overrides) -> ArenaQuestion:
    base = dict(
        question_id="factual_recall-0",
        category=ArenaCategory.FACTUAL_RECALL,
        question="Which language did the user pick?",
        ground_truth=(("rust",),),
        forbidden=(),
        expected_abstention=False,
    )
    base.update(overrides)
    return ArenaQuestion(**base)


# ==================== §10 synthetic case table ====================


class TestSyntheticCases:
    def test_all_groups_present_pass(self):
        q = make_question(ground_truth=(("rust",), ("kyoto",)))
        s = score_answer(q, "The user picked rust while working in kyoto.")
        assert s.outcome == OUTCOME_PASS and s.score == 1.0 and s.correct

    def test_one_group_missing_partial(self):
        q = make_question(ground_truth=(("rust",), ("kyoto",)))
        s = score_answer(q, "The user picked rust.")
        assert s.outcome == OUTCOME_PARTIAL and s.score == 0.5

    def test_forbidden_hit_fail(self):
        q = make_question(ground_truth=(("rust",),), forbidden=("go",))
        s = score_answer(q, "The user picked rust, not go.")
        assert s.outcome == OUTCOME_FAIL and s.score == 0.0
        assert s.forbidden_violation and s.forbidden_hits == ("go",)

    def test_forbidden_zeroes_score_even_when_covered(self):
        q = make_question(ground_truth=(("rust",), ("kyoto",)), forbidden=("go",))
        s = score_answer(q, "rust and kyoto, because of go.")
        assert s.outcome == OUTCOME_FAIL and s.score == 0.0
        assert s.ground_truth_coverage == 1.0  # coverage kept visible, score zeroed

    def test_abstention_asserted_value_false_positive(self):
        q = make_question(
            category=ArenaCategory.ABSTENTION, expected_abstention=True, forbidden=("ripgrep",)
        )
        s = score_answer(q, "The user set the debugger to ripgrep.")
        assert s.outcome == OUTCOME_FALSE_POSITIVE and s.score == 0.0

    def test_abstention_pass(self):
        q = make_question(
            category=ArenaCategory.ABSTENTION, expected_abstention=True, forbidden=("ripgrep",)
        )
        s = score_answer(q, "I don't know; there is no information in the context.")
        assert s.outcome == OUTCOME_PASS and s.score == 1.0 and s.abstention_detected

    def test_abstention_with_forbidden_still_false_positive(self):
        q = make_question(
            category=ArenaCategory.ABSTENTION, expected_abstention=True, forbidden=("ripgrep",)
        )
        s = score_answer(q, "I don't know, but it is probably ripgrep.")
        assert s.outcome == OUTCOME_FALSE_POSITIVE and s.score == 0.0

    def test_case_insensitive(self):
        q = make_question(ground_truth=(("rust",),))
        s = score_answer(q, "The user picked RUST.")
        assert s.outcome == OUTCOME_PASS

    def test_compound_term_order_fixed(self):
        q = make_question(ground_truth=(("visual studio code",),))
        assert score_answer(q, "they use visual studio code").outcome == OUTCOME_PASS
        assert score_answer(q, "they use code visual studio").outcome == OUTCOME_FAIL

    def test_word_boundary(self):
        q = make_question(ground_truth=(("rust",),))
        s = score_answer(q, "The user picked trust for reliability.")
        assert s.outcome == OUTCOME_FAIL

    def test_synonym_group(self):
        q = make_question(ground_truth=(("visual studio code", "vs code"),))
        s = score_answer(q, "they use vs code daily")
        assert s.outcome == OUTCOME_PASS

    def test_symbol_term(self):
        q = make_question(ground_truth=(("f#",),))
        assert score_answer(q, "the user writes f# scripts").outcome == OUTCOME_PASS
        assert score_answer(q, "the user writes f sharp scripts").outcome == OUTCOME_FAIL

    def test_empty_answer_fail(self):
        q = make_question(category=ArenaCategory.ABSTENTION, expected_abstention=True)
        s = score_answer(q, "   ")
        assert s.outcome == OUTCOME_FAIL and s.score == 0.0
        assert s.answer_empty and s.abstention_detected is None

    def test_abstention_lexicon(self):
        assert detect_abstention("There is no information about that.")
        assert detect_abstention("I don't know.")
        assert not detect_abstention("The user picked rust.")


class TestNormalization:
    def test_nfkc_and_whitespace(self):
        assert normalize_text("ＲＵＳＴ   kyoto") == "rust kyoto"

    def test_term_present_boundaries(self):
        assert term_present("rust", normalize_text("rust is great"))
        assert not term_present("rust", normalize_text("trust issues"))
        assert not term_present("rust", normalize_text("industrial"))
        assert term_present("visual studio code", normalize_text("i use visual studio code!"))


# ==================== Repeat reduction (§10 決定性) ====================


class TestRepeatReduction:
    def test_majority_answer_is_scored(self):
        q = make_question(ground_truth=(("rust",),))
        s = score_answer_repeats(q, ["rust it is", "unknown", "rust it is"])
        assert s.outcome == OUTCOME_PASS
        assert s.repeat_outcomes == (OUTCOME_PASS, OUTCOME_FAIL, OUTCOME_PASS)
        assert s.repeat_scores == (1.0, 0.0, 1.0)
        assert s.score_consistency == pytest.approx(2 / 3)
        assert s.answer_text == "rust it is"

    def test_tie_break_takes_smallest_index(self):
        q = make_question(ground_truth=(("rust",),))
        s = score_answer_repeats(q, ["rust", "go rocks"])
        assert s.answer_text == "rust"  # 1-1 tie -> earliest repeat wins

    def test_empty_repeat_list(self):
        q = make_question()
        s = score_answer_repeats(q, [])
        assert s.outcome == OUTCOME_FAIL and s.answer_empty

    def test_pure_function_same_input_same_output(self):
        q = make_question(ground_truth=(("rust",),), forbidden=("go",))
        a = score_answer(q, "rust because of go", "P")
        b = score_answer(q, "rust because of go", "P")
        assert a.to_dict() == b.to_dict()


# ==================== Aggregates (§10 集約メトリクス) ====================


def make_score(outcome: str, category: str = "factual_recall", **kw) -> QuestionScore:
    score = 1.0 if outcome == OUTCOME_PASS else (0.5 if outcome == OUTCOME_PARTIAL else 0.0)
    return QuestionScore(
        question_id=kw.pop("question_id", "q"),
        category=category,
        player_name=kw.pop("player_name", "P"),
        answer_text=kw.pop("answer_text", "answer"),
        outcome=outcome,
        score=kw.pop("score", score),
        correct=outcome == OUTCOME_PASS,
        covered_groups=kw.pop("covered_groups", 1),
        must_total=kw.pop("must_total", 1),
        ground_truth_coverage=kw.pop("ground_truth_coverage", score),
        forbidden_violation=kw.pop("forbidden_violation", False),
    )


class TestAggregates:
    def test_math(self):
        scores = [
            make_score(OUTCOME_PASS, "factual_recall"),
            make_score(OUTCOME_PARTIAL, "multi_session_recall"),
            make_score(OUTCOME_FAIL, "temporal_reasoning", forbidden_violation=True),
            make_score(OUTCOME_PASS, ArenaCategory.ABSTENTION.value),
            make_score(OUTCOME_FALSE_POSITIVE, ArenaCategory.ABSTENTION.value),
        ]
        agg = build_aggregate("P", scores)
        assert agg.question_count == 5
        assert agg.answer_accuracy == pytest.approx(2 / 5)
        assert agg.partial_rate == pytest.approx(1 / 5)
        assert agg.false_positive_rate == pytest.approx(1 / 5)
        assert agg.forbidden_violation_rate == pytest.approx(1 / 5)
        assert agg.abstention_question_count == 2
        assert agg.abstention_accuracy == pytest.approx(0.5)
        assert agg.category_accuracy["factual_recall"] == 1.0
        assert agg.category_accuracy[ArenaCategory.ABSTENTION.value] == 0.5
        assert agg.mean_score == pytest.approx((1 + 0.5 + 0 + 1 + 0) / 5)

    def test_no_abstention_questions_gives_none(self):
        agg = build_aggregate("P", [make_score(OUTCOME_PASS)])
        assert agg.abstention_accuracy is None
        assert agg.abstention_question_count == 0

    def test_player_grouping_preserves_order(self):
        scores = [
            make_score(OUTCOME_PASS, player_name="B"),
            make_score(OUTCOME_FAIL, player_name="A"),
            make_score(OUTCOME_PASS, player_name="B"),
        ]
        aggs = player_aggregates(scores)
        assert [a.player_name for a in aggs] == ["B", "A"]
        assert aggs[0].question_count == 2


# ==================== Replay determinism (8.3.3: traces -> same score) ====================


class ReplayStubPlayer(ControlledPlayer):
    """Deterministic stub whose answers exercise the scorer end-to-end."""

    def __init__(self, config: dict | None = None):
        super().__init__("Stub", config)

    def ingest(self, scenarios) -> None:
        pass

    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        return Answer(
            text=f"rust and kyoto for {question.question_id}",
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=min(100, budget),
            retrieval_latency_ms=1.0,
            context_build_latency_ms=1.0,
            answer_latency_ms=1.0,
        )


class TestRunReplay:
    @pytest.fixture
    def run_dir(self, tmp_path):
        ds = build_dataset()
        ids = [ds.questions[i].question_id for i in (0, 20, 40)]
        config = RunConfig(
            arena_class="controlled",
            players=[ReplayStubPlayer({"context_budget": 2000})],
            dataset=ds,
            answer_repeats=3,
            latency_repeats=3,
            output_dir=str(tmp_path / "results"),
            question_ids=ids,
        )
        ArenaRunner(config).run()
        return next(iter((tmp_path / "results" / "controlled").iterdir()))

    def test_scores_csv_written(self, run_dir):
        scores = score_run_dir(run_dir, build_dataset())
        assert len(scores) == 3
        assert {s.player_name for s in scores} == {"Stub"}
        with (run_dir / "scores.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        # 3 questions x (1 official + 3 repeats)
        assert len(rows) == 12
        assert rows[0]["is_official"] == "1" and rows[0]["repeat_index"] == "majority"
        assert sum(1 for r in rows if r["is_official"] == "1") == 3
        assert (run_dir / "aggregates.csv").exists()
        aggregates = json.loads((run_dir / "aggregates.json").read_text(encoding="utf-8"))
        assert aggregates["scoring_spec"]
        assert aggregates["players"][0]["player"] == "Stub"
        assert aggregates["players"][0]["question_count"] == 3

    def test_replay_is_deterministic(self, run_dir):
        score_run_dir(run_dir, build_dataset())
        first = {
            name: (run_dir / name).read_bytes()
            for name in ("scores.csv", "aggregates.csv", "aggregates.json")
        }
        score_run_dir(run_dir, build_dataset())
        for name, content in first.items():
            assert (run_dir / name).read_bytes() == content

    def test_hash_mismatch_rejected(self, run_dir):
        tampered = build_dataset(version="v0.2.0-arena-1-tampered")
        assert tampered.content_hash() != build_dataset().content_hash()
        with pytest.raises(ValueError, match="dataset hash mismatch"):
            score_run_dir(run_dir, tampered)

    def test_scores_csv_header_matches_spec(self, run_dir):
        score_run_dir(run_dir, build_dataset())
        header = (run_dir / "scores.csv").read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(SCORE_CSV_COLUMNS)
