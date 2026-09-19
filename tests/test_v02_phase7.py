"""Tests for AM v0.2.0 Phase 7: Benchmark Infrastructure (Track B arena).

Covers (plan #33 / #34 and the Phase 7 implementation list):

* dataset shape (10 categories x 20 questions = 200, stable ids),
* scenario integrity (unique session buckets, non-negative days, ordered turns),
* question quality (ground truth, abstention flags, evidence resolvability),
* determinism (content hash stable across builds, created_at provenance),
* freeze roundtrip (write / read hash / verify, tamper detection).
"""

import pytest

from artificial_memory.research.benchmarks.arena import (
    ARENA_EPOCH,
    ARENA_VERSION,
    PROJECTS,
    PROJECTS_UNQUERIED,
    QUESTIONS_PER_CATEGORY,
    SESSION_COUNT,
    TOTAL_QUESTIONS,
    ArenaCategory,
    build_dataset,
    read_frozen_hash,
    verify_freeze,
    write_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


class TestDatasetShape:
    def test_total_questions(self, dataset):
        assert dataset.question_count == TOTAL_QUESTIONS

    def test_categories_times_questions_per_category(self, dataset):
        assert len(list(ArenaCategory)) * QUESTIONS_PER_CATEGORY == TOTAL_QUESTIONS
        counts = dataset.category_counts()
        assert set(counts.values()) == {QUESTIONS_PER_CATEGORY}

    def test_question_ids_unique_and_well_formed(self, dataset):
        ids = [q.question_id for q in dataset.questions]
        assert len(set(ids)) == len(ids)
        for question in dataset.questions:
            category, _, index = question.question_id.rpartition("-")
            assert category == question.category.value
            assert 0 <= int(index) < QUESTIONS_PER_CATEGORY

    def test_by_id_roundtrip(self, dataset):
        first = dataset.questions[0]
        assert dataset.by_id(first.question_id) is first
        assert dataset.by_id("no-such-question") is None

    def test_by_category(self, dataset):
        for category in ArenaCategory:
            selected = dataset.by_category(category)
            assert len(selected) == QUESTIONS_PER_CATEGORY
            assert all(q.category is category for q in selected)


class TestScenarioIntegrity:
    def test_scenario_ids_unique_and_sequential(self, dataset):
        ids = [s.scenario_id for s in dataset.scenarios]
        assert len(set(ids)) == len(ids)
        numbers = [int(sid.rsplit("-", 1)[-1]) for sid in ids]
        assert numbers == sorted(numbers)
        assert numbers[0] == 0
        assert numbers[-1] >= SESSION_COUNT - 1  # base grid fully covered

    def test_base_grid_sessions_exist(self, dataset):
        for index in range(SESSION_COUNT):
            assert dataset.scenario(index) is not None

    def test_scenario_lookup_miss(self, dataset):
        assert dataset.scenario(999) is None

    def test_no_negative_days(self, dataset):
        for scenario in dataset.scenarios:
            for turn in scenario.turns:
                assert turn.day >= 0

    def test_turns_day_ordered_within_scenario(self, dataset):
        for scenario in dataset.scenarios:
            days = [turn.day for turn in scenario.turns]
            assert days == sorted(days)

    def test_turn_session_index_matches_bucket(self, dataset):
        for scenario in dataset.scenarios:
            bucket = int(scenario.scenario_id.rsplit("-", 1)[-1])
            assert all(turn.session_index == bucket for turn in scenario.turns)

    def test_scenario_day_is_first_turn_day(self, dataset):
        for scenario in dataset.scenarios:
            assert scenario.day == scenario.turns[0].day


class TestQuestionQuality:
    def test_ground_truth_by_abstention_flag(self, dataset):
        for question in dataset.questions:
            if question.expected_abstention:
                # Abstention questions have no correct answer: ground truth is
                # empty and the forbidden list holds the planted-but-wrong
                # value that must not be reported.
                assert question.ground_truth == (), question.question_id
                assert question.forbidden, question.question_id
            else:
                assert question.ground_truth, question.question_id
                for group in question.ground_truth:
                    assert group, question.question_id

    def test_abstention_questions_expect_no_answer(self, dataset):
        abstention = dataset.by_category(ArenaCategory.ABSTENTION)
        assert len(abstention) == QUESTIONS_PER_CATEGORY
        assert all(q.expected_abstention for q in abstention)

    def test_abstention_distractor_entities_are_never_queried(self):
        # Phase 8.12 repair: the near-miss entity pool must stay disjoint from
        # the queried project pool, otherwise a later plot can silently supply
        # evidence for an earlier abstention question.
        assert set(PROJECTS).isdisjoint(PROJECTS_UNQUERIED)

    def test_abstention_history_contains_no_evidence(self, dataset):
        """Abstention must be *structurally* unwinnable-by-answering.

        Every abstention question asks for a debugger of a project; the
        history may only deny it. If any turn positively asserts the attribute
        for the queried project, "no evidence" silently became "evidence", and
        a system that answers correctly is scored as a false positive
        (measured pre-repair: 7 of 20 questions were evidenced).
        """
        history = [turn.content for scenario in dataset.scenarios for turn in scenario.turns]
        for question in dataset.by_category(ArenaCategory.ABSTENTION):
            project = question.question.split("for project ")[-1].rstrip("?").strip()
            assertion = f"for project {project}, the user set the debugger to"
            offenders = [text for text in history if assertion in text.lower()]
            assert not offenders, (question.question_id, offenders)

    def test_non_abstention_questions_do_not_expect_abstention(self, dataset):
        others = [
            q
            for q in dataset.questions
            if q.category is not ArenaCategory.ABSTENTION
        ]
        assert all(not q.expected_abstention for q in others)

    def test_evidence_sessions_resolve_to_scenarios(self, dataset):
        for question in dataset.questions:
            assert question.evidence_sessions, question.question_id
            for session in question.evidence_sessions:
                assert dataset.scenario(session) is not None, question.question_id

    def test_required_sessions_within_distinct_evidence(self, dataset):
        for question in dataset.questions:
            distinct = len(set(question.evidence_sessions))
            assert 1 <= question.required_sessions <= distinct, question.question_id

    def test_reference_timestamps_are_inside_arena_horizon(self, dataset):
        for question in dataset.questions:
            if question.reference_timestamp is None:
                continue
            assert question.reference_timestamp >= ARENA_EPOCH, question.question_id

    def test_difficulty_vocabulary(self, dataset):
        assert all(q.difficulty in {"easy", "medium", "hard"} for q in dataset.questions)

    def test_forbidden_carries_the_wrong_value(self, dataset):
        # A forbidden term is always a planted value that would be wrong to
        # report: the superseded value for updates, the distractor for
        # abstention. It never coincides with the ground truth.
        for question in dataset.questions:
            for term in question.forbidden:
                assert not any(
                    term in group for group in question.ground_truth
                ), question.question_id


def ds_scenario(dataset, session_index: int):
    return dataset.scenario(session_index)

class TestDeterminism:
    def test_content_hash_stable_across_builds(self, dataset):
        rebuilt = build_dataset()
        assert rebuilt.content_hash() == dataset.content_hash()

    def test_created_at_excluded_by_default(self, dataset):
        assert dataset.created_at == ""

    def test_explicit_created_at_changes_hash_but_not_content(self, dataset):
        stamped = build_dataset(created_at="2026-01-05T09:00:00+00:00")
        assert stamped.created_at != dataset.created_at
        assert stamped.content_hash() != dataset.content_hash()
        assert [q.to_dict() for q in stamped.questions] == [
            q.to_dict() for q in dataset.questions
        ]

    def test_version_flows_into_metadata(self, dataset):
        assert dataset.version == ARENA_VERSION

class TestFreeze:
    def test_write_read_verify_roundtrip(self, dataset, tmp_path):
        json_path, hash_path = write_dataset(dataset, directory=tmp_path)
        assert json_path.exists()
        assert hash_path.exists()
        frozen = read_frozen_hash(directory=tmp_path)
        assert frozen == dataset.content_hash()
        is_frozen, frozen_hash, current_hash = verify_freeze(directory=tmp_path)
        assert is_frozen is True
        assert frozen_hash == dataset.content_hash()
        assert current_hash == dataset.content_hash()

    def test_verify_detects_stale_hash(self, dataset, tmp_path):
        write_dataset(dataset, directory=tmp_path)
        hash_path = tmp_path / "arena.sha256"
        stale = "0" * 64
        hash_path.write_text(f"{stale}  arena.json\n", encoding="utf-8")
        is_frozen, frozen_hash, _ = verify_freeze(directory=tmp_path)
        assert is_frozen is False
        assert frozen_hash == stale

    def test_verify_without_freeze_reports_unfrozen(self, tmp_path):
        is_frozen, frozen_hash, current_hash = verify_freeze(directory=tmp_path)
        assert is_frozen is False
        assert frozen_hash == ""
        assert current_hash == build_dataset().content_hash()

    def test_write_is_reproducible(self, dataset, tmp_path):
        first_json, _ = write_dataset(dataset, directory=tmp_path)
        second_json, _ = write_dataset(dataset, directory=tmp_path)
        assert first_json.read_text(encoding="utf-8") == second_json.read_text(
            encoding="utf-8"
        )

