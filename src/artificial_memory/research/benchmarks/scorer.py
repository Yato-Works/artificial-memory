"""Judge-free Scorer for the Memory Arena (Phase 8.3, Benchmark_Plan.txt §10).

The Scorer is deliberately **separate from the runner**:

    ArenaRunner --record only--> run/ (traces, metrics, tokens, manifest)
                                      |
                                      v
                                   Scorer
                                      |
                                      v
                                 scores.csv

* The runner never asks "what did this answer score?"; it records facts.
* The Scorer reads those facts plus the *frozen dataset* (matched by
  ``question_id``) and produces ``question_id x player x repeat`` scores.
* Because scoring is a pure function of (frozen dataset, recorded answer), a
  better Scorer can re-score old traces. Production runs are never thrown away.

Specification (frozen, §10):

* No LLM judge. Deterministic term matching only.
* ``ground_truth`` is a tuple of *term groups*; a group is satisfied when any of
  its terms appears (synonym groups), and coverage is the fraction of satisfied
  groups.
* Any ``forbidden`` hit forces ``score = 0.0``.
* Abstention questions (``expected_abstention``) pass only when the answer
  states insufficient information; asserting a concrete value is a
  ``false_positive`` failure (§5 Evaluation Boundary).
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from artificial_memory.research.benchmarks.arena import (
    ArenaCategory,
    ArenaDataset,
    ArenaQuestion,
)
from artificial_memory.research.benchmarks.llm import ABSTENTION_TEXT
from artificial_memory.research.benchmarks.player import majority_index

# Frozen scoring spec version. Changing any rule in §10 changes historical
# scores, so the version travels with every score record.
SCORING_SPEC_VERSION = "v0.2.0-scoring-1"

OUTCOME_PASS = "pass"
OUTCOME_PARTIAL = "partial"
OUTCOME_FAIL = "fail"
OUTCOME_FALSE_POSITIVE = "false_positive"

_WS_RE = re.compile(r"\s+")
_ALNUM_ONLY_RE = re.compile(r"[a-z0-9]+")

# Judge-free abstention lexicon (frozen). Combined with ABSTENTION_TEXT from the
# frozen answer prompt, so a player that follows the system prompt is detected.
ABSTENTION_MARKERS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "don't know",
    "not enough information",
    "insufficient information",
    "no information",
    "not specified",
    "not mentioned",
    "no mention",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "unable to determine",
    "cannot say",
    "no evidence",
    "not in the context",
    "there is no information",
    "unknown",
)


# ==================== Normalization / Matching ====================


def normalize_text(text: str) -> str:
    """Normalize text for matching (spec §10 Normalization).

    NFKC -> lowercase -> collapse whitespace. Punctuation is *not* stripped so
    that symbol-bearing terms such as ``F#`` or ``c++`` survive; surrounding
    punctuation is absorbed by the term-side boundary test instead.
    """
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.lower()
    return _WS_RE.sub(" ", normalized).strip()


def _term_regex(term: str) -> re.Pattern[str] | None:
    """Compile a matcher for one term.

    Alphanumeric terms are matched with word boundaries so ``rust`` does not
    match inside ``trust``. Terms with symbols or spaces (``f#``,
    ``visual studio code``, ``cannot resolve``) are matched as fixed substrings
    (word order preserved).
    """
    normalized = normalize_text(term)
    if not normalized:
        return None
    if _ALNUM_ONLY_RE.fullmatch(normalized):
        return re.compile(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])")
    return re.compile(re.escape(normalized))


def term_present(term: str, normalized_answer: str) -> bool:
    """True when ``term`` appears in an already-normalized answer."""
    pattern = _term_regex(term)
    return pattern is not None and pattern.search(normalized_answer) is not None


def matching_terms(terms: Sequence[str], normalized_answer: str) -> tuple[str, ...]:
    """All terms that appear in the answer (in declaration order)."""
    return tuple(term for term in terms if term_present(term, normalized_answer))


def detect_abstention(answer_text: str) -> bool:
    """Judge-free detection of an "insufficient information" answer."""
    normalized = normalize_text(answer_text)
    if not normalized:
        return False
    if term_present(ABSTENTION_TEXT, normalized):
        return True
    return any(term_present(marker, normalized) for marker in ABSTENTION_MARKERS)


# ==================== Per-question score ====================


@dataclass(frozen=True)
class QuestionScore:
    """Score of one (player, question) pair: official value + repeat detail.

    ``outcome`` / ``score`` are the *official* values and are always computed
    from the majority answer (runner's reduction rule). ``repeat_outcomes`` and
    ``repeat_scores`` keep every repeat so ``score_consistency`` can report how
    stable the player was.
    """

    question_id: str
    category: str
    player_name: str
    answer_text: str

    outcome: str
    score: float
    correct: bool

    covered_groups: int
    must_total: int
    ground_truth_coverage: float
    forbidden_violation: bool
    forbidden_hits: tuple[str, ...] = ()
    abstention_detected: bool | None = None
    answer_empty: bool = False

    repeat_outcomes: tuple[str, ...] = ()
    repeat_scores: tuple[float, ...] = ()
    score_consistency: float = 1.0
    answer_source: str = "majority"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "player_name": self.player_name,
            "answer_text": self.answer_text,
            "outcome": self.outcome,
            "score": self.score,
            "correct": self.correct,
            "covered_groups": self.covered_groups,
            "must_total": self.must_total,
            "ground_truth_coverage": self.ground_truth_coverage,
            "forbidden_violation": self.forbidden_violation,
            "forbidden_hits": list(self.forbidden_hits),
            "abstention_detected": self.abstention_detected,
            "answer_empty": self.answer_empty,
            "repeat_outcomes": list(self.repeat_outcomes),
            "repeat_scores": list(self.repeat_scores),
            "score_consistency": self.score_consistency,
            "answer_source": self.answer_source,
        }

    def _row(self, repeat_index: Any, is_official: bool) -> dict[str, Any]:
        return {
            "player": self.player_name,
            "question_id": self.question_id,
            "category": self.category,
            "repeat_index": repeat_index,
            "is_official": int(is_official),
            "outcome": self.outcome,
            "score": round(self.score, 6),
            "correct": int(self.correct),
            "ground_truth_coverage": round(self.ground_truth_coverage, 6),
            "covered_groups": self.covered_groups,
            "must_total": self.must_total,
            "forbidden_violation": int(self.forbidden_violation),
            "forbidden_hits": "|".join(self.forbidden_hits),
            "abstention_detected": (
                "" if self.abstention_detected is None else int(self.abstention_detected)
            ),
            "answer_empty": int(self.answer_empty),
            "score_consistency": round(self.score_consistency, 6),
            "answer_text": self.answer_text,
        }

    def official_row(self) -> dict[str, Any]:
        """CSV row for the official (majority-answer) score."""
        return self._row(repeat_index="majority", is_official=True)

    def repeat_rows(self) -> list[dict[str, Any]]:
        """CSV rows for every individual repeat."""
        rows: list[dict[str, Any]] = []
        for index, (outcome, score) in enumerate(zip(self.repeat_outcomes, self.repeat_scores)):
            row = self._row(repeat_index=index, is_official=False)
            row["outcome"] = outcome
            row["score"] = round(score, 6)
            row["correct"] = int(outcome == OUTCOME_PASS)
            rows.append(row)
        return rows


SCORE_CSV_COLUMNS: tuple[str, ...] = (
    "player",
    "question_id",
    "category",
    "repeat_index",
    "is_official",
    "outcome",
    "score",
    "correct",
    "ground_truth_coverage",
    "covered_groups",
    "must_total",
    "forbidden_violation",
    "forbidden_hits",
    "abstention_detected",
    "answer_empty",
    "score_consistency",
    "answer_text",
)


# ==================== Judgement (§10 Outcome table) ====================


def _coverage(covered_groups: int, must_total: int) -> float:
    """Group coverage; ``must_total == 0`` counts as fully covered (§10)."""
    if must_total == 0:
        return 1.0
    return covered_groups / must_total


def _empty_score(question: ArenaQuestion, player_name: str, answer_text: str) -> QuestionScore:
    """Blank answer -> ``fail`` / 0.0 (§10).

    ``abstention_detected`` is ``None`` (nothing was said, so the answer is
    neither an abstention statement nor an assertion) and coverage is reported
    as 0.0 even when the question has no term groups, because a blank answer
    satisfies nothing. This keeps blank answers out of the ``false_positive``
    bucket for abstention questions.
    """
    return QuestionScore(
        question_id=question.question_id,
        category=question.category.value,
        player_name=player_name,
        answer_text=answer_text,
        outcome=OUTCOME_FAIL,
        score=0.0,
        correct=False,
        covered_groups=0,
        must_total=len(question.ground_truth),
        ground_truth_coverage=0.0,
        forbidden_violation=False,
        forbidden_hits=(),
        abstention_detected=None,
        answer_empty=True,
    )


def score_answer(
    question: ArenaQuestion,
    answer_text: str,
    player_name: str = "",
) -> QuestionScore:
    """Score one answer against the frozen ground truth (§10).

    Pure and deterministic: same (question, answer) always yields the same
    score. Ground truth never flows back to a player; this is the only place
    where it is read.
    """
    if not normalize_text(answer_text):
        return _empty_score(question, player_name, answer_text)

    normalized = normalize_text(answer_text)

    covered_groups = sum(
        1
        for group in question.ground_truth
        if any(term_present(term, normalized) for term in group)
    )
    must_total = len(question.ground_truth)
    coverage = _coverage(covered_groups, must_total)

    forbidden_hits = matching_terms(question.forbidden, normalized)
    forbidden_violation = bool(forbidden_hits)
    abstained = detect_abstention(answer_text)

    if question.expected_abstention:
        if abstained:
            # Abstaining is correct unless a forbidden value was also asserted.
            outcome = OUTCOME_FALSE_POSITIVE if forbidden_violation else OUTCOME_PASS
        elif must_total > 0 and covered_groups == must_total:
            outcome = OUTCOME_PASS
        else:
            # Asserted a concrete value without evidence: false positive.
            outcome = OUTCOME_FALSE_POSITIVE
    elif forbidden_violation:
        outcome = OUTCOME_FAIL
    elif covered_groups == must_total:
        outcome = OUTCOME_PASS
    elif covered_groups > 0:
        outcome = OUTCOME_PARTIAL
    else:
        outcome = OUTCOME_FAIL

    if outcome == OUTCOME_PASS:
        score = 1.0
    elif outcome == OUTCOME_PARTIAL:
        score = coverage
    else:
        score = 0.0

    return QuestionScore(
        question_id=question.question_id,
        category=question.category.value,
        player_name=player_name,
        answer_text=answer_text,
        outcome=outcome,
        score=score,
        correct=outcome == OUTCOME_PASS,
        covered_groups=covered_groups,
        must_total=must_total,
        ground_truth_coverage=coverage,
        forbidden_violation=forbidden_violation,
        forbidden_hits=forbidden_hits,
        abstention_detected=abstained,
        answer_empty=False,
    )


# ==================== Repeat reduction (§10 Determinism) ====================


def score_answer_repeats(
    question: ArenaQuestion,
    answers: Sequence[str],
    player_name: str = "",
) -> QuestionScore:
    """Score every repeat and reduce to the official (majority) score.

    §9/§10: the official value is the score of the *majority answer*, counted by
    exact answer text with the shared deterministic tie-break
    (:func:`majority_index`, smallest repeat index). All repeat scores are kept
    in ``repeat_scores`` / ``repeat_outcomes`` so instability is visible, and
    ``score_consistency`` reports the share of repeats that agree with the
    official outcome.
    """
    texts = list(answers)
    if not texts:
        return _empty_score(question, player_name, "")

    repeat_scores = [score_answer(question, text, player_name) for text in texts]
    index = majority_index(texts)
    official = repeat_scores[index] if index >= 0 else repeat_scores[0]

    consistency = sum(
        1 for score in repeat_scores if score.outcome == official.outcome
    ) / len(repeat_scores)

    return replace(
        official,
        repeat_outcomes=tuple(score.outcome for score in repeat_scores),
        repeat_scores=tuple(score.score for score in repeat_scores),
        score_consistency=consistency,
        answer_source="majority",
    )


# ==================== Aggregates (§10 集約メトリクス) ====================


@dataclass(frozen=True)
class ScoreAggregate:
    """Aggregated quality metrics over a set of question scores (§10)."""

    player_name: str
    question_count: int
    pass_count: int
    partial_count: int
    fail_count: int
    false_positive_count: int

    answer_accuracy: float
    partial_rate: float
    false_positive_rate: float
    forbidden_violation_rate: float
    mean_ground_truth_coverage: float
    mean_score: float
    abstention_accuracy: float | None
    abstention_question_count: int
    mean_score_consistency: float
    category_accuracy: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player_name,
            "question_count": self.question_count,
            "pass_count": self.pass_count,
            "partial_count": self.partial_count,
            "fail_count": self.fail_count,
            "false_positive_count": self.false_positive_count,
            "answer_accuracy": self.answer_accuracy,
            "partial_rate": self.partial_rate,
            "false_positive_rate": self.false_positive_rate,
            "forbidden_violation_rate": self.forbidden_violation_rate,
            "mean_ground_truth_coverage": self.mean_ground_truth_coverage,
            "mean_score": self.mean_score,
            "abstention_accuracy": self.abstention_accuracy,
            "abstention_question_count": self.abstention_question_count,
            "mean_score_consistency": self.mean_score_consistency,
            "category_accuracy": self.category_accuracy,
        }


AGGREGATE_CSV_COLUMNS: tuple[str, ...] = (
    "player",
    "question_count",
    "pass_count",
    "partial_count",
    "fail_count",
    "false_positive_count",
    "answer_accuracy",
    "partial_rate",
    "false_positive_rate",
    "forbidden_violation_rate",
    "mean_ground_truth_coverage",
    "mean_score",
    "abstention_accuracy",
    "abstention_question_count",
    "mean_score_consistency",
)
def build_aggregate(player_name: str, scores: Sequence[QuestionScore]) -> ScoreAggregate:
    """Aggregate per-question official scores into the §10 quality metrics."""
    total = len(scores)
    pass_count = sum(1 for s in scores if s.outcome == OUTCOME_PASS)
    partial_count = sum(1 for s in scores if s.outcome == OUTCOME_PARTIAL)
    fail_count = sum(1 for s in scores if s.outcome == OUTCOME_FAIL)
    fp_count = sum(1 for s in scores if s.outcome == OUTCOME_FALSE_POSITIVE)

    category_pass: dict[str, int] = {}
    category_total: dict[str, int] = {}
    abstention_pass = 0
    for s in scores:
        category_total[s.category] = category_total.get(s.category, 0) + 1
        if s.outcome == OUTCOME_PASS:
            category_pass[s.category] = category_pass.get(s.category, 0) + 1
            if s.category == ArenaCategory.ABSTENTION.value:
                abstention_pass += 1
    abstention_count = category_total.get(ArenaCategory.ABSTENTION.value, 0)

    return ScoreAggregate(
        player_name=player_name,
        question_count=total,
        pass_count=pass_count,
        partial_count=partial_count,
        fail_count=fail_count,
        false_positive_count=fp_count,
        answer_accuracy=pass_count / total if total else 0.0,
        partial_rate=partial_count / total if total else 0.0,
        false_positive_rate=fp_count / total if total else 0.0,
        forbidden_violation_rate=(
            sum(1 for s in scores if s.forbidden_violation) / total if total else 0.0
        ),
        mean_ground_truth_coverage=(
            sum(s.ground_truth_coverage for s in scores) / total if total else 0.0
        ),
        mean_score=sum(s.score for s in scores) / total if total else 0.0,
        abstention_accuracy=(abstention_pass / abstention_count) if abstention_count else None,
        abstention_question_count=abstention_count,
        mean_score_consistency=(
            sum(s.score_consistency for s in scores) / total if total else 1.0
        ),
        category_accuracy={
            cat: category_pass.get(cat, 0) / count
            for cat, count in sorted(category_total.items())
        },
    )


def player_aggregates(scores: Sequence[QuestionScore]) -> list[ScoreAggregate]:
    """One :class:`ScoreAggregate` per player, in first-seen order."""
    order: list[str] = []
    grouped: dict[str, list[QuestionScore]] = {}
    for score in scores:
        if score.player_name not in grouped:
            grouped[score.player_name] = []
            order.append(score.player_name)
        grouped[score.player_name].append(score)
    return [build_aggregate(name, grouped[name]) for name in order]


def aggregate_csv_row(agg: ScoreAggregate) -> dict[str, Any]:
    """Flat CSV row matching :data:`AGGREGATE_CSV_COLUMNS`."""
    data = agg.to_dict()
    data.pop("category_accuracy")  # JSON-only: per-category columns are not fixed
    return data


def write_aggregates_csv(aggregates: Sequence[ScoreAggregate], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AGGREGATE_CSV_COLUMNS))
        writer.writeheader()
        for agg in aggregates:
            writer.writerow(aggregate_csv_row(agg))


def write_scores_csv(scores: Sequence[QuestionScore], path: Path) -> None:
    """Official row + per-repeat rows, ordered player -> question -> repeat."""
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SCORE_CSV_COLUMNS))
        writer.writeheader()
        for score in scores:
            writer.writerow(score.official_row())
            for row in score.repeat_rows():
                writer.writerow(row)


def score_run_dir(
    run_dir: Path | str,
    dataset: ArenaDataset,
    write: bool = True,
) -> list[QuestionScore]:
    """Replay-scorer for a persisted run directory (§10 Scorer/runner分離).

    The Scorer is a pure post-step: it never talks to the runner, never sees a
    player and never calls an LLM. It reads ``traces.jsonl`` (facts recorded by
    the runner) plus the frozen dataset and produces ``scores.csv`` /
    ``aggregates.csv`` / ``aggregates.json`` in the run directory. Because
    scoring is a pure function of (frozen dataset, recorded answer text), this
    can be re-run at any time — including against an improved Scorer — without
    invalidating the recorded run.

    Raises ``ValueError`` when the run's recorded ``dataset_sha256`` does not
    match the provided dataset: scores are only defined against the exact
    freeze the run was executed on.
    """
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    recorded_hash = str(result.get("dataset_hash", ""))
    if recorded_hash != dataset.content_hash():
        raise ValueError(
            "dataset hash mismatch: run "
            f"{run_dir.name} recorded {recorded_hash[:16]}... but this dataset "
            f"hashes to {dataset.content_hash()[:16]}... — scoring is only "
            "defined against the frozen dataset the run was executed on"
        )

    traces_path = run_dir / "traces.jsonl"
    players = [
        json.loads(line)
        for line in traces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    scores: list[QuestionScore] = []
    for player_record in players:
        player_name = str(player_record.get("player_name", ""))
        asked = player_record.get("questions", {})
        # Dataset order for deterministic output; subset runs only score what
        # the run actually asked (manifest keeps the full-freeze hash).
        for question in dataset.questions:
            record = asked.get(question.question_id)
            if record is None:
                continue
            answers = sorted(
                record.get("answers", []),
                key=lambda a: int(a.get("run_index", 0)),
            )
            texts = [str(a.get("text", "")) for a in answers]
            if not texts:
                continue
            scores.append(score_answer_repeats(question, texts, player_name))

    if write:
        write_scores_csv(scores, run_dir / "scores.csv")
        aggregates = player_aggregates(scores)
        write_aggregates_csv(aggregates, run_dir / "aggregates.csv")
        (run_dir / "aggregates.json").write_text(
            json.dumps(
                {
                    "scoring_spec": SCORING_SPEC_VERSION,
                    "players": [agg.to_dict() for agg in aggregates],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return scores

