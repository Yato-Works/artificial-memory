"""AM Memory Arena - the frozen 200-question Track B benchmark (Phase 7).

Plan #29 defines a custom benchmark of 200 questions across ten categories,
specified and frozen *before* final system results are observed (Fairness
Rule 1). This module owns the dataset half of the Phase 7 infrastructure::

    benchmark/
      dataset/    <- this module writes arena.json + arena.sha256
      protocol/
      configs/
      results/
      analysis/

Design notes
------------
* The dataset is generated **deterministically** from index-arithmetic tables
  and fixed templates. No RNG, no clock, no network: a version string always
  yields the same question ids and the same content hash, so a run can be
  reproduced exactly and the freeze can be verified (Fairness Rules 1, 6, 8).
* Ground truth is expressed as *term groups* rather than one golden string:
  each group requires at least one of its terms to appear in an answer. This
  keeps scoring deterministic and free of an LLM judge, while still supporting
  "must include X and must not include Y" questions.
* Questions never name their own answer. Every expectation is derived from the
  planted storyline, so a system cannot score by matching the question text.
* The Arena is not designed to make AM win (Fairness Rule 10 preamble); it
  encodes capabilities AM claims, including the ones AM is expected to lose.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

ARENA_VERSION = "v0.2.0-arena-1"
QUESTIONS_PER_CATEGORY = 20
SESSION_INTERVAL_DAYS = 15
SESSION_COUNT = 8
# The dataset epoch is fixed so that every generated timestamp is identical
# across machines and runs.
ARENA_EPOCH = datetime(2026, 1, 5, 9, 0, 0)


class ArenaCategory(StrEnum):
    """The ten Track B categories from plan #29 (20 questions each)."""

    FACTUAL_RECALL = "factual_recall"
    MULTI_SESSION_RECALL = "multi_session_recall"
    TEMPORAL_REASONING = "temporal_reasoning"
    KNOWLEDGE_UPDATES = "knowledge_updates"
    CONTRADICTION_HANDLING = "contradiction_handling"
    INDIRECT_RECALL = "indirect_recall"
    DISTRACTOR_RESISTANCE = "distractor_resistance"
    ABSTENTION = "abstention"
    COMPRESSION_RECOVERY = "compression_recovery"
    REFLECTION_REINTERPRETATION = "reflection_reinterpretation"


TOTAL_QUESTIONS = QUESTIONS_PER_CATEGORY * len(ArenaCategory)


@dataclass(frozen=True)
class ArenaTurn:
    """A single utterance inside a session."""

    session_index: int
    speaker: str
    content: str
    day: int

    @property
    def timestamp(self) -> datetime:
        return ARENA_EPOCH + timedelta(days=self.day)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_index": self.session_index,
            "speaker": self.speaker,
            "content": self.content,
            "day": self.day,
        }


@dataclass(frozen=True)
class ArenaScenario:
    """One session of the shared conversation history."""

    scenario_id: str
    title: str
    day: int
    turns: tuple[ArenaTurn, ...]

    @property
    def timestamp(self) -> datetime:
        return ARENA_EPOCH + timedelta(days=self.day)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "day": self.day,
            "turns": [turn.to_dict() for turn in self.turns],
        }

# Deterministic vocabulary tables. ``_pick`` selects an entry by integer index,
# so every generated fact is a pure function of (category, index).
PROJECTS: tuple[str, ...] = (
    "atlas", "beacon", "cinder", "delta", "ember", "fjord", "glacier", "harbor",
    "ivory", "juniper", "kelp", "lumen", "moss", "nectar", "onyx", "prism",
    "quartz", "reef", "summit", "tundra",
)
LANGUAGES: tuple[str, ...] = (
    "Rust", "Go", "Elixir", "Zig", "Kotlin", "Swift", "Julia", "Nim",
    "Crystal", "OCaml", "Haskell", "Clojure", "F#", "D", "V", "Pony",
    "Gleam", "Roc", "Haxe", "Red",
)
CITIES: tuple[str, ...] = (
    "Kyoto", "Lisbon", "Tallinn", "Porto", "Ljubljana", "Bergen", "Ghent",
    "Turku", "Aarhus", "Bologna", "Galway", "Sibiu", "Lund", "Coimbra",
    "Utrecht", "Graz", "Tartu", "Bratislava", "Trieste", "Nantes",
)
TOOLS: tuple[str, ...] = (
    "ripgrep", "bat", "fd", "delta", "zoxide", "starship", "just", "watchexec",
    "helix", "hyperfine", "dust", "procs", "sd", "tokei", "xh", "mise",
    "typos", "bacon", "cargo-nextest", "sccache",
)
DATABASES: tuple[str, ...] = (
    "SQLite", "DuckDB", "Postgres", "RocksDB", "LMDB", "ClickHouse", "TiDB",
    "CockroachDB", "SurrealDB", "Redis", "Cassandra", "MongoDB", "Neo4j",
    "QuestDB", "TimescaleDB", "ArangoDB", "RethinkDB", "CouchDB", "RavenDB",
    "ScyllaDB",
)
METRICS: tuple[str, ...] = (
    "p99 latency", "throughput", "memory footprint", "cold start time",
    "bundle size", "cache hit rate", "error rate", "queue depth", "disk usage",
    "cpu saturation", "gc pause time", "compile time", "index size",
    "read amplification", "write amplification", "tail latency",
    "context switch rate", "page fault rate", "allocation count", "startup cost",
)
DRINKS: tuple[str, ...] = (
    "matcha", "espresso", "oolong", "kombucha", "horchata", "chai",
    "yerba mate", "cider", "kvass", "lassi", "rooibos", "genmaicha",
    "cold brew", "mulled wine", "sbiten", "buttermilk", "barley tea",
    "hibiscus tea", "ginger beer", "sarsaparilla",
)
PETS: tuple[str, ...] = (
    "corgi", "shiba", "border collie", "maine coon", "tabby", "greyhound",
    "pug", "ragdoll", "beagle", "husky", "sphynx", "dachshund", "samoyed",
    "bengal", "poodle", "akita", "birman", "chihuahua", "vizsla", "burmese",
)
OBJECTS: tuple[str, ...] = (
    "ceramic bowl", "wool blanket", "brass lamp", "linen cushion",
    "cast iron pan", "teak stool", "copper kettle", "canvas satchel",
    "glass terrarium", "rattan basket", "felt hat", "leather journal",
    "birch shelf", "clay mug", "marble tray", "walnut pen",
    "cork mat", "iron hook", "paper lantern", "stone mortar",
)
# Distractor values are deliberately *related* to the target so that a system
# relying on topical similarity alone will retrieve the wrong record.
DISTRACTOR_TOOLS: tuple[str, ...] = (
    "fzf", "exa", "lsd", "delta-lake", "zellij", "nushell", "oil", "amber",
    "xsv", "batcat", "ctags", "asciinema", "gitui", "difftastic", "ranger",
    "skim", "nu", "fossil", "hg", "svn",
)
SERIAL_PREFIX = "AM"
SESSION_TITLES: tuple[str, ...] = (
    "Kickoff", "Mid-sprint", "Migration", "Retrospective", "Planning",
    "Review", "Deep work", "Wrap-up",
)
SLOTS_PER_SESSION = TOTAL_QUESTIONS // SESSION_COUNT
ABSTENTION_MARKERS: tuple[str, ...] = (
    "don't know", "do not know", "no information", "not enough information",
    "unknown", "not sure", "cannot find", "can't find", "no record",
    "never mentioned", "no memory", "no evidence", "not mentioned",
    "no supporting", "unavailable",
    "no record", "unclear", "cannot determine", "not verified",
)
# Terms that demonstrate a system surfaced a conflict instead of silently
# choosing a side (plan #21).
CONFLICT_MARKERS: tuple[str, ...] = (
    "conflict", "conflicting", "contradict", "contradiction", "contested",
    "inconsistent", "both", "disagree", "unresolved", "cannot resolve",
    "not resolve", "two different", "two conflicting", "ambiguous",
)



def _pick(pool: tuple[str, ...], index: int) -> str:
    """Deterministic choice from a pool (no RNG, no hashing)."""
    return pool[index % len(pool)]


def _slot_index(category: ArenaCategory, index: int) -> int:
    """Global slot number for (category, index): unique across all 200 questions."""
    return list(ArenaCategory).index(category) * QUESTIONS_PER_CATEGORY + index


def _day_for_slot(slot: int) -> int:
    """Day offset of the session that carries a given slot."""
    return (slot // SLOTS_PER_SESSION) * SESSION_INTERVAL_DAYS


def _session_for_slot(slot: int) -> int:
    return slot // SLOTS_PER_SESSION


@dataclass(frozen=True)
class ArenaQuestion:
    """One Track B question with deterministic, LLM-free ground truth.

    ``ground_truth`` is a tuple of *term groups*. Every group must be satisfied
    by at least one of its terms for the question to count as correct. This
    expresses "must mention X", "must mention X and Y" and (together with
    ``forbidden``) "must mention X but not Z" without a golden sentence and
    without a judge model.
    """

    question_id: str
    category: ArenaCategory
    question: str
    ground_truth: tuple[tuple[str, ...], ...] = ()
    forbidden: tuple[str, ...] = ()
    expected_abstention: bool = False
    reference_timestamp: datetime | None = None
    evidence_sessions: tuple[int, ...] = ()
    required_sessions: int = 1
    difficulty: str = "medium"
    notes: str = ""

    @property
    def slot(self) -> int:
        index = int(self.question_id.rsplit("-", 1)[-1])
        return _slot_index(self.category, index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category.value,
            "question": self.question,
            "ground_truth": [list(group) for group in self.ground_truth],
            "forbidden": list(self.forbidden),
            "expected_abstention": self.expected_abstention,
            "reference_timestamp": (
                self.reference_timestamp.isoformat() if self.reference_timestamp else None
            ),
            "evidence_sessions": list(self.evidence_sessions),
            "required_sessions": self.required_sessions,
            "difficulty": self.difficulty,
            "notes": self.notes,
        }

@dataclass(frozen=True)
class _PlantedFact:
    """An utterance that plants information in a session."""

    day: int
    speaker: str
    content: str


def _date(day: int) -> str:
    return (ARENA_EPOCH + timedelta(days=day)).strftime("%Y-%m-%d")


def _plot_factual_recall(index: int) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """Direct recall of a single planted fact (the capability floor)."""
    slot = _slot_index(ArenaCategory.FACTUAL_RECALL, index)
    day = _day_for_slot(slot)
    project = _pick(PROJECTS, index)
    tool = _pick(TOOLS, slot)
    drink = _pick(DRINKS, slot)
    facts = [
        _PlantedFact(
            day,
            "user",
            f"For project {project} the user said their preferred editor toolchain is {tool}.",
        ),
        _PlantedFact(
            day,
            "assistant",
            f"Recorded for project {project}: preferred editor toolchain {tool}.",
        ),
        _PlantedFact(day, "user", f"The user usually drinks {drink} while working on {project}."),
    ]
    return facts, {
        "question": (
            f"For project {project}, what did the user say their preferred editor toolchain is?"
        ),
        "ground_truth": ((tool,),),
        "difficulty": "easy",
        "evidence_sessions": (_session_for_slot(slot),),
        "notes": "single-memory factual recall",
    }


def _plot_multi_session_recall(index: int) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """A fact stated once, refined later, and generalised even later."""
    slot = _slot_index(ArenaCategory.MULTI_SESSION_RECALL, index)
    session = _session_for_slot(slot)
    project = _pick(PROJECTS, index)
    subject = _pick(LANGUAGES, slot)
    early = max(0, session - 1)
    late = session + 1
    facts = [
        _PlantedFact(
            max(0, _day_for_slot(slot) - SESSION_INTERVAL_DAYS),
            "user",
            f"For project {project}, the user began studying {subject} in the evenings.",
        ),
        _PlantedFact(
            _day_for_slot(slot),
            "user",
            f"For project {project}, the user finished their first {subject} exercise set.",
        ),
        _PlantedFact(
            _day_for_slot(slot) + SESSION_INTERVAL_DAYS,
            "user",
            f"For project {project}, the user now recommends {subject} to colleagues.",
        ),
    ]
    return facts, {
        "question": (
            f"Across our conversations about project {project}, which subject did the user begin "
            "studying, practise, and later recommend to colleagues?"
        ),
        "ground_truth": ((subject,),),
        "evidence_sessions": (early, session, late),
        "required_sessions": len({early, session, late}),
        "difficulty": "medium",
        "notes": "the same subject is asserted in three separate sessions",
    }


def _plot_temporal_reasoning(index: int) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """The user's base changed at a known date; ask for the state at a date."""
    slot = _slot_index(ArenaCategory.TEMPORAL_REASONING, index)
    day = _day_for_slot(slot)
    project = _pick(PROJECTS, index)
    city_before = _pick(CITIES, slot)
    city_after = _pick(CITIES, slot + 7)
    relocation_day = day + 14
    ask_after = index % 2 == 1
    reference_day = day + 22 if ask_after else day + 7
    expected = city_after if ask_after else city_before
    facts = [
        _PlantedFact(
            day,
            "user",
            f"For project {project}, the user was working from {city_before} until "
            f"{_date(relocation_day)}.",
        ),
        _PlantedFact(
            relocation_day,
            "user",
            f"For project {project}, the user relocated to {city_after} on "
            f"{_date(relocation_day)}.",
        ),
    ]
    return facts, {
        "question": (
            f"Where was the user working on {_date(reference_day)} while project {project} "
            "was active?"
        ),
        "ground_truth": ((expected,),),
        "reference_timestamp": ARENA_EPOCH + timedelta(days=reference_day),
        "evidence_sessions": (_session_for_slot(slot),),
        "difficulty": "medium",
        "notes": "answer depends on the validity window, not on recency",
    }


def _plot_knowledge_updates(index: int) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """A value was updated; the current value must win, the old one must not."""
    slot = _slot_index(ArenaCategory.KNOWLEDGE_UPDATES, index)
    day = _day_for_slot(slot)
    project = _pick(PROJECTS, index)
    db_old = _pick(DATABASES, slot)
    db_new = _pick(DATABASES, slot + 5)
    facts = [
        _PlantedFact(day, "user", f"For project {project}, the primary database was {db_old}."),
        _PlantedFact(
            day + 7,
            "user",
            f"For project {project}, the user migrated the primary database from {db_old} "
            f"to {db_new}.",
        ),
    ]
    return facts, {
        "question": f"What is the user's primary database for project {project} now?",
        "ground_truth": ((db_new,),),
        "forbidden": (db_old,),
        "evidence_sessions": (_session_for_slot(slot),),
        "difficulty": "medium",
        "notes": "recency alone is not enough: the update must supersede the old value",
    }

def _plot_contradiction_handling(index: int) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """Two incompatible assertions with overlapping windows: state is contested.

    The expected behaviour is *not* to pick a winner. Both values must be
    reported and the conflict must be surfaced, because nothing in the history
    establishes which one is authoritative (plan #21).
    """
    slot = _slot_index(ArenaCategory.CONTRADICTION_HANDLING, index)
    day = _day_for_slot(slot)
    project = _pick(PROJECTS, index)
    db_first = _pick(DATABASES, slot)
    db_second = _pick(DATABASES, slot + 9)
    facts = [
        _PlantedFact(
            day,
            "user",
            f"For project {project}, the user reported the cache layer runs on {db_first}.",
        ),
        _PlantedFact(
            day + 3,
            "assistant",
            f"For project {project}, a teammate reported the cache layer actually runs on "
            f"{db_second}.",
        ),
        _PlantedFact(
            day + 4,
            "user",
            f"For project {project}, the user has not verified which cache layer report is "
            "correct.",
        ),
    ]
    return facts, {
        "question": (
            f"For project {project}, which cache layer should I assume the system runs on?"
        ),
        "ground_truth": ((db_first, db_second), CONFLICT_MARKERS),
        "reference_timestamp": ARENA_EPOCH + timedelta(days=day + 5),
        "evidence_sessions": (_session_for_slot(slot),),
        "difficulty": "hard",
        "notes": (
            "both reports stay valid at the reference time; answering with a single value "
            "is a failure"
        ),
    }


def _plot_indirect_recall(index: int) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """The answer is never stated; it must be joined from two memories.

    One memory links a project to a toolchain, another links the same toolchain
    to a place. Only a system that combines them can answer (plan #15).
    """
    slot = _slot_index(ArenaCategory.INDIRECT_RECALL, index)
    day = _day_for_slot(slot)
    project = _pick(PROJECTS, index)
    tool = _pick(TOOLS, slot)
    city = _pick(CITIES, slot + 3)
    facts = [
        _PlantedFact(
            day,
            "user",
            f"For project {project} the user settled on {tool} as the daily driver.",
        ),
        _PlantedFact(
            day + 7,
            "user",
            f"The workstation in {city} is the only machine with {tool} installed.",
        ),
    ]
    return facts, {
        "question": (
            f"In which city is the machine that runs the daily driver toolchain of project "
            f"{project} located?"
        ),
        "ground_truth": ((city,),),
        "evidence_sessions": (_session_for_slot(slot),),
        "difficulty": "hard",
        "notes": "answer requires joining two distinct memories; no single memory states it",
    }

def _plot_distractor_resistance(index: int) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """A near-identical distractor exists for a different project.

    The target and the distractor are topically almost identical, so a system
    that ranks by similarity alone will retrieve the wrong record. The correct
    answer must be attributed to the right project (plan #30).
    """
    slot = _slot_index(ArenaCategory.DISTRACTOR_RESISTANCE, index)
    day = _day_for_slot(slot)
    project = _pick(PROJECTS, index)
    other_project = _pick(PROJECTS, index + 11)
    tool = _pick(TOOLS, slot)
    distractor = _pick(DISTRACTOR_TOOLS, slot)
    facts = [
        _PlantedFact(
            day,
            "user",
            f"For project {project}, the user pinned the file search tool to {tool}.",
        ),
        _PlantedFact(
            day + 2,
            "user",
            f"For project {other_project}, the user pinned the file search tool to {distractor}.",
        ),
        _PlantedFact(
            day + 3,
            "assistant",
            f"Noted two file search preferences: {project} uses {tool}, "
            f"{other_project} uses {distractor}.",
        ),
    ]
    return facts, {
        "question": f"Which file search tool is pinned for project {project}?",
        "ground_truth": ((tool,),),
        "forbidden": (distractor,),
        "evidence_sessions": (_session_for_slot(slot),),
        "difficulty": "medium",
        "notes": "the distractor belongs to a different project and must not be reported",
    }


def _plot_abstention(index: int) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """Nothing in the history supports an answer; abstention is correct.

    The queried attribute is never asserted for the queried project, while
    related projects do have such attributes. This separates \"no evidence\"
    from \"weak evidence\" (plan #30: abstention accuracy).
    """
    slot = _slot_index(ArenaCategory.ABSTENTION, index)
    day = _day_for_slot(slot)
    project = _pick(PROJECTS, index)
    other_project = _pick(PROJECTS, index + 7)
    tool = _pick(TOOLS, slot)
    facts = [
        _PlantedFact(
            day,
            "user",
            f"For project {project}, the user only discussed scheduling and never a debugger.",
        ),
        _PlantedFact(
            day + 1,
            "user",
            f"For project {other_project}, the user set the debugger to {tool}.",
        ),
    ]
    return facts, {
        "question": f"Which debugger did the user configure for project {project}?",
        "ground_truth": (),
        "forbidden": (tool,),
        "expected_abstention": True,
        "evidence_sessions": (_session_for_slot(slot),),
        "difficulty": "hard",
        "notes": "the attribute is only planted for a different project; any value is wrong",
    }


def _plot_compression_recovery(index: int) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """A detail that only survives if compression keeps it recoverable.

    The fact is stated once with a distinctive detail. After compression the
    detail is expected to be reachable through stored versions or the evidence
    graph, so this category measures resolution recovery (plan #30).
    """
    slot = _slot_index(ArenaCategory.COMPRESSION_RECOVERY, index)
    day = _day_for_slot(slot)
    project = _pick(PROJECTS, index)
    metric = _pick(METRICS, slot)
    serial = f"{SERIAL_PREFIX}-{1000 + slot}"
    facts = [
        _PlantedFact(
            day,
            "user",
            f"For project {project} the user measured {metric} as 12.5 ms and filed it under "
            f"serial {serial}.",
        ),
        _PlantedFact(
            day,
            "assistant",
            f"Recorded measurement for project {project}: {metric} = 12.5 ms, "
            f"serial {serial}.",
        ),
        _PlantedFact(
            day + 1,
            "user",
            f"For project {project} the user only remembers that a {metric} measurement was "
            "filed.",
        ),
    ]
    return facts, {
        "question": (
            f"What serial number was filed for the {metric} measurement of project {project}?"
        ),
        "ground_truth": ((serial,),),
        "evidence_sessions": (_session_for_slot(slot),),
        "difficulty": "hard",
        "notes": (
            "a low-resolution summary loses the serial; only stored versions or graph "
            "expansion recover it"
        ),
    }



# Reflection themes: three concrete observations that together support one
# higher-level interpretation, plus the phrasings accepted for it. The
# interpretation is never stated in the history, so the answer can only come
# from REINTERPRETing the combined evidence (plan #12/#13).
THEME_OBSERVATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "the user always reverts the build configuration to the last known good state",
        "the user copies files by hand instead of symlinking them",
        "the user keeps an untouched backup of the previous schema",
    ),
    (
        "the user refuses to add a new dependency for a small helper",
        "the user rewrote a helper to drop a transitive package",
        "the user asked for a standard-library solution before a library",
    ),
    (
        "the user benchmarks before and after every change",
        "the user logs every measurement under a serial number",
        "the user compares profiles from two consecutive releases",
    ),
)
THEME_MARKERS: tuple[tuple[str, ...], ...] = (
    ("stability", "stable", "conservative", "safe", "backward", "revert", "rollback"),
    ("minimal", "dependency", "dependencies", "lean", "no library", "reduce"),
    ("measurement", "measure", "benchmark", "profile", "empirical", "data"),
)


def _plot_reflection_reinterpretation(
    index: int,
) -> tuple[list[_PlantedFact], dict[str, Any]]:
    """A pattern spread over sessions must be reinterpreted into one conclusion.

    No session states the conclusion. Each contributes one observation, and the
    answer is the interpretation that the combined evidence supports (plan #12).
    """
    slot = _slot_index(ArenaCategory.REFLECTION_REINTERPRETATION, index)
    day = _day_for_slot(slot)
    project = _pick(PROJECTS, index)
    theme_index = slot % len(THEME_OBSERVATIONS)
    observations = THEME_OBSERVATIONS[theme_index]
    markers = THEME_MARKERS[theme_index]

    facts = [
        _PlantedFact(day, "user", f"For project {project}, {observations[0]}."),
        _PlantedFact(
            day + SESSION_INTERVAL_DAYS,
            "user",
            f"For project {project}, {observations[1]}.",
        ),
        _PlantedFact(
            day + 2 * SESSION_INTERVAL_DAYS,
            "user",
            f"For project {project}, {observations[2]}.",
        ),
    ]
    return facts, {
        "question": (
            f"Considering everything the user said about project {project}, what recurring "
            "preference does the user consistently show?"
        ),
        "ground_truth": (markers,),
        "evidence_sessions": (
            _session_for_slot(slot),
            _session_for_slot(slot) + 1,
            _session_for_slot(slot) + 2,
        ),
        "required_sessions": 3,
        "difficulty": "hard",
        "notes": (
            "the conclusion is never stated; three sessions must be reinterpreted together"
        ),
    }


# ==================== Assembly ====================

PlotFunction = Callable[[int], tuple[list["_PlantedFact"], dict[str, Any]]]


def _plot_functions() -> dict[ArenaCategory, PlotFunction]:
    """Category -> plot generator. Order defines question-id ordering."""
    return {
        ArenaCategory.FACTUAL_RECALL: _plot_factual_recall,
        ArenaCategory.MULTI_SESSION_RECALL: _plot_multi_session_recall,
        ArenaCategory.TEMPORAL_REASONING: _plot_temporal_reasoning,
        ArenaCategory.KNOWLEDGE_UPDATES: _plot_knowledge_updates,
        ArenaCategory.CONTRADICTION_HANDLING: _plot_contradiction_handling,
        ArenaCategory.INDIRECT_RECALL: _plot_indirect_recall,
        ArenaCategory.DISTRACTOR_RESISTANCE: _plot_distractor_resistance,
        ArenaCategory.ABSTENTION: _plot_abstention,
        ArenaCategory.COMPRESSION_RECOVERY: _plot_compression_recovery,
        ArenaCategory.REFLECTION_REINTERPRETATION: _plot_reflection_reinterpretation,
    }


def _normalize_groups(raw: Any) -> tuple[tuple[str, ...], ...]:
    """Normalize a spec's ``ground_truth`` into term groups.

    A flat sequence of strings becomes one group per string; a nested sequence
    is preserved. Earlier categories return either shape.
    """
    if not raw:
        return ()
    groups: list[tuple[str, ...]] = []
    for entry in raw:
        if isinstance(entry, str):
            groups.append((entry,))
        else:
            groups.append(tuple(entry))
    return tuple(groups)


def build_questions() -> tuple[ArenaQuestion, ...]:
    """Build all 200 questions deterministically.

    ``Question.question_id`` is ``<category>-<index>`` with a zero-padded index,
    which makes ids stable, sortable and self-describing.
    """
    questions: list[ArenaQuestion] = []
    for category, plot in _plot_functions().items():
        for index in range(QUESTIONS_PER_CATEGORY):
            _, spec = plot(index)
            questions.append(
                ArenaQuestion(
                    question_id=f"{category.value}-{index:02d}",
                    category=category,
                    question=spec["question"],
                    ground_truth=_normalize_groups(spec.get("ground_truth")),
                    forbidden=tuple(spec.get("forbidden", ())),
                    expected_abstention=bool(spec.get("expected_abstention", False)),
                    reference_timestamp=spec.get("reference_timestamp"),
                    evidence_sessions=tuple(spec.get("evidence_sessions", ())),
                    required_sessions=int(spec.get("required_sessions", 1)),
                    difficulty=spec.get("difficulty", "medium"),
                    notes=spec.get("notes", ""),
                )
            )
    return tuple(questions)


def build_scenarios() -> tuple[ArenaScenario, ...]:
    """Build the shared conversation history every system replays.

    All 200 questions are planted into *one* history, exactly as in the plan:
    systems are compared on the same stream, not on per-question conversations.

    A scenario is one **session bucket**: ``day // SESSION_INTERVAL_DAYS``.
    All facts planted on days of the same bucket are merged into that
    bucket's scenario (ordered by day, then speaker/content). The bucket
    index agrees with ``_session_for_slot`` for the base days the plots use,
    and stays monotone for cross-session follow-ups (``+15`` / ``+30``), so
    late questions can reference trailing sessions (8, 9, ...) that exist
    only because a plot planted evidence there.
    """
    facts_by_day: dict[int, set[tuple[str, str]]] = {}
    for plot in _plot_functions().values():
        for index in range(QUESTIONS_PER_CATEGORY):
            facts, _ = plot(index)
            for fact in facts:
                facts_by_day.setdefault(fact.day, set()).add((fact.speaker, fact.content))

    turns_by_bucket: dict[int, list[ArenaTurn]] = {}
    for day in sorted(facts_by_day):
        bucket = day // SESSION_INTERVAL_DAYS
        for speaker, content in sorted(facts_by_day[day]):
            turns_by_bucket.setdefault(bucket, []).append(
                ArenaTurn(
                    session_index=bucket,
                    speaker=speaker,
                    content=content,
                    day=day,
                )
            )

    scenarios: list[ArenaScenario] = []
    for bucket in sorted(turns_by_bucket):
        turns = tuple(turns_by_bucket[bucket])
        scenarios.append(
            ArenaScenario(
                scenario_id=f"session-{bucket:02d}",
                title=SESSION_TITLES[bucket % len(SESSION_TITLES)],
                day=turns[0].day,
                turns=turns,
            )
        )
    return tuple(scenarios)

@dataclass(frozen=True)
class ArenaDataset:
    """The frozen Track B dataset: one shared history plus 200 questions."""

    version: str
    scenarios: tuple[ArenaScenario, ...]
    questions: tuple[ArenaQuestion, ...]
    created_at: str = ""

    # ---------- Queries ----------

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def session_count(self) -> int:
        return len(self.scenarios)

    @property
    def turn_count(self) -> int:
        return sum(len(scenario.turns) for scenario in self.scenarios)

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {category.value: 0 for category in ArenaCategory}
        for question in self.questions:
            counts[question.category.value] += 1
        return counts

    def by_category(self, category: ArenaCategory) -> tuple[ArenaQuestion, ...]:
        return tuple(q for q in self.questions if q.category is category)

    def by_id(self, question_id: str) -> ArenaQuestion | None:
        for question in self.questions:
            if question.question_id == question_id:
                return question
        return None

    def scenario(self, session_index: int) -> ArenaScenario | None:
        for scenario in self.scenarios:
            if scenario.scenario_id == f"session-{session_index:02d}":
                return scenario
        return None

    # ---------- Freeze ----------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "questions": [question.to_dict() for question in self.questions],
        }

    def canonical_json(self) -> str:
        """Canonical serialization: sorted keys, no whitespace, stable order.

        Hashing this string is what makes the freeze verifiable; any change to
        a question, a planted fact or a timestamp changes the hash.
        """
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def build_dataset(version: str = ARENA_VERSION, created_at: str = "") -> ArenaDataset:
    """Build the dataset. ``created_at`` defaults to a fixed empty value.

    A wall-clock timestamp would make the content hash differ per run, so
    callers that want provenance must pass an explicit value. Fairness Rule 6
    (reproducibility) depends on this.
    """
    return ArenaDataset(
        version=version,
        scenarios=build_scenarios(),
        questions=build_questions(),
        created_at=created_at,
    )

# ==================== Persistence and freeze ====================

ARENA_DATASET_DIRNAME = "benchmark/dataset"
ARENA_JSON_NAME = "arena.json"
ARENA_HASH_NAME = "arena.sha256"


def arena_dataset_dir(root: Path | str = ".") -> Path:
    return Path(root) / ARENA_DATASET_DIRNAME


def write_dataset(
    dataset: ArenaDataset,
    directory: Path | str | None = None,
    root: Path | str = ".",
) -> tuple[Path, Path]:
    """Persist the dataset and its content hash (Fairness Rule 8).

    The hash file is written next to the dataset so a later run can prove that
    the questions did not change after results were observed (Fairness Rule 1).
    Returns ``(dataset_path, hash_path)``.
    """
    target_dir = Path(directory) if directory is not None else arena_dataset_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = target_dir / ARENA_JSON_NAME
    hash_path = target_dir / ARENA_HASH_NAME
    content_hash = dataset.content_hash()

    dataset_path.write_text(
        json.dumps(dataset.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    hash_path.write_text(f"{content_hash}  {ARENA_JSON_NAME}\n", encoding="utf-8")
    return dataset_path, hash_path


def read_frozen_hash(
    directory: Path | str | None = None,
    root: Path | str = ".",
) -> str | None:
    """Return the previously frozen hash, or ``None`` when not yet frozen."""
    target_dir = Path(directory) if directory is not None else arena_dataset_dir(root)
    hash_path = target_dir / ARENA_HASH_NAME
    if not hash_path.exists():
        return None
    line = hash_path.read_text(encoding="utf-8").strip()
    return line.split()[0] if line else None


def verify_freeze(
    directory: Path | str | None = None,
    root: Path | str = ".",
    version: str = ARENA_VERSION,
) -> tuple[bool, str, str]:
    """Re-generate the dataset and compare it against the frozen hash.

    Returns ``(is_frozen, frozen_hash, current_hash)``. A ``False`` result means
    the questions, the storylines or the timestamps changed after the freeze -
    which invalidates comparability of any results collected before the change.
    """
    current_hash = build_dataset(version=version).content_hash()
    frozen_hash = read_frozen_hash(directory, root)
    return (frozen_hash == current_hash, frozen_hash or "", current_hash)


__all__ = [
    "ARENA_DATASET_DIRNAME",
    "ARENA_EPOCH",
    "ARENA_HASH_NAME",
    "ARENA_JSON_NAME",
    "ARENA_VERSION",
    "QUESTIONS_PER_CATEGORY",
    "SESSION_COUNT",
    "SESSION_INTERVAL_DAYS",
    "TOTAL_QUESTIONS",
    "ArenaCategory",
    "ArenaDataset",
    "ArenaQuestion",
    "ArenaScenario",
    "ArenaTurn",
    "arena_dataset_dir",
    "build_dataset",
    "build_questions",
    "build_scenarios",
    "read_frozen_hash",
    "verify_freeze",
    "write_dataset",
]
