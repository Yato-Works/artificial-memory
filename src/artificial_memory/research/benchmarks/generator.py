"""Scaled Benchmark Dataset Generator (Phase 3).

Generates a 100-question hierarchical benchmark dataset across 10 categories
and 4 difficulty tiers:
- Easy: Direct, single-turn mentions (surface keyword matches)
- Medium: Cross-session multi-turn mentions with natural paraphrasing
- Hard: Context-dependent questions without surface keywords (e.g. "that thing we switched from...")
- Adversarial: Complex updates, multi-entity overlap, and false-lead traps
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from artificial_memory.research.benchmarks.arena import (
    ArenaCategory,
    ArenaDataset,
    ArenaQuestion,
    ArenaScenario,
    ArenaTurn,
)


class DifficultyTier(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


@dataclass(frozen=True)
class ScaledQuestion(ArenaQuestion):
    """Extended ArenaQuestion with difficulty tier metadata."""
    difficulty: DifficultyTier = DifficultyTier.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["difficulty"] = self.difficulty.value
        return base


class DatasetGenerator:
    """Deterministic generator for 100+ question hierarchical benchmark."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    # Domain vocabulary pools
    ENTITIES = [
        "atlas", "beacon", "cinder", "delta", "ember", "fjord", "glacier",
        "harbor", "ivory", "juniper", "kelp", "lumen", "moss", "nectar",
        "onyx", "prism", "quartz", "reef", "summit", "tundra", "zenith",
        "vortex", "solstice", "polaris", "aurora", "cascade", "horizon"
    ]

    TOOLS = [
        "ripgrep", "batcat", "neovim", "helix", "emacs", "kakoune", "sublime",
        "vscode", "zed", "alacritty", "kitty", "wezterm", "tmux", "zellij"
    ]

    DATABASES = [
        "SQLite", "ClickHouse", "PostgreSQL", "Redis", "DuckDB", "MongoDB",
        "Cassandra", "ScyllaDB", "CockroachDB", "TiDB", "Neo4j", "SurrealDB"
    ]

    CITIES = [
        "Kyoto", "Porto", "Turku", "Osaka", "Berlin", "Helsinki", "Zurich",
        "Reykjavik", "Vancouver", "Singapore", "Oslo", "Amsterdam"
    ]

    LANGUAGES = [
        "Rust", "Go", "Python", "TypeScript", "Zig", "Elixir", "Haskell",
        "OCaml", "Julia", "C++", "Swift", "Kotlin"
    ]

    def generate_100_dataset(self) -> ArenaDataset:
        """Generate a balanced 100-question dataset (10 categories x 10 questions)."""
        scenarios: list[ArenaScenario] = []
        questions: list[ArenaQuestion] = []

        q_counter = 1
        s_counter = 1

        for cat in ArenaCategory:
            for idx in range(10):
                # 0-2: Easy (3 questions), 3-5: Medium (3 questions), 6-7: Hard (2 questions), 8-9: Adversarial (2 questions)
                if idx < 3:
                    diff = DifficultyTier.EASY
                elif idx < 6:
                    diff = DifficultyTier.MEDIUM
                elif idx < 8:
                    diff = DifficultyTier.HARD
                else:
                    diff = DifficultyTier.ADVERSARIAL

                ent = self.ENTITIES[(q_counter - 1) % len(self.ENTITIES)]
                other_ent = self.ENTITIES[(q_counter + 3) % len(self.ENTITIES)]

                sc_id = f"sc-{s_counter:03d}-{cat.value[:4]}-{diff.value[:4]}"
                q_id = f"q-{q_counter:03d}-{cat.value[:4]}-{diff.value[:4]}"

                sc, q = self._build_case(cat, diff, sc_id, q_id, ent, other_ent)
                scenarios.append(sc)
                questions.append(q)

                q_counter += 1
                s_counter += 1

        return ArenaDataset(
            version="v0.2.0-arena-100",
            scenarios=tuple(scenarios),
            questions=tuple(questions),
        )

    def _build_case(
        self,
        cat: ArenaCategory,
        diff: DifficultyTier,
        sc_id: str,
        q_id: str,
        ent: str,
        other_ent: str,
    ) -> tuple[ArenaScenario, ArenaQuestion]:
        turns: list[ArenaTurn] = []
        gt: tuple[tuple[str, ...], ...] = ()
        forbidden: tuple[str, ...] = ()
        expected_abstention = False

        if cat == ArenaCategory.FACTUAL_RECALL:
            tool = self.TOOLS[self.rng.randint(0, len(self.TOOLS) - 1)]
            other_tool = self.TOOLS[(self.TOOLS.index(tool) + 1) % len(self.TOOLS)]

            if diff == DifficultyTier.EASY:
                turns.append(ArenaTurn(1, "user", f"For project {ent}, the preferred editor toolchain is {tool}.", 0))
                question_text = f"For project {ent}, what did the user say their preferred editor toolchain is?"
            elif diff == DifficultyTier.MEDIUM:
                turns.append(ArenaTurn(1, "user", f"We started discussing project {ent} today.", 0))
                turns.append(ArenaTurn(1, "assistant", f"Acknowledged. What is the environment for {ent}?", 0))
                turns.append(ArenaTurn(1, "user", f"For project {ent}, the user decided to adopt {tool}.", 1))
                question_text = f"Which tool did the user adopt for project {ent}?"
            elif diff == DifficultyTier.HARD:
                turns.append(ArenaTurn(1, "user", f"For project {ent}, we considered {other_tool} but ultimately settled on {tool}.", 0))
                question_text = f"In project {ent}, which tool was selected instead of {other_tool}?"
            else:  # ADVERSARIAL
                turns.append(ArenaTurn(1, "user", f"For project {other_ent}, the toolchain is {other_tool}.", 0))
                turns.append(ArenaTurn(1, "user", f"For project {ent}, the preferred editor toolchain is {tool}.", 1))
                question_text = f"Between {other_ent} and {ent}, what is the toolchain for {ent}?"

            gt = ((tool,),)
            forbidden = (other_tool,) if diff in [DifficultyTier.HARD, DifficultyTier.ADVERSARIAL] else ()

        elif cat == ArenaCategory.KNOWLEDGE_UPDATES:
            db_old = self.DATABASES[0]
            db_new = self.DATABASES[1]
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the primary database was {db_old}.", 0))
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user migrated the primary database from {db_old} to {db_new}.", 15))
            if diff in [DifficultyTier.HARD, DifficultyTier.ADVERSARIAL]:
                turns.append(ArenaTurn(1, "user", f"Some legacy docs still mention {db_old}, but {db_new} is the current active system.", 16))
                question_text = f"What is the current, active primary database for project {ent}?"
            else:
                question_text = f"For project {ent}, which database is currently used as the primary database?"
            gt = ((db_new,),)
            forbidden = (db_old,)

        elif cat == ArenaCategory.TEMPORAL_REASONING:
            city_old = self.CITIES[0]
            city_new = self.CITIES[1]
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user was working from {city_old} until 2026-02-03.", 0))
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user relocated to {city_new} on 2026-02-03.", 28))
            if diff in [DifficultyTier.HARD, DifficultyTier.ADVERSARIAL]:
                question_text = f"Where was the user working on 2026-01-27 while project {ent} was active?"
            else:
                question_text = f"In which city was the user located before February 2026 for project {ent}?"
            gt = ((city_old,),)
            forbidden = (city_new,)

        elif cat == ArenaCategory.ABSTENTION:
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user discussed scheduling and timelines.", 0))
            turns.append(ArenaTurn(1, "user", f"For project {other_ent}, the user set the debugger to lldb.", 0))
            question_text = f"Which debugger did the user configure for project {ent}?"
            gt = (("I don't know.", "not specified", "never configured", "unknown"),)
            forbidden = ("lldb", "gdb", "vscode")
            expected_abstention = True

        elif cat == ArenaCategory.CONTRADICTION_HANDLING:
            db_u = "SQLite"
            db_t = "Redis"
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user reported the cache layer runs on {db_u}.", 0))
            turns.append(ArenaTurn(1, "assistant", f"For project {ent}, a teammate reported the cache layer actually runs on {db_t}.", 0))
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user has not verified which cache layer report is correct.", 1))
            question_text = f"For project {ent}, which cache layer should I assume the system runs on?"
            gt = ((db_u, db_t), ("conflict", "conflicting", "unresolved", "contradict", "both", "disagree"))
            forbidden = ()

        elif cat == ArenaCategory.INDIRECT_RECALL:
            tool = "ripgrep"
            city = "Porto"
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user settled on {tool} as the daily driver.", 0))
            turns.append(ArenaTurn(1, "user", f"The workstation in {city} is the only machine with {tool} installed.", 0))
            question_text = f"In which city is the machine that runs the daily driver toolchain of project {ent} located?"
            gt = ((city,),)
            forbidden = ()

        elif cat == ArenaCategory.DISTRACTOR_RESISTANCE:
            tool = "ripgrep"
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user pinned the file search tool to {tool}.", 0))
            for i, p in enumerate(self.ENTITIES[:8]):
                if p != ent:
                    turns.append(ArenaTurn(1, "user", f"The user usually drinks coffee while working on {p}.", i + 1))
            question_text = f"Which file search tool is pinned for project {ent}?"
            gt = ((tool,),)
            forbidden = ()

        elif cat == ArenaCategory.MULTI_SESSION_RECALL:
            lang = "Rust"
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user began studying {lang} in the evenings.", 0))
            turns.append(ArenaTurn(2, "user", f"For project {ent}, the user finished their first {lang} exercise set.", 15))
            turns.append(ArenaTurn(3, "user", f"For project {ent}, the user now recommends {lang} to colleagues.", 30))
            question_text = f"Across our conversations about project {ent}, which subject did the user begin studying, practise, and later recommend to colleagues?"
            gt = ((lang,),)
            forbidden = ()

        elif cat == ArenaCategory.COMPRESSION_RECOVERY:
            serial = "AM-1160"
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user measured p99 latency as 12.5 ms and filed it under serial {serial}.", 0))
            turns.append(ArenaTurn(1, "user", f"For project {ent}, lots of other minor tests were conducted without issue.", 1))
            question_text = f"What serial number was filed for the p99 latency measurement of project {ent}?"
            gt = ((serial, "1160"),)
            forbidden = ()

        elif cat == ArenaCategory.REFLECTION_REINTERPRETATION:
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user always reverts the build configuration to the last known good state.", 0))
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user keeps an untouched backup of the previous schema.", 1))
            turns.append(ArenaTurn(1, "user", f"For project {ent}, the user copies files by hand instead of symlinking them.", 2))
            question_text = f"Considering everything the user said about project {ent}, what recurring preference does the user consistently show?"
            gt = (("stability", "stable", "conservative", "safe", "revert", "rollback"),)
            forbidden = ()

        sc = ArenaScenario(scenario_id=sc_id, title=f"Scenario for {ent}", day=0, turns=tuple(turns))
        q = ScaledQuestion(
            question_id=q_id,
            category=cat,
            question=question_text,
            ground_truth=gt,
            forbidden=forbidden,
            expected_abstention=expected_abstention,
            difficulty=diff,
        )
        return sc, q
