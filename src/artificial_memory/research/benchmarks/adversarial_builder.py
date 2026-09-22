"""AM Adversarial Collision Benchmark Builder (Phase 3).

Constructs targeted adversarial collision scenarios designed to stress-test
cognitive memory systems:
1. Entity Collision (Project Atlas vs Atlas Company vs Atlas Character vs Atlas Dataset)
2. Semantic Collision (Redis vs Redis-compatible vs Redis discussion vs Redis abandoned)
3. Temporal Collision (Multi-hop timeline updates: 2025 -> 2026-01 -> 2026-03 -> 2026-06)
4. Contradiction & Truth vs Evidence (Current Decision vs Historical Utterance)
"""

from __future__ import annotations

from artificial_memory.research.benchmarks.arena import (
    ArenaCategory,
    ArenaDataset,
    ArenaQuestion,
    ArenaScenario,
    ArenaTurn,
)
from artificial_memory.research.benchmarks.generator import DifficultyTier, ScaledQuestion


class AdversarialBuilder:
    """Builds extreme adversarial collision test suites."""

    def build_adversarial_suite(self) -> ArenaDataset:
        scenarios: list[ArenaScenario] = []
        questions: list[ArenaQuestion] = []

        # ====================================================================
        # 1. Entity Collision
        # ====================================================================
        turns_entity = [
            ArenaTurn(1, "user", "For Project Atlas, the team selected ClickHouse as the primary database.", 0),
            ArenaTurn(1, "user", "Atlas Company, our key vendor, uses Oracle DB for their enterprise records.", 0),
            ArenaTurn(1, "user", "In our game project, the Atlas Character has a local inventory stored in SQLite.", 0),
            ArenaTurn(1, "user", "The Atlas Dataset was generated and exported to Parquet files on S3.", 0),
        ]
        sc_entity = ArenaScenario("sc-adv-01-entity-collision", "Entity Collision", 0, tuple(turns_entity))
        q_entity = ScaledQuestion(
            question_id="q-adv-01-entity-collision",
            category=ArenaCategory.DISTRACTOR_RESISTANCE,
            question="Which database was selected for Project Atlas?",
            ground_truth=(("ClickHouse",),),
            forbidden=("Oracle", "SQLite", "Parquet"),
            expected_abstention=False,
            difficulty=DifficultyTier.ADVERSARIAL,
        )
        scenarios.append(sc_entity)
        questions.append(q_entity)

        # ====================================================================
        # 2. Semantic Collision
        # ====================================================================
        turns_semantic = [
            ArenaTurn(1, "user", "In yesterday's meeting, the team had a long discussion about Redis.", 0),
            ArenaTurn(1, "user", "We tested Dragonfly as a Redis-compatible database, but latency was unstable.", 1),
            ArenaTurn(1, "user", "The proposed Redis migration was officially abandoned due to memory overhead.", 2),
            ArenaTurn(1, "user", "For our production environment, the active in-memory cache layer is Memcached.", 3),
        ]
        sc_semantic = ArenaScenario("sc-adv-02-semantic-collision", "Semantic Collision", 0, tuple(turns_semantic))
        q_semantic = ScaledQuestion(
            question_id="q-adv-02-semantic-collision",
            category=ArenaCategory.FACTUAL_RECALL,
            question="What is the active in-memory cache layer currently in production?",
            ground_truth=(("Memcached",),),
            forbidden=("Redis", "Dragonfly"),
            expected_abstention=False,
            difficulty=DifficultyTier.ADVERSARIAL,
        )
        scenarios.append(sc_semantic)
        questions.append(q_semantic)

        # ====================================================================
        # 3. Temporal Collision (Multi-step timeline)
        # ====================================================================
        turns_temporal = [
            ArenaTurn(1, "user", "In 2025, the system ran on PostgreSQL.", 0),
            ArenaTurn(2, "user", "On 2026-01-15, the team migrated the database to Redis.", 15),
            ArenaTurn(3, "user", "On 2026-03-01, the team migrated the database from Redis to ClickHouse.", 60),
            ArenaTurn(4, "user", "On 2026-06-01, due to unexpected query patterns, the team migrated back to PostgreSQL.", 150),
        ]
        sc_temporal = ArenaScenario("sc-adv-03-temporal-collision", "Temporal Collision", 0, tuple(turns_temporal))
        # Question 3A: Point-in-time query (2026-04)
        q_temp_pit = ScaledQuestion(
            question_id="q-adv-03a-temporal-point-in-time",
            category=ArenaCategory.TEMPORAL_REASONING,
            question="Which database was active on 2026-04-10?",
            ground_truth=(("ClickHouse",),),
            forbidden=("Redis", "PostgreSQL"),
            expected_abstention=False,
            difficulty=DifficultyTier.ADVERSARIAL,
        )
        # Question 3B: Current state query
        q_temp_curr = ScaledQuestion(
            question_id="q-adv-03b-temporal-current-state",
            category=ArenaCategory.KNOWLEDGE_UPDATES,
            question="What is the current active database for the system?",
            ground_truth=(("PostgreSQL",),),
            forbidden=("Redis", "ClickHouse"),
            expected_abstention=False,
            difficulty=DifficultyTier.ADVERSARIAL,
        )
        scenarios.append(sc_temporal)
        questions.extend([q_temp_pit, q_temp_curr])

        # ====================================================================
        # 4. Contradiction: Truth Retrieval vs Evidence Retrieval
        # ====================================================================
        turns_contradiction = [
            ArenaTurn(1, "user", "Alice proposed: 'We must use Redis for the queue.'", 0),
            ArenaTurn(1, "assistant", "Bob responded: 'No, we formally decided on PostgreSQL.'", 0),
            ArenaTurn(1, "user", "Alice replied later: 'Actually, Redis is back on the table, but Bob rejected it again. PostgreSQL remains the final decision.'", 1),
        ]
        sc_contradiction = ArenaScenario("sc-adv-04-truth-vs-evidence", "Truth vs Evidence", 0, tuple(turns_contradiction))
        # Question 4A: Truth Retrieval (What was decided?)
        q_truth = ScaledQuestion(
            question_id="q-adv-04a-truth-retrieval",
            category=ArenaCategory.FACTUAL_RECALL,
            question="What was the final decision regarding the queue technology?",
            ground_truth=(("PostgreSQL",),),
            forbidden=("Redis",),
            expected_abstention=False,
            difficulty=DifficultyTier.ADVERSARIAL,
        )
        # Question 4B: Evidence Retrieval (Did Alice ever propose Redis?)
        q_evidence = ScaledQuestion(
            question_id="q-adv-04b-evidence-retrieval",
            category=ArenaCategory.FACTUAL_RECALL,
            question="Did Alice propose using Redis for the queue at any point?",
            ground_truth=(("yes", "proposed", "did propose"),),
            forbidden=("no", "never"),
            expected_abstention=False,
            difficulty=DifficultyTier.ADVERSARIAL,
        )
        scenarios.append(sc_contradiction)
        questions.extend([q_truth, q_evidence])

        return ArenaDataset(
            version="v0.2.0-adversarial",
            scenarios=tuple(scenarios),
            questions=tuple(questions),
        )
