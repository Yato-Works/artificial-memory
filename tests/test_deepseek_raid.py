"""Tests for the DeepSeek Raid retrieval acceleration (Phase 8.4).

Covers the three raid primitives and their composition:

- ``RetrievalPlanCache``: FULL/REINDEX/REUSE tiering, query families, TTL,
  eviction, determinism.
- ``HierarchicalGate``: pool narrowing, no-op threshold, order-stable
  tie-breaks.
- ``EphemeralStore``: scratch get/put, bounded replay, discard semantics.
- ``DeepSeekRecallEngine``: composition with the frozen BasicRecallEngine —
  candidate gating, CSA2 tiering with side-effect parity, scorer-compatible
  results.
- Runtime opt-in wiring: "classic" (frozen) vs "deepseek" (raid).

The frozen Scorer and the S0-S3 baseline players must be untouched by this
phase; a small regression block at the bottom asserts that.
"""

from __future__ import annotations

import asyncio

import pytest

from artificial_memory.core.models import Memory, MemoryType, RecallLevel, ResolutionLevel
from artificial_memory.recall.engine import BasicRecallEngine
from artificial_memory.recall.retrieval_cache import (
    FULL,
    REINDEX,
    REUSE,
    EphemeralStore,
    HierarchicalGate,
    RetrievalPlan,
    RetrievalPlanCache,
)
from artificial_memory.recall.deepseek_engine import DeepSeekRecallEngine
from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_memory(content: str, memory_id: int | None = None) -> Memory:
    return Memory(
        id=memory_id,
        topic_id=1,
        memory_type=MemoryType.EPISODE,
        content=content,
        resolution=ResolutionLevel.EPISODE,
    )


class CountingCache(RetrievalPlanCache):
    """Plan cache that counts fresh retrievals for miss accounting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fresh_calls = 0


# ---------------------------------------------------------------------------
# RetrievalPlanCache (CSA2)
# ---------------------------------------------------------------------------

class TestRetrievalPlanCache:
    def test_first_query_is_full(self):
        cache = RetrievalPlanCache()
        calls = []

        def fresh(q):
            calls.append(q)
            return [1, 2], []

        plan, mode = cache.lookup_or_plan("What is X?", fresh)
        assert mode == FULL
        assert plan.candidate_ids == (1, 2)
        assert calls == ["What is X?"]

    def test_identical_query_reuses(self):
        cache = RetrievalPlanCache()
        calls = []

        def fresh(q):
            calls.append(q)
            return [1], []

        cache.lookup_or_plan("What is X?", fresh)
        plan, mode = cache.lookup_or_plan("What is X?", fresh)
        assert mode == REUSE
        assert calls == ["What is X?"]  # fresh run only once
        assert plan.reuse_count == 1

    def test_rephrase_reindexes(self):
        cache = RetrievalPlanCache()
        calls = []

        def fresh(q):
            calls.append(q)
            return [1], []

        cache.lookup_or_plan("what is the project codename", fresh)
        plan, mode = cache.lookup_or_plan("what is the project codename?", fresh)
        assert mode == REINDEX  # punctuation differs -> not verbatim REUSE
        assert calls == ["what is the project codename"]
        assert plan.candidate_ids == (1,)

    def test_signature_drops_punctuation_keeps_order(self):
        sig_a = RetrievalPlanCache._signature("Why did X fail?")
        sig_b = RetrievalPlanCache._signature("why did X fail")
        sig_c = RetrievalPlanCache._signature("How did X fail?")
        assert sig_a == sig_b
        assert sig_a != sig_c  # word order preserved

    def test_ttl_expiry_returns_full(self):
        import time as _t

        cache = RetrievalPlanCache(ttl_seconds=0.0)
        calls = []

        def fresh(q):
            calls.append(q)
            return [1], []

        cache.lookup_or_plan("same query", fresh)
        _t.sleep(0.01)
        _plan, mode = cache.lookup_or_plan("same query", fresh)
        assert mode == FULL
        assert len(calls) == 2

    def test_eviction_is_insertion_order(self):
        cache = RetrievalPlanCache(max_families=2)

        def fresh(q):
            return [1], []

        cache.lookup_or_plan("alpha beta", fresh)
        cache.lookup_or_plan("gamma delta", fresh)
        cache.lookup_or_plan("epsilon zeta", fresh)
        # "alpha beta" was evicted -> next lookup on it is FULL again.
        _plan, mode = cache.lookup_or_plan("alpha beta", fresh)
        assert mode == FULL
        assert cache.stats_snapshot()[FULL] == 4

    def test_stats_are_counted(self):
        cache = RetrievalPlanCache()

        def fresh(q):
            return [1], []

        cache.lookup_or_plan("q one", fresh)
        cache.lookup_or_plan("q one", fresh)   # REUSE
        cache.lookup_or_plan("q one?", fresh)  # REINDEX
        stats = cache.stats_snapshot()
        assert stats == {FULL: 1, REINDEX: 1, REUSE: 1}


# ---------------------------------------------------------------------------
# HierarchicalGate (HSI)
# ---------------------------------------------------------------------------

class TestHierarchicalGate:
    def _pool(self, n: int) -> list[Memory]:
        return [make_memory(f"memory about topic {i}", memory_id=i) for i in range(n)]

    def test_small_pool_is_noop(self):
        gate = HierarchicalGate(pool_size=64)
        pool = self._pool(10)
        assert gate.filter("anything", pool) is pool

    def test_large_pool_narrows_to_pool_size(self):
        gate = HierarchicalGate(pool_size=8, min_keep=2)
        pool = self._pool(100)
        out = gate.filter("memory about topic 42", pool)
        assert len(out) <= 8

    def test_relevant_memory_survives(self):
        gate = HierarchicalGate(pool_size=4, min_keep=1)
        pool = [make_memory(f"unrelated filler {i}", memory_id=i) for i in range(50)]
        target = make_memory("the launch codename is Blue Falcon", memory_id=999)
        out = gate.filter("what is the launch codename?", pool + [target])
        assert target in out

    def test_tie_break_keeps_earliest(self):
        gate = HierarchicalGate(pool_size=2, min_keep=1)
        a = make_memory("alpha alpha", memory_id=1)
        b = make_memory("alpha alpha", memory_id=2)
        out = gate.filter("alpha", [a, b])
        assert out[0] is a  # earliest index wins on equal score


# ---------------------------------------------------------------------------
# EphemeralStore (Bounded Replay)
# ---------------------------------------------------------------------------

class TestEphemeralStore:
    def test_put_get_roundtrip(self):
        store = EphemeralStore()
        store.put("k", "v")
        assert store.get("k") == "v"

    def test_replay_rebuilds_missing_key(self):
        store = EphemeralStore(replay=lambda k: f"rebuilt:{k}")
        assert store.get("missing") == "rebuilt:missing"
        assert store.replays == 1
        assert store.get("missing") == "rebuilt:missing"  # now cached
        assert store.replays == 1

    def test_no_replay_closure_returns_none(self):
        store = EphemeralStore()
        assert store.get("missing") is None

    def test_bounded_fifo_eviction(self):
        store = EphemeralStore(max_keys=2)
        store.put("a", 1)
        store.put("b", 2)
        store.put("c", 3)
        assert store.keys() == ["b", "c"]
        assert store.discards == 1

    def test_discard_all_does_not_persist(self):
        store = EphemeralStore(replay=lambda k: "x")
        store.put("a", 1)
        store.discard_all()
        assert len(store) == 0
        # Rebuilt from source after discard (Bounded Replay core).
        assert store.get("a") == "x"


# ---------------------------------------------------------------------------
# DeepSeekRecallEngine (composition)
# ---------------------------------------------------------------------------

class TestDeepSeekRecallEngine:
    @staticmethod
    def _seeded_engine(contents, gate=None):
        from artificial_memory.core.models import Project, Topic

        store = SQLiteMemoryStore(":memory:")
        project = store.create_project(Project(name="t"))
        topic = store.create_topic(
            Topic(project_id=project.id, name="T", path="T")
        )
        for text in contents:
            store.create_memory(Memory(
                topic_id=topic.id, memory_type=MemoryType.EPISODE,
                content=text, resolution=ResolutionLevel.EPISODE,
            ))
        return DeepSeekRecallEngine(store, gate=gate), topic.id

    def test_full_then_reuse_same_selection(self):
        engine, topic_id = self._seeded_engine(
            ["alpha report", "beta launch", "alpha summary"]
        )
        m1, t1 = engine.recall(
            "alpha report", topic_id=topic_id, level=RecallLevel.CURRENT_ONLY
        )
        m2, t2 = engine.recall(
            "alpha report", topic_id=topic_id, level=RecallLevel.CURRENT_ONLY
        )
        assert [x.id for x in m1] == [x.id for x in m2]
        assert t1 == t2
        assert engine.plan_cache.stats_snapshot()[REUSE] == 1

    def test_gate_narrows_candidates(self):
        contents = [f"filler number {i} with unrelated words" for i in range(30)]
        contents.append("the special zebra crosses the river")
        engine, topic_id = self._seeded_engine(
            contents, gate=HierarchicalGate(pool_size=8)
        )
        memories, _ = engine.recall(
            "where does the special zebra go?",
            topic_id=topic_id, level=RecallLevel.CURRENT_ONLY,
        )
        assert engine.gate_applications >= 1
        assert any("zebra" in m.content for m in memories)

    def test_frozen_engine_untouched(self):
        """BasicRecallEngine must have no plan cache / gate attributes."""
        engine = BasicRecallEngine(SQLiteMemoryStore(":memory:"))
        assert not hasattr(engine, "plan_cache")
        assert not hasattr(engine, "gate")
        assert not hasattr(engine, "candidate_window")


# ---------------------------------------------------------------------------
# Candidate window (Phase 8.5 funnel-audit finding)
# ---------------------------------------------------------------------------

class TestCandidateWindow:
    @staticmethod
    def _seeded_store(n: int):
        from artificial_memory.core.models import Project, Topic

        store = SQLiteMemoryStore(":memory:")
        project = store.create_project(Project(name="t"))
        topic = store.create_topic(
            Topic(project_id=project.id, name="T", path="T")
        )
        for i in range(n):
            store.create_memory(Memory(
                topic_id=topic.id, memory_type=MemoryType.EPISODE,
                content=f"memory {i} filler {i * 7919 % 101}",  # spread lexicon
                resolution=ResolutionLevel.SEMANTIC,
            ))
        return store, topic.id

    def test_window_none_matches_frozen_exactly(self):
        """window=None must reproduce the frozen candidate fetch byte-identically."""
        store, topic_id = self._seeded_store(120)
        frozen = BasicRecallEngine(store)
        raid = DeepSeekRecallEngine(store)
        assert raid.candidate_window is None
        frozen_pool = frozen._get_candidates(
            "memory 3 filler", topic_id, RecallLevel.CURRENT_ONLY
        )
        raid_pool = raid._get_candidates(
            "memory 3 filler", topic_id, RecallLevel.CURRENT_ONLY
        )
        assert [m.id for m in frozen_pool] == [m.id for m in raid_pool]
        assert len(frozen_pool) == 50  # frozen per-level cap

    def test_window_widens_candidates(self):
        store, topic_id = self._seeded_store(120)
        wide = DeepSeekRecallEngine(store, candidate_window=200)
        pool = wide._get_candidates(
            "memory 3 filler", topic_id, RecallLevel.CURRENT_ONLY
        )
        # Window is an upper bound: capped by store size (120 < 200), but
        # strictly wider than the frozen recent-50 window.
        assert len(pool) == 120
        assert len(pool) > 50

    def test_window_covers_store(self):
        """A window >= store size must surface every active memory (level 0)."""
        store, topic_id = self._seeded_store(80)
        wide = DeepSeekRecallEngine(store, candidate_window=450)
        pool = wide._get_candidates(
            "anything", topic_id, RecallLevel.CURRENT_ONLY
        )
        assert len(pool) == 80

    def test_window_applies_per_bucket(self):
        """Window W is a per-bucket fetch bound, not a ratio-scaled total.

        Deliberate Phase 8.5 design: ratio-scaling a widened window (e.g.
        450 * 20/60 = 150 per bucket) would still hide older evidence when
        buckets are unevenly filled (a 100%-semantic store gets 450 * 40/60
        for its only bucket... capped at the frozen level split, never the
        full store).  Per-bucket W means "how deep into each partition we
        look", so W >= store size surfaces every active memory.
        """
        from artificial_memory.core.models import Project, Topic

        store = SQLiteMemoryStore(":memory:")
        project = store.create_project(Project(name="t"))
        topic = store.create_topic(
            Topic(project_id=project.id, name="T", path="T")
        )
        for i in range(45):
            store.create_memory(Memory(
                topic_id=topic.id, memory_type=MemoryType.EPISODE,
                content=f"episode {i}", resolution=ResolutionLevel.EPISODE,
            ))
            store.create_memory(Memory(
                topic_id=topic.id, memory_type=MemoryType.SEMANTIC,
                content=f"semantic {i}", resolution=ResolutionLevel.SEMANTIC,
            ))
        wide = DeepSeekRecallEngine(store, candidate_window=60)
        pool = wide._get_candidates("q", topic.id, RecallLevel.EPISODE)
        episodes = sum(1 for m in pool if m.resolution == ResolutionLevel.EPISODE)
        semantics = sum(1 for m in pool if m.resolution == ResolutionLevel.SEMANTIC)
        # Per-bucket W=60 with 45 available per bucket: both buckets fill
        # completely (ratio-scaling would give the frozen 40:20 instead).
        assert episodes == 45
        assert semantics == 45
        assert len(pool) == 90

    def test_frozen_engine_still_has_hardcoded_window(self):
        """The frozen engine's own path must be untouched by the raid window."""
        store, topic_id = self._seeded_store(120)
        frozen = BasicRecallEngine(store)
        pool = frozen._get_candidates(
            "anything", topic_id, RecallLevel.CURRENT_ONLY
        )
        assert len(pool) == 50  # unchanged: recent-50 window


# ---------------------------------------------------------------------------
# Runtime opt-in wiring
# ---------------------------------------------------------------------------

class TestRuntimeWiring:
    @staticmethod
    def _config(strategy: str) -> RuntimeConfig:
        return RuntimeConfig(
            database_path=":memory:",
            memory_files_path=None,
            vector_index_path=None,
            retrieval_strategy=strategy,
        )

    def test_default_config_is_classic(self):
        runtime = ArtificialMemoryRuntime(RuntimeConfig(
            database_path=":memory:", memory_files_path=None, vector_index_path=None,
        ))
        assert type(runtime.recall_engine) is BasicRecallEngine

    def test_classic_is_frozen(self):
        runtime = ArtificialMemoryRuntime(self._config("classic"))
        assert type(runtime.recall_engine) is BasicRecallEngine

    def test_deepseek_wires_raid_engine(self):
        runtime = ArtificialMemoryRuntime(self._config("deepseek"))
        assert isinstance(runtime.recall_engine, DeepSeekRecallEngine)

    def test_deepseek_end_to_end_tiering(self):
        import asyncio

        runtime = ArtificialMemoryRuntime(self._config("deepseek"))

        async def scenario():
            await runtime.remember("The launch codename is Blue Falcon.", topic="Arena")
            await runtime.remember("Blue Falcon launches in March.", topic="Arena")
            r1 = await runtime.recall("What is the launch codename?", topic="Arena", level=2)
            r2 = await runtime.recall("What is the launch codename?", topic="Arena", level=2)
            return r1, r2

        r1, r2 = asyncio.run(scenario())
        assert r1.memories_retrieved == r2.memories_retrieved
        stats = runtime.recall_engine.plan_cache.stats_snapshot()
        assert stats[REUSE] >= 1


# ---------------------------------------------------------------------------
# SessionGate (Phase 8.7 winner) + deepseek_session runtime wiring
# ---------------------------------------------------------------------------

class TestSessionGate:
    @staticmethod
    def _memories(contents, session_ids=None):
        """Build in-memory ``Memory`` objects with *explicit unique ids*.

        ``SessionGate`` keys its corpus/IDF/session maps by ``Memory.id``, so a
        test fixture without ids would silently collapse every document into a
        single ``None`` key and pass vacuously.  Ids are therefore assigned
        here (1-based, matching store-assigned ids).
        """
        from datetime import datetime, timedelta

        memories = []
        base = datetime(2026, 1, 1, 12, 0, 0)
        for i, text in enumerate(contents):
            mem = Memory(
                topic_id=1,
                memory_type=MemoryType.SEMANTIC,
                content=text,
                resolution=ResolutionLevel.SEMANTIC,
                created_at=base + timedelta(seconds=i),
            )
            mem.id = i + 1
            memories.append(mem)
        return memories

    def test_sessions_kept_whole(self):
        """A session whose member scores high keeps *all* its members."""
        from artificial_memory.recall.retrieval_cache import SessionGate

        gate = SessionGate(pool_size=4)
        # Session A: one member mentions ripgrep, the rest are generic.
        contents = [
            "atlas daily driver is ripgrep",      # distinctive member
            "atlas notes follow-up",              # generic member of same session
            "atlas meeting minutes archived",     # generic member of same session
            "gardening filler one",
            "gardening filler two",
            "gardening filler three",
        ]
        memories = self._memories(contents)
        kept = gate.filter("which ripgrep tool does atlas use?", memories)
        assert len(kept) <= 4
        kept_ids = {m.id for m in kept}
        # The distinctive member survives...
        assert memories[0].id in kept_ids
        # ...and its whole session rides along (block behaviour).
        assert memories[1].id in kept_ids
        assert memories[2].id in kept_ids

    def test_small_pool_is_noop(self):
        from artificial_memory.recall.retrieval_cache import SessionGate

        gate = SessionGate(pool_size=100)
        memories = self._memories([f"filler {i}" for i in range(10)])
        kept = gate.filter("anything", memories)
        assert kept == memories

    def test_deterministic_order_stable(self):
        from artificial_memory.recall.retrieval_cache import SessionGate

        gate = SessionGate(pool_size=3)
        memories = self._memories(
            ["alpha topic", "beta topic", "gamma topic", "delta topic"]
        )
        first = gate.filter("alpha", memories)
        second = gate.filter("alpha", memories)
        assert [m.id for m in first] == [m.id for m in second]

    def test_explicit_sessions_override_derivation(self):
        from artificial_memory.recall.retrieval_cache import SessionGate

        memories = self._memories(["a one", "a two", "b one", "b two"])
        # Force all four into one session regardless of timestamps.
        explicit = {m.id: 0 for m in memories}
        gate = SessionGate(session_by_id=explicit, pool_size=2)
        kept = gate.filter("a one", memories)
        assert len(kept) <= 2

    def test_max_sessions_keeps_top_k_only(self):
        """Phase 8.10 lever: only the K best-scoring whole sessions survive."""
        from artificial_memory.recall.retrieval_cache import SessionGate

        memories = self._memories([
            "atlas daily driver is ripgrep",   # session 0, distinctive
            "atlas meeting minutes archived",  # session 0
            "gardening filler one",            # session 1
            "gardening filler two",            # session 1
            "cooking filler three",            # session 2
            "cooking filler four",             # session 2
        ])
        # Explicit session assignment (timing-derived sessions would collapse
        # these into one: the fixture spaces timestamps 1s apart).
        session_by_id = {
            memories[0].id: 0, memories[1].id: 0,
            memories[2].id: 1, memories[3].id: 1,
            memories[4].id: 2, memories[5].id: 2,
        }
        gate = SessionGate(
            session_by_id=session_by_id, pool_size=3, max_sessions=1
        )
        kept = gate.filter("which ripgrep tool does atlas use?", memories)
        kept_ids = {m.id for m in kept}
        # The high-scoring session survives whole...
        assert memories[0].id in kept_ids
        assert memories[1].id in kept_ids
        # ...and no member of a dropped session is kept.
        assert memories[2].id not in kept_ids
        assert memories[4].id not in kept_ids

    def test_max_sessions_deterministic(self):
        from artificial_memory.recall.retrieval_cache import SessionGate

        memories = self._memories(
            [f"filler {i} beta topic" for i in range(8)]
        )
        session_by_id = {m.id: i // 2 for i, m in enumerate(memories)}
        gate = SessionGate(
            session_by_id=session_by_id, pool_size=3, max_sessions=2
        )
        first = gate.filter("beta topic", memories)
        second = gate.filter("beta topic", memories)
        assert [m.id for m in first] == [m.id for m in second]

    def test_floor_applies_below_pool_size(self):
        """Phase 8.11: the floor must work even when pool <= pool_size.

        Regression for the silent no-op: the old early-return at
        ``len(memories) <= pool_size`` disabled the floor whenever the
        candidate window was small (measured: floor=8 with floor_drops=0
        across 60 validation runs).
        """
        from artificial_memory.recall.retrieval_cache import SessionGate

        memories = self._memories([
            "atlas daily driver is ripgrep",
            "gardening filler one",
            "gardening filler two",
        ])
        gate = SessionGate(pool_size=10, score_floor=0.5)
        kept = gate.filter("which ripgrep tool does atlas use?", memories)
        kept_ids = {m.id for m in kept}
        assert len(kept) < len(memories)      # floor fired
        assert memories[0].id in kept_ids     # evidenced memory survives
        assert gate.floor_drops == 2          # both fillers dropped

    def test_floor_zero_keeps_small_pool_intact(self):
        from artificial_memory.recall.retrieval_cache import SessionGate

        memories = self._memories(["alpha one", "beta two", "gamma three"])
        gate = SessionGate(pool_size=10, score_floor=0.0)
        kept = gate.filter("anything", memories)
        assert kept == memories
        assert gate.floor_drops == 0


class TestPlanCacheGateFlush:
    """Phase 8.11: the plan cache must not replay across gate configs."""

    @staticmethod
    def _runtime() -> "ArtificialMemoryRuntime":
        import asyncio  # noqa: F401  (used inside test closures)

        return ArtificialMemoryRuntime(RuntimeConfig(
            database_path=":memory:",
            memory_files_path=None,
            vector_index_path=None,
            retrieval_strategy="deepseek_session",
        ))

    def test_gate_swap_flushes_plan_cache(self):
        from artificial_memory.recall.retrieval_cache import SessionGate

        runtime = self._runtime()
        engine = runtime.recall_engine

        async def seed_and_recall():
            for i in range(1, 12):
                await runtime.remember(
                    f"turn {i}: project atlas uses ripgrep daily.", topic="T"
                )
            # Warm the plan cache under the runtime's default gate.
            await runtime.recall("atlas ripgrep", topic="T", level=2)

            # Swap in a gate with a different floor: the cached plans were
            # seeded by a different gate config and must not be replayed.
            old_cache = engine.plan_cache
            engine.gate = SessionGate(
                pool_size=8, corpus_provider=runtime._store_snapshot,
                score_floor=1.0,
            )
            await runtime.recall("atlas ripgrep", topic="T", level=2)
            return old_cache

        old_cache = asyncio.run(seed_and_recall())
        assert engine.plan_cache is not old_cache

    def test_same_gate_config_keeps_plans(self):
        runtime = self._runtime()
        engine = runtime.recall_engine

        async def scenario():
            for i in range(1, 12):
                await runtime.remember(
                    f"turn {i}: project atlas uses ripgrep daily.", topic="T"
                )
            await runtime.recall("atlas ripgrep", topic="T", level=2)
            old_cache = engine.plan_cache
            await runtime.recall("atlas ripgrep", topic="T", level=2)
            return old_cache

        old_cache = asyncio.run(scenario())
        assert engine.plan_cache is old_cache
        assert engine.plan_cache.stats[REUSE] >= 1


class TestDeepSeekSessionWiring:
    @staticmethod
    def _config(**overrides) -> RuntimeConfig:
        defaults = dict(
            database_path=":memory:",
            memory_files_path=None,
            vector_index_path=None,
            retrieval_strategy="deepseek_session",
        )
        defaults.update(overrides)
        return RuntimeConfig(**defaults)

    def test_wires_session_gate_engine(self):
        from artificial_memory.recall.retrieval_cache import SessionGate

        runtime = ArtificialMemoryRuntime(self._config())
        assert isinstance(runtime.recall_engine, DeepSeekRecallEngine)
        assert isinstance(runtime.recall_engine.gate, SessionGate)
        assert runtime.recall_engine.candidate_window == 450

    def test_tuning_fields_reach_the_gate(self):
        from artificial_memory.recall.retrieval_cache import SessionGate

        runtime = ArtificialMemoryRuntime(self._config(
            session_gate_pool=32, candidate_window=64,
        ))
        assert runtime.recall_engine.gate.pool_size == 32
        assert runtime.recall_engine.candidate_window == 64

    def test_end_to_end_session_tiering(self):
        import asyncio

        runtime = ArtificialMemoryRuntime(self._config())

        async def scenario():
            for i in range(1, 21):
                await runtime.remember(
                    f"Session A turn {i}: project atlas uses ripgrep daily.",
                    topic="Arena",
                )
            for i in range(1, 21):
                await runtime.remember(
                    f"Session B turn {i}: unrelated gardening filler {i}.",
                    topic="Arena",
                )
            r1 = await runtime.recall(
                "What toolchain does project atlas use?",
                topic="Arena", level=2,
            )
            r2 = await runtime.recall(
                "What toolchain does project atlas use?",
                topic="Arena", level=2,
            )
            return r1, r2

        r1, r2 = asyncio.run(scenario())
        assert r1.memories_retrieved == r2.memories_retrieved
        assert any(
            "ripgrep" in m.semantic_content.content for m in r1.memories
        ), "wide window + session gate must recover the evidence session"
        stats = runtime.recall_engine.plan_cache.stats_snapshot()
        assert stats[REUSE] >= 1





