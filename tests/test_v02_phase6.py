"""Tests for AM v0.2.0 Phase 6: Predictive Recall.

Covers (plan #19 and the Phase 6 implementation list):

* topic prediction (association / recency / keyword signals, normalization),
* prefetch (bounded, cache-only, idempotent),
* cache (TTL, deterministic eviction, stats),
* deferred context injection (memories served only on a real query; the
  prefetch itself never touches the context path).
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from artificial_memory.core.models import (
    Association,
    AssociationType,
    Conversation,
    Memory,
    MemoryType,
    Project,
    RecallEvent,
    RecallLevel,
    Topic,
)
from artificial_memory.recall.engine import BasicRecallEngine
from artificial_memory.recall.predictive import (
    PredictiveRecallConfig,
    PredictiveRecallEngine,
    PrefetchCache,
    PrefetchEntry,
    TopicPredictor,
)
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    import gc
    import time

    gc.collect()
    time.sleep(0.1)
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        pass


@pytest.fixture
def store(temp_db):
    s = SQLiteMemoryStore(temp_db)
    yield s
    s.close()


@pytest.fixture
def setup(store):
    """Two topics: current (Rust chat) and a linked one (Deployment)."""
    project = store.create_project(Project(name="v02-phase6"))
    current = store.create_topic(
        Topic(project_id=project.id, name="Rust", path="Projects/v02-phase6/Rust")
    )
    deployment = store.create_topic(
        Topic(project_id=project.id, name="Deployment", path="Projects/v02-phase6/Deployment")
    )
    unlinked = store.create_topic(
        Topic(project_id=project.id, name="Cooking", path="Projects/v02-phase6/Cooking")
    )
    conv = store.create_conversation(Conversation(topic_id=current.id, title="Rust chat"))
    deployment_conv = store.create_conversation(
        Conversation(topic_id=deployment.id, title="Deploy chat")
    )

    def make_memory(topic_id: int, content: str, **kwargs) -> Memory:
        return store.create_memory(
            Memory(
                topic_id=topic_id,
                memory_type=MemoryType.SEMANTIC,
                content=content,
                source_conversation_id=kwargs.pop("source_conversation_id", conv.id),
                **kwargs,
            )
        )

    return {
        "store": store,
        "current": current,
        "deployment": deployment,
        "unlinked": unlinked,
        "make_memory": make_memory,
        "conversation_id": conv.id,
        "deployment_conversation_id": deployment_conv.id,
    }


NOW = datetime.now()

# ==================== Topic prediction ====================


class TestTopicPrediction:
    def test_association_signal_ranks_linked_topic(self, setup):
        store = setup["store"]
        rust_memory = setup["make_memory"](
            setup["current"].id, "The user tunes the Rust build pipeline."
        )
        deploy_memory = setup["make_memory"](
            setup["deployment"].id,
            "The build pipeline is deployed on Fridays.",
            source_conversation_id=setup["deployment_conversation_id"],
        )
        store.create_association(
            Association(
                source_memory_id=rust_memory.id,
                target_memory_id=deploy_memory.id,
                association_type=AssociationType.RELATED,
                strength=0.9,
            )
        )

        predictions = TopicPredictor(store).predict(setup["current"].id, now=NOW)
        ids = [prediction.topic_id for prediction in predictions]
        assert setup["deployment"].id in ids
        assert setup["unlinked"].id not in ids

        top = predictions[0]
        assert top.topic_id == setup["deployment"].id
        assert top.association_score > 0.0
        assert 0.0 < top.probability <= 1.0

    def test_recency_signal_from_recall_events(self, setup):
        store = setup["store"]
        store.log_recall(
            RecallEvent(
                query="deployment checklist",
                topic_id=setup["deployment"].id,
                recall_level=RecallLevel.CURRENT_ONLY,
            )
        )
        predictions = TopicPredictor(store).predict(setup["current"].id, now=NOW)
        by_topic = {p.topic_id: p for p in predictions}
        assert setup["deployment"].id in by_topic
        assert by_topic[setup["deployment"].id].recency_score > 0.0

    def test_keyword_signal_matches_query(self, setup):
        store = setup["store"]
        setup["make_memory"](
            setup["deployment"].id,
            "Deployment checklist for production releases.",
            source_conversation_id=setup["deployment_conversation_id"],
        )
        predictor = TopicPredictor(store)
        predictions = predictor.predict(
            setup["current"].id, query="deployment checklist", now=NOW
        )
        top = predictions[0]
        assert top.topic_id == setup["deployment"].id
        assert top.keyword_score > 0.0
        assert any("shared terms" in item for item in top.evidence)

    def test_probabilities_are_normalized_and_bounded(self, setup):
        store = setup["store"]
        rust_memory = setup["make_memory"](setup["current"].id, "Rust build tuning.")
        deploy_memory = setup["make_memory"](
            setup["deployment"].id,
            "Deployment pipeline.",
            source_conversation_id=setup["deployment_conversation_id"],
        )
        store.create_association(
            Association(
                source_memory_id=rust_memory.id,
                target_memory_id=deploy_memory.id,
                association_type=AssociationType.RELATED,
                strength=0.9,
            )
        )
        predictions = TopicPredictor(store).predict(setup["current"].id, now=NOW)
        assert predictions
        total = sum(p.probability for p in predictions)
        assert total == pytest.approx(1.0)

    def test_current_topic_is_never_predicted(self, setup):
        setup["make_memory"](setup["current"].id, "Rust content.")
        predictions = TopicPredictor(store=setup["store"]).predict(
            setup["current"].id, now=NOW
        )
        assert all(p.topic_id != setup["current"].id for p in predictions)

    def test_empty_store_yields_no_predictions(self, setup):
        predictions = TopicPredictor(store=setup["store"]).predict(
            setup["current"].id, now=NOW
        )
        assert predictions == []

    def test_predictions_are_deterministic(self, setup):
        store = setup["store"]
        rust_memory = setup["make_memory"](setup["current"].id, "Rust build tuning.")
        deploy_memory = setup["make_memory"](
            setup["deployment"].id,
            "Deployment pipeline.",
            source_conversation_id=setup["deployment_conversation_id"],
        )
        store.create_association(
            Association(
                source_memory_id=rust_memory.id,
                target_memory_id=deploy_memory.id,
                association_type=AssociationType.RELATED,
                strength=0.9,
            )
        )
        predictor = TopicPredictor(store)
        first = [p.to_dict() for p in predictor.predict(setup["current"].id, now=NOW)]
        second = [p.to_dict() for p in predictor.predict(setup["current"].id, now=NOW)]
        assert first == second

    def test_config_rejects_invalid_values(self):
        with pytest.raises(ValueError):
            PredictiveRecallConfig(association_weight=1.5)
        with pytest.raises(ValueError):
            PredictiveRecallConfig(max_memories_per_topic=0)
        with pytest.raises(ValueError):
            PredictiveRecallConfig(cache_ttl_seconds=0)

# ==================== Prefetch and cache ====================


class TestPrefetchAndCache:
    def test_prefetch_fills_cache_without_context_injection(self, setup):
        store = setup["store"]
        rust_memory = setup["make_memory"](setup["current"].id, "Rust build tuning.")
        deploy_memory = setup["make_memory"](
            setup["deployment"].id,
            "Deployment pipeline.",
            source_conversation_id=setup["deployment_conversation_id"],
        )
        store.create_association(
            Association(
                source_memory_id=rust_memory.id,
                target_memory_id=deploy_memory.id,
                association_type=AssociationType.RELATED,
                strength=0.9,
            )
        )
        engine = PredictiveRecallEngine(
            store, recall_engine=BasicRecallEngine(store)
        )
        result = engine.prefetch(setup["current"].id, now=NOW)

        assert setup["deployment"].id in result.fetched
        fetched_ids = result.fetched[setup["deployment"].id]
        assert deploy_memory.id in fetched_ids
        assert engine.cache.stats()["entries"] == 1

    def test_prefetch_is_idempotent_via_cache(self, setup):
        store = setup["store"]
        rust_memory = setup["make_memory"](setup["current"].id, "Rust build tuning.")
        deploy_memory = setup["make_memory"](
            setup["deployment"].id,
            "Deployment pipeline.",
            source_conversation_id=setup["deployment_conversation_id"],
        )
        store.create_association(
            Association(
                source_memory_id=rust_memory.id,
                target_memory_id=deploy_memory.id,
                association_type=AssociationType.RELATED,
                strength=0.9,
            )
        )
        engine = PredictiveRecallEngine(store, recall_engine=BasicRecallEngine(store))
        first = engine.prefetch(setup["current"].id, now=NOW)
        second = engine.prefetch(setup["current"].id, now=NOW)

        assert first.fetched
        assert not second.fetched  # reused from cache instead
        assert second.reused[setup["deployment"].id] == first.fetched[setup["deployment"].id]

    def test_cache_ttl_expires_entries(self, setup):
        cache = PrefetchCache(max_entries=4, ttl_seconds=60)
        entry = PrefetchEntry(
            topic_id=1,
            memories=[],
            fetched_at=NOW - timedelta(seconds=120),
        )
        cache.put(entry)
        assert cache.get(1, NOW) is None
        assert cache.stats()["evictions"] == 1

    def test_cache_evicts_oldest_deterministically(self, setup):
        cache = PrefetchCache(max_entries=2, ttl_seconds=3600)
        for topic_id in (1, 2, 3):
            cache.put(
                PrefetchEntry(
                    topic_id=topic_id,
                    memories=[],
                    fetched_at=NOW + timedelta(seconds=topic_id),
                )
            )
        assert cache.get(1, NOW) is None  # evicted first (oldest)
        assert cache.get(2, NOW) is not None
        assert cache.get(3, NOW) is not None

    def test_prefetch_honors_max_topics(self, setup):
        store = setup["store"]
        project = store.list_projects()[0]
        rust_memory = setup["make_memory"](setup["current"].id, "Rust tuning.")
        # Two additional linked topics; only prefetch_topics should fill.
        extra_topics = []
        for index in range(2):
            topic = store.create_topic(
                Topic(project_id=project.id, name=f"Extra {index}", path=f"p/extra{index}")
            )
            extra = store.create_memory(
                Memory(
                    topic_id=topic.id,
                    memory_type=MemoryType.SEMANTIC,
                    content=f"Extra memory {index}.",
                    source_conversation_id=setup["conversation_id"],
                )
            )
            store.create_association(
                Association(
                    source_memory_id=rust_memory.id,
                    target_memory_id=extra.id,
                    association_type=AssociationType.RELATED,
                    strength=0.8,
                )
            )
            extra_topics.append(topic)
        config = PredictiveRecallConfig(prefetch_topics=1)
        engine = PredictiveRecallEngine(store, recall_engine=BasicRecallEngine(store), config=config)
        result = engine.prefetch(setup["current"].id, now=NOW)
        assert len(result.fetched) <= 1

    def test_stats_track_hits_and_misses(self, setup):
        store = setup["store"]
        engine = PredictiveRecallEngine(store, recall_engine=BasicRecallEngine(store))
        engine.cache.put(PrefetchEntry(topic_id=42, memories=[], fetched_at=NOW))
        assert engine.cache.get(42, NOW) is not None
        assert engine.cache.get(43, NOW) is None
        stats = engine.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

# ==================== Deferred context injection ====================


class TestDeferredInjection:
    def test_prefetch_does_not_call_recall_engine(self, setup):
        store = setup["store"]
        rust_memory = setup["make_memory"](setup["current"].id, "Rust build tuning.")
        deploy_memory = setup["make_memory"](
            setup["deployment"].id,
            "Deployment pipeline.",
            source_conversation_id=setup["deployment_conversation_id"],
        )
        store.create_association(
            Association(
                source_memory_id=rust_memory.id,
                target_memory_id=deploy_memory.id,
                association_type=AssociationType.RELATED,
                strength=0.9,
            )
        )

        calls: list[str] = []

        class SpyRecallEngine:
            def recall(self, query, topic_id=None, level=RecallLevel.CURRENT_ONLY, max_tokens=4000):
                calls.append(query)
                return [], 0

        engine = PredictiveRecallEngine(store, recall_engine=SpyRecallEngine())
        engine.prefetch(setup["current"].id, now=NOW)
        assert calls == []  # prefetch stayed out of the response path

    def test_recall_serves_from_cache_on_hit(self, setup):
        store = setup["store"]
        rust_memory = setup["make_memory"](setup["current"].id, "Rust build tuning.")
        deploy_memory = setup["make_memory"](
            setup["deployment"].id,
            "Deployment pipeline.",
            source_conversation_id=setup["deployment_conversation_id"],
        )
        store.create_association(
            Association(
                source_memory_id=rust_memory.id,
                target_memory_id=deploy_memory.id,
                association_type=AssociationType.RELATED,
                strength=0.9,
            )
        )
        engine = PredictiveRecallEngine(store, recall_engine=BasicRecallEngine(store))
        engine.prefetch(setup["current"].id, query="deployment", now=NOW)

        memories, source = engine.recall(
            "deployment pipeline", current_topic_id=setup["current"].id, now=NOW
        )
        assert source == "cache"
        assert deploy_memory.id in [memory.id for memory in memories]

    def test_recall_falls_back_to_engine_on_miss(self, setup):
        store = setup["store"]
        memory = setup["make_memory"](setup["current"].id, "Rust build tuning.")

        class StubRecallEngine:
            def recall(self, query, topic_id=None, level=RecallLevel.CURRENT_ONLY, max_tokens=4000):
                return [memory], 5

        engine = PredictiveRecallEngine(store, recall_engine=StubRecallEngine())
        memories, source = engine.recall(
            "anything", current_topic_id=setup["current"].id, now=NOW
        )
        assert source == "recall_engine"
        assert [m.id for m in memories] == [memory.id]

    def test_recall_without_topic_goes_to_engine(self, setup):
        store = setup["store"]
        memory = setup["make_memory"](setup["current"].id, "Rust build tuning.")

        class StubRecallEngine:
            def recall(self, query, topic_id=None, level=RecallLevel.CURRENT_ONLY, max_tokens=4000):
                return [memory], 5

        engine = PredictiveRecallEngine(store, recall_engine=StubRecallEngine())
        _memories, source = engine.recall("anything", current_topic_id=None, now=NOW)
        assert source == "recall_engine"

    def test_prefetch_result_is_serializable(self, setup):
        store = setup["store"]
        rust_memory = setup["make_memory"](setup["current"].id, "Rust build tuning.")
        deploy_memory = setup["make_memory"](
            setup["deployment"].id,
            "Deployment pipeline.",
            source_conversation_id=setup["deployment_conversation_id"],
        )
        store.create_association(
            Association(
                source_memory_id=rust_memory.id,
                target_memory_id=deploy_memory.id,
                association_type=AssociationType.RELATED,
                strength=0.9,
            )
        )
        engine = PredictiveRecallEngine(store, recall_engine=BasicRecallEngine(store))
        payload = engine.prefetch(setup["current"].id, now=NOW).to_dict()
        assert payload["memory_count"] >= 1
        assert payload["predictions"]



