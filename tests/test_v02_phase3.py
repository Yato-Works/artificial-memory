"""Tests for AM v0.2.0 Phase 3: Reconstruction Layer.

Covers (plan #15 / #16 / #17 and the Phase 3 implementation list):

* graph traversal (expandable relations, depth/node bounds, determinism),
* multi-memory evidence aggregation (A -> B -> C chains),
* contextual reconstruction (validity windows at a reconstruction timestamp),
* evidence ranking (lexical + graph + state + temporal, with penalties).
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
    ResolutionLevel,
    Topic,
)
from artificial_memory.memory.contradiction_edges import ContradictionEdgeManager
from artificial_memory.memory.evidence import (
    EvidenceRankConfig,
    EvidenceRanker,
    tokenize,
)
from artificial_memory.memory.graph import (
    DEFAULT_EXPANDABLE_TYPES,
    GraphTraversalConfig,
    MemoryGraph,
)
from artificial_memory.memory.reconstruction import (
    MemoryReconstructor,
    ReconstructionConfig,
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
def topic_id(store):
    project = store.create_project(Project(name="v02-phase3"))
    topic = store.create_topic(
        Topic(project_id=project.id, name="Test", path="Projects/v02-phase3/Test")
    )
    return topic.id


@pytest.fixture
def conversation_id(store, topic_id):
    conv = store.create_conversation(Conversation(topic_id=topic_id, title="Rust chat"))
    return conv.id


@pytest.fixture
def make_memory(store, topic_id, conversation_id):
    def _make(content: str, **kwargs) -> Memory:
        return store.create_memory(
            Memory(
                topic_id=topic_id,
                memory_type=kwargs.pop("memory_type", MemoryType.SEMANTIC),
                content=content,
                source_conversation_id=kwargs.pop("source_conversation_id", conversation_id),
                **kwargs,
            )
        )

    return _make


@pytest.fixture
def link(store):
    def _link(source: Memory, target: Memory, strength: float = 0.8, **kwargs) -> Association:
        return store.create_association(
            Association(
                source_memory_id=source.id,
                target_memory_id=target.id,
                association_type=kwargs.pop("association_type", AssociationType.RELATED),
                strength=strength,
                **kwargs,
            )
        )

    return _link


@pytest.fixture
def rust_chain(make_memory, link):
    """Plan #15's distributed-evidence scenario.

    Only the first memory lexically matches ``RUST_QUERY``; the other two are
    reachable solely through the FOLLOWS chain, which is exactly the case a
    single vector match cannot solve (plan #15).
    """
    a = make_memory("The user started learning Rust.")
    b = make_memory("Built a first CLI application in Rust.")
    c = make_memory("Continued using it for later projects.")
    link(a, b, 0.9, association_type=AssociationType.FOLLOWS)
    link(b, c, 0.9, association_type=AssociationType.FOLLOWS)
    return a, b, c


RUST_QUERY = "Which programming language has the user increasingly adopted?"

# ==================== Graph configuration ====================


class TestGraphConfig:
    def test_defaults_are_sane(self):
        config = GraphTraversalConfig()
        assert config.max_depth == 2
        assert config.include_non_current is False
        assert AssociationType.RELATED in config.expandable_types

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_depth": -1},
            {"max_nodes": 0},
            {"min_strength": 1.5},
            {"min_strength": -0.1},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            GraphTraversalConfig(**kwargs)

    def test_default_expandable_types_exclude_lifecycle_relations(self):
        assert AssociationType.CONTRADICTS not in DEFAULT_EXPANDABLE_TYPES
        assert AssociationType.SUPERSEDES not in DEFAULT_EXPANDABLE_TYPES


# ==================== Graph traversal ====================


class TestGraphTraversal:
    def test_neighbours_returns_expandable_pairs(self, store, rust_chain):
        a, b, _ = rust_chain
        pairs = MemoryGraph(store).neighbours(a.id)
        assert [memory.id for memory, _ in pairs] == [b.id]
        assert pairs[0][1].association_type == AssociationType.FOLLOWS

    def test_neighbours_exclude_lifecycle_relations(self, store, rust_chain, link):
        a, b, _ = rust_chain
        link(a, b, 0.95, association_type=AssociationType.CONTRADICTS)
        link(a, b, 0.95, association_type=AssociationType.SUPERSEDES)
        pairs = MemoryGraph(store).neighbours(a.id)
        types = {association.association_type for _, association in pairs}
        assert AssociationType.CONTRADICTS not in types
        assert AssociationType.SUPERSEDES not in types
        assert len(pairs) == 1  # only the FOLLOWS edge from the fixture

    def test_neighbours_respect_min_strength(self, store, make_memory, link):
        a = make_memory("alpha claim about storage")
        b = make_memory("beta claim about storage")
        link(a, b, 0.2)
        graph = MemoryGraph(store, GraphTraversalConfig(min_strength=0.5))
        assert graph.neighbours(a.id) == []
        assert graph.has_expandable_edges(a.id) is False

    def test_traverse_reaches_the_full_chain(self, store, rust_chain):
        a, b, c = rust_chain
        nodes = MemoryGraph(store).traverse([a.id])
        assert [node.memory_id for node in nodes] == [a.id, b.id, c.id]
        assert nodes[0].is_seed is True
        assert [node.depth for node in nodes] == [0, 1, 2]

    def test_traverse_records_path_and_strength(self, store, rust_chain):
        a, b, c = rust_chain
        nodes = MemoryGraph(store).traverse([a.id])
        node_c = next(node for node in nodes if node.memory_id == c.id)
        assert node_c.path == [a.id, b.id, c.id]
        assert node_c.path_strength == pytest.approx(0.81)  # 0.9 * 0.9
        assert node_c.graph_proximity == pytest.approx(0.81 / 3)

    def test_traverse_respects_max_depth(self, store, rust_chain):
        a, b, c = rust_chain
        graph = MemoryGraph(store, GraphTraversalConfig(max_depth=1))
        ids = [node.memory_id for node in graph.traverse([a.id])]
        assert ids == [a.id, b.id]
        assert c.id not in ids

    def test_traverse_respects_max_nodes(self, store, topic_id, conversation_id, link):
        memories = []
        previous = None
        for index in range(6):
            memory = store.create_memory(
                Memory(
                    topic_id=topic_id,
                    memory_type=MemoryType.EPISODE,
                    content=f"chain node {index}",
                    source_conversation_id=conversation_id,
                )
            )
            if previous is not None:
                link(previous, memory, 0.9, association_type=AssociationType.FOLLOWS)
            memories.append(memory)
            previous = memory
        graph = MemoryGraph(store, GraphTraversalConfig(max_depth=5, max_nodes=3))
        assert len(graph.traverse([memories[0].id])) == 3

    def test_traverse_is_deterministic(self, store, make_memory, link):
        a = make_memory("Seed memory about caching.")
        b = make_memory("Neighbour one about caching.")
        c = make_memory("Neighbour two about caching.")
        # Equal strengths: ordering must fall back to association id.
        link(a, b, 0.6)
        link(a, c, 0.6)
        graph = MemoryGraph(store)
        first = [node.to_dict() for node in graph.traverse([a.id])]
        second = [node.to_dict() for node in graph.traverse([a.id])]
        assert first == second
        assert [node.memory_id for node in graph.traverse([a.id])] == [a.id, b.id, c.id]

    def test_cycle_terminates_without_duplicates(self, store, make_memory, link):
        a = make_memory("Cycle A about Rust.")
        b = make_memory("Cycle B about Rust.")
        link(a, b, 0.9)
        link(b, a, 0.9)
        nodes = MemoryGraph(store, GraphTraversalConfig(max_depth=5)).traverse([a.id])
        assert [node.memory_id for node in nodes] == [a.id, b.id]

    def test_traversal_orders_neighbours_by_strength(self, store, make_memory, link):
        root = make_memory("Root memory about Rust.")
        weak = make_memory("Weakly related Rust memory.")
        strong = make_memory("Strongly related Rust memory.")
        link(root, weak, 0.4)
        link(root, strong, 0.95)
        nodes = MemoryGraph(store, GraphTraversalConfig(max_depth=1)).traverse([root.id])
        assert [node.memory_id for node in nodes] == [root.id, strong.id, weak.id]

    def test_non_current_memories_are_excluded_by_default(self, store, make_memory, link):
        root = make_memory("Current deployment note.")
        archived = make_memory("Archived Rust memory.")
        link(root, archived, 0.9)
        store.update_memory(archived.model_copy(update={"is_current": False}))

        default_nodes = MemoryGraph(store).traverse([root.id])
        assert {node.memory_id for node in default_nodes} == {root.id}

        inclusive = MemoryGraph(store, GraphTraversalConfig(include_non_current=True)).traverse(
            [root.id]
        )
        assert {node.memory_id for node in inclusive} == {root.id, archived.id}

    def test_traverse_ignores_unknown_seed(self, store):
        assert MemoryGraph(store).traverse([9999]) == []

    def test_graph_proximity_decreases_with_depth(self, store, rust_chain):
        a, b, c = rust_chain
        by_id = {node.memory_id: node for node in MemoryGraph(store).traverse([a.id])}
        assert by_id[a.id].graph_proximity == 1.0
        assert by_id[a.id].graph_proximity > by_id[b.id].graph_proximity
        assert by_id[b.id].graph_proximity > by_id[c.id].graph_proximity

    def test_find_path_returns_shortest_chain(self, store, rust_chain):
        a, b, c = rust_chain
        path = MemoryGraph(store, GraphTraversalConfig(max_depth=3)).find_path(a.id, c.id)
        assert len(path) == 2
        assert path[0].source_memory_id == a.id
        assert path[0].target_memory_id == b.id
        assert path[1].target_memory_id == c.id

    def test_find_path_returns_empty_when_unreachable(self, store, rust_chain, make_memory):
        a, _b, _c = rust_chain
        isolated = make_memory("Disconnected memory.")
        graph = MemoryGraph(store)
        assert graph.find_path(a.id, isolated.id) == []
        assert graph.find_path(a.id, a.id) == []

    def test_graph_node_to_dict_is_serializable(self, store, rust_chain):
        a, _b, _c = rust_chain
        payload = MemoryGraph(store).traverse([a.id])[0].to_dict()
        assert payload["memory_id"] == a.id
        assert payload["is_seed"] is True
        assert payload["path"] == [a.id]

# ==================== Evidence ranking ====================


class TestEvidenceRanking:
    def test_tokenize_handles_latin_and_cjk(self):
        assert {"rust", "learning"} <= tokenize("Learning Rust.")
        japanese = tokenize("記憶の再構築")
        assert "記憶" in japanese
        assert "再構" in japanese

    def test_tokenize_is_case_insensitive(self):
        assert tokenize("Rust") == tokenize("rust")

    @pytest.mark.parametrize(
        "kwargs",
        [{"lexical_weight": 1.5}, {"max_items": 0}, {"contradiction_penalty": -0.5}],
    )
    def test_config_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            EvidenceRankConfig(**kwargs)

    def test_lexical_score_is_query_coverage(self, store, make_memory):
        memory = make_memory("The user started learning Rust.")
        nodes = MemoryGraph(store).traverse([memory.id])
        item = EvidenceRanker(store).rank("Which language did the user learn?", nodes)[0]
        assert item.lexical > 0.0
        assert "user" in item.shared_terms
        assert item.depth == 0
        assert item.is_distributed is False

    def test_unrelated_seed_is_dropped(self, store, make_memory):
        make_memory("The user prefers dark mode.")
        nodes = MemoryGraph(store).traverse(
            [store.get_memories(limit=1)[0].id]
        )
        assert EvidenceRanker(store).rank("What database engine is used?", nodes) == []

    def test_distributed_evidence_is_kept_without_lexical_match(self, store, make_memory, link):
        seed = make_memory("The user started learning Rust.")
        follow_up = make_memory("That project now serves 12 production nodes.")
        link(seed, follow_up, 0.9, association_type=AssociationType.FOLLOWS)
        nodes = MemoryGraph(store).traverse([seed.id])
        items = {item.memory_id: item for item in EvidenceRanker(store).rank("learning Rust", nodes)}
        assert follow_up.id in items
        assert items[follow_up.id].is_distributed is True
        assert items[follow_up.id].graph > 0.0

    def test_ranks_lexical_match_first(self, store, make_memory):
        make_memory("The user owns a cat named Mugi.")
        rust = make_memory("The user increasingly adopted Rust for CLI projects.")
        items = EvidenceRanker(store).rank_memories(RUST_QUERY, store.get_memories(limit=10))
        assert items[0].memory_id == rust.id
        assert items[0].lexical > 0.0

    def test_contradiction_risk_lowers_the_score(self, store, make_memory):
        memory = make_memory("The user deploys on Friday.")
        rival = make_memory("The user never deploys on Friday.")
        ranker = EvidenceRanker(store)
        before = ranker.rank("deploys Friday", MemoryGraph(store).traverse([memory.id]))[0]

        ContradictionEdgeManager(store).create_edge(memory.id, rival.id, severity=0.9)
        after = ranker.rank("deploys Friday", MemoryGraph(store).traverse([memory.id]))[0]

        assert after.penalty > 0.0
        assert after.score < before.score
        assert after.state_vector.contradiction_risk > 0.0

    def test_non_current_memory_is_penalised_and_filtered(self, store, make_memory):
        active = make_memory("The user adopted Rust for CLI projects.")
        retired = make_memory("The user adopted Rust for CLI projects previously.")
        store.update_memory(retired.model_copy(update={"is_current": False}))

        ranker = EvidenceRanker(store)
        items = ranker.rank_memories(RUST_QUERY, store.get_memories(limit=10))
        assert retired.id not in {item.memory_id for item in items}

        inclusive = EvidenceRanker(store, config=EvidenceRankConfig(include_non_current=True))
        by_id = {item.memory_id: item for item in inclusive.rank_memories(
            RUST_QUERY, store.get_memories(limit=10)
        )}
        assert retired.id in by_id
        assert by_id[retired.id].penalty >= inclusive.config.non_current_penalty
        assert by_id[active.id].score > by_id[retired.id].score

    def test_reasons_explain_the_inclusion(self, store, rust_chain):
        a, _b, _c = rust_chain
        nodes = MemoryGraph(store).traverse([a.id])
        item = EvidenceRanker(store).rank(RUST_QUERY, nodes)[0]
        assert item.reasons
        assert any("direct candidate" in reason for reason in item.reasons)

    def test_rank_memories_wraps_plain_lists(self, store, make_memory):
        memory = make_memory("The user uses PostgreSQL for storage.")
        items = EvidenceRanker(store).rank_memories(
            "Which database does the user use?", [memory]
        )
        assert [item.memory_id for item in items] == [memory.id]

    def test_max_items_truncates_the_package(self, store, make_memory):
        memories = [make_memory(f"The user uses Rust in project {i}.") for i in range(6)]
        ranker = EvidenceRanker(store, config=EvidenceRankConfig(max_items=2))
        assert len(ranker.rank_memories("Rust project", memories)) == 2

    def test_evidence_item_to_dict_is_serializable(self, store, rust_chain):
        a, _b, _c = rust_chain
        nodes = MemoryGraph(store).traverse([a.id])
        payload = EvidenceRanker(store).rank(RUST_QUERY, nodes)[0].to_dict()
        assert payload["memory_id"] == a.id
        assert payload["resolution"] == ResolutionLevel.RAW.name
        assert isinstance(payload["path"], list)

    def test_ranking_is_deterministic(self, store, rust_chain):
        a, _, _ = rust_chain
        nodes = MemoryGraph(store).traverse([a.id])
        ranker = EvidenceRanker(store)
        when = datetime.now()
        first = [item.to_dict() for item in ranker.rank(RUST_QUERY, nodes, now=when)]
        second = [item.to_dict() for item in ranker.rank(RUST_QUERY, nodes, now=when)]
        assert first == second

# ==================== Reconstruction pipeline ====================


class TestReconstruction:
    def test_reconstructs_distributed_chain(self, store, rust_chain):
        a, b, c = rust_chain
        package = MemoryReconstructor(store).reconstruct(RUST_QUERY, timestamp=datetime.now())

        assert {a.id, b.id, c.id} <= set(package.memory_ids)
        assert package.has_distributed_evidence is True
        assert any(chain == [a.id, b.id, c.id] for chain in package.chains)

    def test_synthesis_cites_memory_ids_and_scores(self, store, rust_chain):
        package = MemoryReconstructor(store).reconstruct(RUST_QUERY)
        assert package.synthesis
        for item in package.evidence:
            assert f"memory={item.memory_id}" in package.synthesis
        assert "score=" in package.synthesis

    def test_reconstruction_is_deterministic(self, store, rust_chain):
        reconstructor = MemoryReconstructor(store)
        when = datetime.now()
        first = reconstructor.reconstruct(RUST_QUERY, timestamp=when)
        second = reconstructor.reconstruct(RUST_QUERY, timestamp=when)
        assert first.memory_ids == second.memory_ids
        assert first.synthesis == second.synthesis
        assert first.stats == second.stats

    def test_explicit_seed_ids_are_pinned(self, store, rust_chain):
        a, b, c = rust_chain
        package = MemoryReconstructor(store).reconstruct(RUST_QUERY, seed_ids=[b.id])
        assert package.seeds == [b.id]
        assert {a.id, b.id, c.id} <= set(package.memory_ids)

    def test_multi_hop_chain_is_recoverable(self, store, rust_chain):
        a, b, c = rust_chain
        package = MemoryReconstructor(store).reconstruct_multi_hop(
            RUST_QUERY, [a.id, b.id, c.id]
        )
        assert {a.id, b.id, c.id} <= set(package.memory_ids)

    def test_unknown_query_yields_empty_package(self, store, make_memory):
        make_memory("The user owns a cat named Mugi.")
        package = MemoryReconstructor(store).reconstruct("quantum chromodynamics lattice")
        assert package.evidence == []
        assert package.synthesis == ""
        assert package.stats["evidence_count"] == 0

    def test_missing_seed_yields_empty_package(self, store, rust_chain):
        package = MemoryReconstructor(store).reconstruct(RUST_QUERY, seed_ids=[9999])
        assert package.seeds == []
        assert package.evidence == []

    def test_stats_report_pipeline_shape(self, store, rust_chain):
        package = MemoryReconstructor(store).reconstruct(RUST_QUERY)
        assert package.stats["seed_count"] >= 1
        assert package.stats["candidate_count"] >= package.stats["evidence_count"]
        assert package.stats["distributed"] is True

    def test_temporal_filter_excludes_out_of_window_memories(self, store, make_memory):
        now = datetime.now()
        make_memory(
            "The user deploys on Fridays.",
            valid_from=now - timedelta(days=10),
            valid_until=now - timedelta(days=5),
        )
        reconstructor = MemoryReconstructor(store)

        gone = reconstructor.reconstruct("deploys", timestamp=now)
        assert gone.evidence == []

        historical = reconstructor.reconstruct("deploys", timestamp=now - timedelta(days=7))
        assert historical.evidence
        assert "deploys" in historical.synthesis

    def test_unresolved_conflicts_are_surfaced(self, store, make_memory):
        claim = make_memory("The user deploys on Fridays.")
        rival = make_memory("The user never deploys on Fridays.")
        ContradictionEdgeManager(store).create_edge(claim.id, rival.id, severity=0.9)

        package = MemoryReconstructor(store).reconstruct("deploys", timestamp=datetime.now())
        assert len(package.unresolved_conflicts) == 1
        assert package.unresolved_conflicts[0].other_in_evidence is True
        assert "CONFLICT" in package.synthesis

    def test_resolved_conflicts_are_reported_as_resolved(self, store, make_memory):
        now = datetime.now()
        claim = make_memory("The user deploys on Fridays.")
        rival = make_memory(
            "The user never deploys on Fridays.",
            valid_from=now - timedelta(days=10),
            valid_until=now - timedelta(days=1),
        )
        ContradictionEdgeManager(store).create_edge(claim.id, rival.id, severity=0.9)

        package = MemoryReconstructor(store).reconstruct("deploys", timestamp=now)
        assert package.conflicts
        assert package.unresolved_conflicts == []
        assert "CONFLICT" not in package.synthesis

    def test_provenance_expansion_adds_evidence_lineage(self, store, make_memory, link):
        parent = make_memory("Raw note from the chat log.")
        child = make_memory("The deployment schedule is weekly.")
        link(parent, child, 0.8, association_type=AssociationType.ELABORATES)

        package = MemoryReconstructor(store).reconstruct("deployment schedule")
        assert package.provenance
        assert any(
            trace.memory_id == parent.id and trace.relationship == "elaborates"
            for trace in package.provenance
        )
        assert parent.id in package.memory_ids

    def test_provenance_expansion_can_be_disabled(self, store, make_memory, link):
        parent = make_memory("Raw note from the chat log.")
        child = make_memory("The deployment schedule is weekly.")
        link(parent, child, 0.8, association_type=AssociationType.ELABORATES)

        config = ReconstructionConfig(expand_provenance=False)
        package = MemoryReconstructor(store, config).reconstruct("deployment schedule")
        assert package.provenance == []

    def test_package_to_text_renders_the_pipeline(self, store, rust_chain):
        package = MemoryReconstructor(store).reconstruct(RUST_QUERY)
        text = package.to_text()
        assert "Reconstructed Evidence Package" in text
        assert "Distributed chains:" in text

    def test_package_to_dict_is_serializable(self, store, rust_chain):
        package = MemoryReconstructor(store).reconstruct(RUST_QUERY)
        payload = package.to_dict()
        assert payload["seeds"] == package.seeds
        assert payload["stats"]["evidence_count"] == len(package.evidence)






