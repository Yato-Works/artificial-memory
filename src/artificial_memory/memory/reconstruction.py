"""Memory reconstruction layer (AM v0.2.0 Phase 3).

Plan #15 defines the AM reconstruction pipeline::

    query
      -> initial candidates
      -> semantic relations
      -> temporal relations
      -> provenance expansion
      -> contradiction checks
      -> multi-memory synthesis
      -> reconstructed evidence package

Unlike classic retrieval (``query -> top-k``) this answers questions whose
relevant information is *distributed across several memories*, for example::

    Memory A: user started learning Rust.
    Memory B: user built a Rust CLI application.
    Memory C: user has continued using Rust for later projects.

A single vector match returns one of these; reconstruction walks
``A -> B -> C`` and returns the chain as an evidence package.

Synthesis here is **deterministic text composition**, not generation: Phase 3
produces an auditable evidence package (ordered evidence, chains, conflicts,
provenance, stats). An answering model consumes that package; any LLM-written
summary belongs to a later phase and must not silently replace the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory
from artificial_memory.memory.contradiction_edges import ContradictionEdgeManager
from artificial_memory.memory.evidence import (
    EvidenceItem,
    EvidenceRankConfig,
    EvidenceRanker,
)
from artificial_memory.memory.graph import GraphNode, GraphTraversalConfig, MemoryGraph
from artificial_memory.memory.provenance import ProvenanceChainBuilder
from artificial_memory.memory.state_signals import StateSignalCalculator
from artificial_memory.memory.supersession import SupersessionManager
from artificial_memory.memory.temporal_validity import TemporalValidityManager


@dataclass
class ReconstructionConfig:
    """Bounds for one reconstruction pass."""

    # Seed selection ("initial candidates").
    max_seeds: int = 5
    scan_limit: int = 200
    # Temporal relations: memories outside their validity window at the
    # reconstruction timestamp are excluded (recovered later by RESTORE).
    include_out_of_window: bool = False
    # Graph expansion.
    graph: GraphTraversalConfig = field(default_factory=GraphTraversalConfig)
    # Provenance expansion adds the evidence lineage of the top items.
    expand_provenance: bool = True
    provenance_depth: int = 3
    provenance_for_top: int = 3
    # Contradiction checks.
    check_contradictions: bool = True
    max_conflicts: int = 10
    # Evidence ranking.
    ranking: EvidenceRankConfig = field(default_factory=EvidenceRankConfig)
    # Multi-memory synthesis.
    synthesis_max_chars: int = 600

    def __post_init__(self) -> None:
        if self.max_seeds <= 0:
            raise ValueError("max_seeds must be positive")
        if self.scan_limit <= 0:
            raise ValueError("scan_limit must be positive")
        if self.provenance_depth <= 0:
            raise ValueError("provenance_depth must be positive")
        if self.provenance_for_top < 0:
            raise ValueError("provenance_for_top must be non-negative")
        if self.max_conflicts < 0:
            raise ValueError("max_conflicts must be non-negative")
        if self.synthesis_max_chars <= 0:
            raise ValueError("synthesis_max_chars must be positive")


@dataclass
class ConflictContext:
    """An unresolved or resolved contradiction touching the evidence."""

    memory_id: int
    other_memory_id: int
    severity: float
    resolved: bool
    other_in_evidence: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "other_memory_id": self.other_memory_id,
            "severity": self.severity,
            "resolved": self.resolved,
            "other_in_evidence": self.other_in_evidence,
            "detail": self.detail,
        }


@dataclass
class ProvenanceTrace:
    """One node of an evidence item's provenance lineage."""

    memory_id: int
    root_memory_id: int
    relationship: str
    depth: int
    resolution: str
    content_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "root_memory_id": self.root_memory_id,
            "relationship": self.relationship,
            "depth": self.depth,
            "resolution": self.resolution,
            "content_preview": self.content_preview,
        }


@dataclass
class ReconstructedEvidencePackage:
    """The output of reconstruction: a small, auditable evidence package."""

    query: str
    timestamp: datetime
    seeds: list[int] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    chains: list[list[int]] = field(default_factory=list)
    conflicts: list[ConflictContext] = field(default_factory=list)
    provenance: list[ProvenanceTrace] = field(default_factory=list)
    synthesis: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def memory_ids(self) -> list[int]:
        return [item.memory_id for item in self.evidence]

    @property
    def has_distributed_evidence(self) -> bool:
        """True when the answer depends on more than a single-hit match."""
        return any(item.is_distributed for item in self.evidence)

    @property
    def unresolved_conflicts(self) -> list[ConflictContext]:
        return [conflict for conflict in self.conflicts if not conflict.resolved]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "seeds": list(self.seeds),
            "evidence": [item.to_dict() for item in self.evidence],
            "chains": [list(chain) for chain in self.chains],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "provenance": [trace.to_dict() for trace in self.provenance],
            "synthesis": self.synthesis,
            "stats": dict(self.stats),
        }

    def to_text(self) -> str:
        """Human-readable rendering (inspection / debugging)."""
        lines = [
            "Reconstructed Evidence Package",
            f"Query: {self.query}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Seeds: {', '.join(str(s) for s in self.seeds) or 'none'}",
            f"Evidence: {len(self.evidence)} item(s)",
            "",
            self.synthesis or "(no evidence)",
        ]
        if self.chains:
            lines.append("")
            lines.append("Distributed chains:")
            lines.extend("  " + " -> ".join(str(mid) for mid in chain) for chain in self.chains)
        if self.conflicts:
            lines.append("")
            lines.append("Conflicts:")
            lines.extend(
                f"  {c.memory_id} vs {c.other_memory_id} "
                f"(severity={c.severity:.2f}, {'resolved' if c.resolved else 'unresolved'})"
                for c in self.conflicts
            )
        return "\n".join(lines)


class MemoryReconstructor:
    """Reconstructs a multi-memory evidence package for a query."""

    def __init__(
        self,
        store: MemoryStore,
        config: ReconstructionConfig | None = None,
        calculator: StateSignalCalculator | None = None,
    ):
        self.store = store
        self.config = config or ReconstructionConfig()
        self.calculator = calculator or StateSignalCalculator(store)
        self.temporal = TemporalValidityManager(store)
        self.supersession = SupersessionManager(store)
        self.contradictions = ContradictionEdgeManager(store)
        self.provenance = ProvenanceChainBuilder(store)

    # ==================== Public API ====================

    def reconstruct(
        self,
        query: str,
        topic_id: int | None = None,
        timestamp: datetime | None = None,
        seed_ids: list[int] | None = None,
        config: ReconstructionConfig | None = None,
    ) -> ReconstructedEvidencePackage:
        """Run the full reconstruction pipeline for ``query``.

        ``timestamp`` is the point in time the reconstruction is performed for;
        temporal relations (validity windows, unresolved contradictions) are
        evaluated against it, which is what makes historical reconstruction
        possible. ``seed_ids`` bypasses candidate selection when the caller
        already knows which memories to start from; those seeds are pinned into
        the package even without a lexical match.
        """
        timestamp = timestamp or datetime.now()
        effective = config or self.config

        seeds = self._select_seeds(query, topic_id, seed_ids, effective)
        if not seeds:
            return ReconstructedEvidencePackage(
                query=query,
                timestamp=timestamp,
                stats={"seed_count": 0, "evidence_count": 0, "distributed": False},
            )

        graph = MemoryGraph(self.store, effective.graph)
        nodes = graph.traverse([seed.id for seed in seeds if seed.id is not None], effective.graph)

        # Provenance expansion: add the evidence lineage of the seed memories
        # as additional candidates before ranking.
        provenance_traces: list[ProvenanceTrace] = []
        if effective.expand_provenance:
            nodes, provenance_traces = self._expand_provenance(nodes, effective)

        # Temporal relations: drop candidates whose validity window does not
        # contain the reconstruction timestamp.
        nodes = self._apply_temporal_filter(nodes, timestamp, effective)

        ranker = EvidenceRanker(self.store, self.calculator, effective.ranking)
        # Explicitly supplied seeds are pinned: the caller asked for them, so
        # they stay in the package even without a lexical match.
        pinned_ids = set(seed_ids) if seed_ids else None
        evidence = ranker.rank(query, nodes, now=timestamp, keep_ids=pinned_ids)

        conflicts: list[ConflictContext] = []
        if effective.check_contradictions:
            conflicts = self._check_contradictions(evidence, timestamp, effective)

        chains = self._collect_chains(evidence)
        synthesis = self._synthesize(evidence, conflicts, effective)

        return ReconstructedEvidencePackage(
            query=query,
            timestamp=timestamp,
            seeds=[seed.id for seed in seeds if seed.id is not None],
            evidence=evidence,
            chains=chains,
            conflicts=conflicts,
            provenance=provenance_traces,
            synthesis=synthesis,
            stats={
                "seed_count": len(seeds),
                "candidate_count": len(nodes),
                "evidence_count": len(evidence),
                "distributed": any(item.is_distributed for item in evidence),
                "chain_count": len(chains),
                "conflict_count": len(conflicts),
                "unresolved_conflict_count": sum(
                    1 for conflict in conflicts if not conflict.resolved
                ),
            },
        )

    # ==================== Candidate selection ====================

    def _select_seeds(
        self,
        query: str,
        topic_id: int | None,
        seed_ids: list[int] | None,
        config: ReconstructionConfig,
    ) -> list[Memory]:
        """Pick the memories the reconstruction starts from.

        Explicit ``seed_ids`` are honoured as given. Otherwise all active
        current memories in scope are scanned and the best lexical matches are
        used, so seed selection needs neither embeddings nor an LLM.
        """
        if seed_ids:
            seeds: list[Memory] = []
            seen: set[int] = set()
            for seed_id in seed_ids:
                if seed_id in seen:
                    continue
                memory = self.store.get_memory(seed_id)
                if memory is None:
                    continue
                seen.add(seed_id)
                seeds.append(memory)
            return seeds[: config.max_seeds]

        candidates = self.store.get_memories(
            topic_id=topic_id,
            is_current=True,
            limit=config.scan_limit,
        )
        ranker = EvidenceRanker(self.store, self.calculator, config.ranking)
        ranked = ranker.rank_memories(query, candidates)
        selected = [item.memory for item in ranked[: config.max_seeds]]
        if selected:
            return selected

        # No lexical match: fall back to the most useful current memories so
        # graph expansion can still reach distributed evidence.
        if not candidates:
            return []
        by_utility = self.calculator.rank_by_future_utility(candidates)
        return [memory for memory, _ in by_utility[: config.max_seeds]]

    # ==================== Internal stages ====================

    def _expand_provenance(
        self,
        nodes: list[GraphNode],
        config: ReconstructionConfig,
    ) -> tuple[list[GraphNode], list[ProvenanceTrace]]:
        """Add the evidence lineage of the top-ranked seeds as candidates.

        Provenance expansion is how AM reaches the *sources* behind a distilled
        semantic memory. Added candidates carry a synthetic depth so they are
        ranked as distributed evidence, never as direct matches.
        """
        existing = {node.memory_id for node in nodes}
        traces: list[ProvenanceTrace] = []
        added: list[GraphNode] = []

        seeds = [node for node in nodes if node.is_seed]
        for node in seeds[: config.provenance_for_top]:
            chain = self.provenance.build_chain(node.memory_id, max_depth=config.provenance_depth)
            for provenance_node in chain.nodes:
                traces.append(
                    ProvenanceTrace(
                        memory_id=provenance_node.memory_id,
                        root_memory_id=node.memory_id,
                        relationship=provenance_node.relationship,
                        depth=provenance_node.depth,
                        resolution=provenance_node.resolution,
                        content_preview=provenance_node.content_preview,
                    )
                )
                if provenance_node.depth == 0 or provenance_node.memory_id in existing:
                    continue
                memory = self.store.get_memory(provenance_node.memory_id)
                if memory is None:
                    continue
                existing.add(provenance_node.memory_id)
                added.append(
                    GraphNode(
                        memory=memory,
                        depth=provenance_node.depth,
                        seed_id=node.memory_id,
                        is_seed=False,
                        path=[node.memory_id, provenance_node.memory_id],
                        edges=[],
                        path_strength=1.0 / (1.0 + provenance_node.depth),
                    )
                )

        return [*nodes, *added], traces

    def _apply_temporal_filter(
        self,
        nodes: list[GraphNode],
        timestamp: datetime,
        config: ReconstructionConfig,
    ) -> list[GraphNode]:
        """Keep only candidates valid at ``timestamp`` (plan #19-#21)."""
        if config.include_out_of_window:
            return nodes
        return [node for node in nodes if self.temporal.is_valid_at(node.memory, timestamp)]

    def _check_contradictions(
        self,
        evidence: list[EvidenceItem],
        timestamp: datetime,
        config: ReconstructionConfig,
    ) -> list[ConflictContext]:
        """Surface contradictions touching the evidence (plan #16/#21).

        A contradiction is *resolved* when the two records' validity windows no
        longer overlap at ``timestamp``; unresolved conflicts are reported
        explicitly so an answering model never silently picks a side.
        """
        if config.max_conflicts == 0:
            return []

        evidence_ids = {item.memory_id for item in evidence}
        conflicts: list[ConflictContext] = []
        seen: set[tuple[int, int]] = set()

        for item in evidence:
            for edge in self.contradictions.get_edges(item.memory_id):
                memory_a_id = edge.memory_a.id or 0
                memory_b_id = edge.memory_b.id or 0
                pair = (min(memory_a_id, memory_b_id), max(memory_a_id, memory_b_id))
                if pair in seen:
                    continue
                seen.add(pair)

                both_valid = self.temporal.is_valid_at(
                    edge.memory_a, timestamp
                ) and self.temporal.is_valid_at(edge.memory_b, timestamp)
                conflicts.append(
                    ConflictContext(
                        memory_id=memory_a_id,
                        other_memory_id=memory_b_id,
                        severity=edge.severity,
                        resolved=not both_valid,
                        other_in_evidence=(
                            memory_b_id in evidence_ids and memory_a_id in evidence_ids
                        ),
                        detail=(
                            "both records valid at reconstruction time: state is contested"
                            if both_valid
                            else "validity windows separated: contradiction resolved as a "
                            "change of state"
                        ),
                    )
                )
                if len(conflicts) >= config.max_conflicts:
                    return conflicts

        conflicts.sort(key=lambda conflict: (-conflict.severity, conflict.memory_id))
        return conflicts

    def _collect_chains(self, evidence: list[EvidenceItem]) -> list[list[int]]:
        """Distributed chains (A -> B -> C) that contributed evidence."""
        chains: list[list[int]] = []
        for item in evidence:
            if not item.is_distributed or len(item.path) < 2:
                continue
            if item.path not in chains:
                chains.append(list(item.path))
        return chains

    def _synthesize(
        self,
        evidence: list[EvidenceItem],
        conflicts: list[ConflictContext],
        config: ReconstructionConfig,
    ) -> str:
        """Compose the evidence package text deterministically.

        Each line carries the resolution level, the score and the memory id, so
        the answering model can cite its evidence and a reviewer can trace it.
        No text is invented: contents are copied from the memories themselves.
        """
        if not evidence:
            return ""

        per_item = max(80, config.synthesis_max_chars // len(evidence))
        lines: list[str] = []
        for item in evidence:
            memory = item.memory
            content = " ".join(memory.content.split())
            if len(content) > per_item:
                content = content[:per_item].rstrip() + "..."
            descriptor = (
                "direct" if item.depth == 0 else f"via {' -> '.join(str(p) for p in item.path)}"
            )
            lines.append(
                f"[{memory.resolution.name} score={item.score:.2f} "
                f"memory={item.memory_id} {descriptor}] {content}"
            )

        for conflict in conflicts:
            if conflict.resolved:
                continue
            lines.append(
                f"[CONFLICT memory={conflict.memory_id} vs {conflict.other_memory_id} "
                f"severity={conflict.severity:.2f}] the two records are both valid at this "
                "time; the state is contested"
            )

        return "\n".join(lines)

    # ==================== Convenience ====================

    def reconstruct_multi_hop(
        self,
        query: str,
        chain: list[int],
        timestamp: datetime | None = None,
        config: ReconstructionConfig | None = None,
    ) -> ReconstructedEvidencePackage:
        """Reconstruct along an explicitly supplied memory chain.

        Used to verify that a known multi-memory chain (A -> B -> C) is
        recoverable, and by tests / benchmarks to pin a chain.
        """
        effective = config or self.config
        graph_config = GraphTraversalConfig(
            max_depth=max(len(chain) + 1, effective.graph.max_depth),
            min_strength=effective.graph.min_strength,
            max_nodes=effective.graph.max_nodes,
            expandable_types=effective.graph.expandable_types,
            include_non_current=effective.graph.include_non_current,
        )
        pinned = ReconstructionConfig(
            max_seeds=len(chain) or 1,
            scan_limit=effective.scan_limit,
            include_out_of_window=effective.include_out_of_window,
            graph=graph_config,
            expand_provenance=effective.expand_provenance,
            provenance_depth=effective.provenance_depth,
            provenance_for_top=effective.provenance_for_top,
            check_contradictions=effective.check_contradictions,
            max_conflicts=effective.max_conflicts,
            ranking=effective.ranking,
            synthesis_max_chars=effective.synthesis_max_chars,
        )
        return self.reconstruct(
            query=query,
            timestamp=timestamp,
            seed_ids=list(chain),
            config=pinned,
        )


def create_memory_reconstructor(
    store: MemoryStore,
    config: ReconstructionConfig | None = None,
    calculator: StateSignalCalculator | None = None,
) -> MemoryReconstructor:
    return MemoryReconstructor(store, config, calculator)


__all__ = [
    "ConflictContext",
    "MemoryReconstructor",
    "ProvenanceTrace",
    "ReconstructedEvidencePackage",
    "ReconstructionConfig",
    "create_memory_reconstructor",
]
