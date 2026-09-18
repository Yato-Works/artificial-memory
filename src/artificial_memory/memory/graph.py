"""Memory graph traversal (AM v0.2.0 Phase 3).

Plan #15 (Memory Reconstruction) and #16 (Memory Graph): reconstruction does
not rely on a single vector match. It walks the memory graph instead::

    query
      -> initial candidates
      -> semantic relations
      -> temporal relations
      -> provenance expansion
      -> contradiction checks
      -> multi-memory synthesis
      -> reconstructed evidence package

This module owns the *traversal* half of that pipeline: given seed memories it
discovers the connected evidence neighbourhood deterministically.

Design notes
------------
* Supporting relations (``RELATED`` / ``ELABORATES`` / ``SUMMARIZES`` /
  ``CAUSES`` / ``FOLLOWS``) are expandable. Conflict and lifecycle relations
  (``CONTRADICTS`` / ``SUPERSEDES``) are **never** expanded implicitly: a
  contradicted or superseded memory is not *supporting* evidence. Those edges
  are surfaced explicitly by the reconstruction layer's contradiction checks
  and supersession queries instead.
* Traversal is breadth-first and fully deterministic: neighbours are ordered by
  descending strength, then association id, then memory id. Ties can never
  reorder results across runs.
* Cycles are guarded with a visited set, so a cyclic graph terminates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Association, AssociationType, Memory

# Relations that carry supporting, interpretable evidence and may be expanded.
DEFAULT_EXPANDABLE_TYPES: frozenset[AssociationType] = frozenset(
    {
        AssociationType.RELATED,
        AssociationType.ELABORATES,
        AssociationType.SUMMARIZES,
        AssociationType.CAUSES,
        AssociationType.FOLLOWS,
    }
)

# Conflict / lifecycle relations: surfaced explicitly, never expanded.
LIFECYCLE_TYPES: frozenset[AssociationType] = frozenset(
    {
        AssociationType.CONTRADICTS,
        AssociationType.SUPERSEDES,
    }
)


@dataclass
class GraphTraversalConfig:
    """Bounds and filters for graph expansion."""

    max_depth: int = 2
    min_strength: float = 0.3
    max_nodes: int = 64
    expandable_types: frozenset[AssociationType] = DEFAULT_EXPANDABLE_TYPES
    # Archived / dormant memories stay out of the walk unless explicitly
    # requested: they are reachable through RESTORE or resolution recovery.
    include_non_current: bool = False

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if not 0.0 <= self.min_strength <= 1.0:
            raise ValueError("min_strength must be within [0, 1]")


@dataclass
class GraphNode:
    """A memory discovered by graph expansion, with the walk that found it."""

    memory: Memory
    depth: int
    seed_id: int
    is_seed: bool = False
    path: list[int] = field(default_factory=list)
    edges: list[Association] = field(default_factory=list)
    path_strength: float = 1.0

    @property
    def memory_id(self) -> int:
        return self.memory.id or 0

    @property
    def graph_proximity(self) -> float:
        """Proximity of this node to its seed, in ``(0, 1]``.

        Seeds score 1.0; a neighbour reached over strong edges at shallow depth
        scores higher than the same relation reached indirectly.
        """
        if self.is_seed:
            return 1.0
        return self.path_strength / (1.0 + self.depth)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "depth": self.depth,
            "seed_id": self.seed_id,
            "is_seed": self.is_seed,
            "path": list(self.path),
            "path_strength": self.path_strength,
            "edge_ids": [edge.id for edge in self.edges],
        }


class MemoryGraph:
    """Traverses the memory association graph deterministically."""

    def __init__(self, store: MemoryStore, config: GraphTraversalConfig | None = None):
        self.store = store
        self.config = config or GraphTraversalConfig()

    # ==================== Neighbourhood ====================

    def neighbours(
        self,
        memory_id: int,
        config: GraphTraversalConfig | None = None,
    ) -> list[tuple[Memory, Association]]:
        """Associations touching ``memory_id`` that may be expanded.

        Returns ``(neighbour_memory, association)`` pairs. Associations are
        treated as undirected for expansion, but the returned
        :class:`Association` keeps its original direction so callers can tell
        which side was the source. Ordered deterministically.
        """
        config = config or self.config
        associations = self.store.get_associations(memory_id)
        found: list[tuple[Memory, Association]] = []
        for association in associations:
            if association.association_type not in config.expandable_types:
                continue
            if association.strength < config.min_strength:
                continue
            other_id = self._other_side(memory_id, association)
            if other_id is None:
                continue
            other = self.store.get_memory(other_id)
            if other is None:
                continue
            # A neighbour is never a seed, so it must satisfy the neighbourhood
            # eligibility rules (archived / superseded memories stay out of the
            # walk unless explicitly requested).
            if not self._is_eligible(other, config, is_seed=False):
                continue
            found.append((other, association))
        self._sort_pairs(found)
        return found

    def has_expandable_edges(
        self,
        memory_id: int,
        config: GraphTraversalConfig | None = None,
    ) -> bool:
        return bool(self.neighbours(memory_id, config))

    # ==================== Traversal ====================

    def traverse(
        self,
        seed_ids: list[int],
        config: GraphTraversalConfig | None = None,
        max_depth: int | None = None,
    ) -> list[GraphNode]:
        """Breadth-first expansion from ``seed_ids``.

        The returned list starts with the seeds (in the given order) followed by
        discovered nodes ordered by ``(depth, -path_strength, memory id)``.
        Each node records the seed that reached it first and the traversed path,
        so multi-memory chains (A -> B -> C) are recoverable.
        """
        config = config or self.config
        depth_limit = config.max_depth if max_depth is None else max_depth

        nodes: list[GraphNode] = []
        visited: set[int] = set()
        frontier: list[GraphNode] = []

        for seed_id in seed_ids:
            if seed_id in visited:
                continue
            seed = self.store.get_memory(seed_id)
            if seed is None:
                continue
            if not self._is_eligible(seed, config, is_seed=True):
                continue
            visited.add(seed_id)
            node = GraphNode(
                memory=seed,
                depth=0,
                seed_id=seed_id,
                is_seed=True,
                path=[seed_id],
                edges=[],
                path_strength=1.0,
            )
            nodes.append(node)
            frontier.append(node)

        depth = 0
        while frontier and depth < depth_limit and len(nodes) < config.max_nodes:
            next_frontier: list[GraphNode] = []
            for current in frontier:
                if len(nodes) >= config.max_nodes:
                    break
                for neighbour, association in self.neighbours(current.memory_id, config):
                    if len(nodes) >= config.max_nodes:
                        break
                    neighbour_id = neighbour.id
                    if neighbour_id is None or neighbour_id in visited:
                        continue
                    if not self._is_eligible(neighbour, config, is_seed=False):
                        continue
                    visited.add(neighbour_id)
                    node = GraphNode(
                        memory=neighbour,
                        depth=current.depth + 1,
                        seed_id=current.seed_id,
                        is_seed=False,
                        path=[*current.path, neighbour_id],
                        edges=[*current.edges, association],
                        path_strength=current.path_strength * association.strength,
                    )
                    nodes.append(node)
                    next_frontier.append(node)
            frontier = self._sort_nodes(next_frontier)
            depth += 1

        return nodes

    def find_path(
        self,
        source_id: int,
        target_id: int,
        max_depth: int | None = None,
        config: GraphTraversalConfig | None = None,
    ) -> list[Association]:
        """Shortest association path between two memories, or ``[]``.

        Deterministic BFS: the first path found is the one reported, and ties
        are broken by the neighbour ordering.
        """
        config = config or self.config
        depth_limit = config.max_depth if max_depth is None else max_depth
        if source_id == target_id:
            return []

        visited = {source_id}
        queue: list[tuple[int, list[Association]]] = [(source_id, [])]
        while queue and len(visited) < config.max_nodes:
            current_id, path = queue.pop(0)
            if len(path) >= depth_limit:
                continue
            for neighbour, association in self.neighbours(current_id, config):
                neighbour_id = neighbour.id
                if neighbour_id is None or neighbour_id in visited:
                    continue
                new_path = [*path, association]
                if neighbour_id == target_id:
                    return new_path
                visited.add(neighbour_id)
                queue.append((neighbour_id, new_path))
        return []

    # ==================== Internals ====================

    @staticmethod
    def _other_side(memory_id: int, association: Association) -> int | None:
        if association.source_memory_id == memory_id:
            return association.target_memory_id
        if association.target_memory_id == memory_id:
            return association.source_memory_id
        return None

    @staticmethod
    def _sort_pairs(pairs: list[tuple[Memory, Association]]) -> None:
        pairs.sort(
            key=lambda item: (
                -item[1].strength,
                item[1].id if item[1].id is not None else 0,
                item[0].id if item[0].id is not None else 0,
            )
        )

    @staticmethod
    def _sort_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
        return sorted(
            nodes,
            key=lambda node: (
                node.depth,
                -node.path_strength,
                node.memory.id if node.memory.id is not None else 0,
            ),
        )

    @staticmethod
    def _is_eligible(memory: Memory, config: GraphTraversalConfig, is_seed: bool) -> bool:
        if is_seed:
            return True
        return config.include_non_current or memory.is_current


def create_memory_graph(
    store: MemoryStore,
    config: GraphTraversalConfig | None = None,
) -> MemoryGraph:
    return MemoryGraph(store, config)


__all__ = [
    "DEFAULT_EXPANDABLE_TYPES",
    "LIFECYCLE_TYPES",
    "GraphNode",
    "GraphTraversalConfig",
    "MemoryGraph",
    "create_memory_graph",
]
