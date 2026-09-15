from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Association, AssociationType


class DependencyType(StrEnum):
    """Types of memory dependencies."""
    CAUSAL = "causal"           # A causes B
    TEMPORAL = "temporal"       # A happens before B
    LOGICAL = "logical"         # A implies B
    COMPOSITIONAL = "compositional"  # A is part of B
    EVIDENTIAL = "evidential"   # A is evidence for B
    SEMANTIC = "semantic"       # A elaborates on B


@dataclass
class Dependency:
    """A directed dependency between memories."""
    source_id: int
    target_id: int
    dep_type: DependencyType
    strength: float = 0.5  # 0-1
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactAnalysis:
    """Result of impact analysis for a memory change."""
    memory_id: int
    affected_memories: list[tuple[int, float]]  # (memory_id, impact_score)
    affected_beliefs: list[int] = field(default_factory=list)
    cascade_risk: float = 0.0
    recommendation: str = ""


class DependencyGraph:
    """Manages the memory dependency graph for impact analysis and invalidation propagation.

    Enables:
    - What depends on this memory?
    - What would be affected if this memory changed?
    - Invalidation propagation
    - Consistency checks
    """

    def __init__(self, store: MemoryStore):
        self.store = store
        # In-memory graph (would be persisted in production)
        self._edges: dict[int, list[Dependency]] = defaultdict(list)  # source -> [deps]
        self._reverse_edges: dict[int, list[Dependency]] = defaultdict(list)  # target -> [deps]
        self._load_from_store()

    def _load_from_store(self) -> None:
        """Load existing associations as dependencies."""
        # This would query the associations table
        # For now, we build from existing associations on demand
        pass

    def add_dependency(
        self,
        source_id: int,
        target_id: int,
        dep_type: DependencyType,
        strength: float = 0.5,
        metadata: dict | None = None,
    ) -> Dependency:
        """Add a dependency edge."""
        dep = Dependency(
            source_id=source_id,
            target_id=target_id,
            dep_type=dep_type,
            strength=strength,
            metadata=metadata or {},
        )

        self._edges[source_id].append(dep)
        self._reverse_edges[target_id].append(dep)

        # Also create Association in store
        assoc_type = self._dep_type_to_assoc(dep_type)
        self.store.create_association(Association(
            source_memory_id=source_id,
            target_memory_id=target_id,
            association_type=assoc_type,
            strength=strength,
        ))

        return dep

    def remove_dependency(self, source_id: int, target_id: int) -> bool:
        """Remove a dependency edge."""
        removed = False

        if source_id in self._edges:
            self._edges[source_id] = [d for d in self._edges[source_id]
                                       if d.target_id != target_id]
            if not self._edges[source_id]:
                del self._edges[source_id]
            removed = True

        if target_id in self._reverse_edges:
            self._reverse_edges[target_id] = [d for d in self._reverse_edges[target_id]
                                               if d.source_id != source_id]
            if not self._reverse_edges[target_id]:
                del self._reverse_edges[target_id]
            removed = True

        return removed

    def get_dependencies(self, memory_id: int) -> list[Dependency]:
        """Get all outgoing dependencies (what this memory depends on)."""
        return self._edges.get(memory_id, [])

    def get_dependents(self, memory_id: int) -> list[Dependency]:
        """Get all incoming dependencies (what depends on this memory)."""
        return self._reverse_edges.get(memory_id, [])

    def get_all_dependents(self, memory_id: int, max_depth: int = 3) -> list[tuple[int, float]]:
        """Get all transitive dependents with impact scores.

        Returns: List of (memory_id, impact_score) sorted by impact.
        """
        visited = set()
        queue = deque([(memory_id, 1.0)])
        dependents = []

        while queue:
            current_id, current_impact = queue.popleft()

            if current_id in visited or len(visited) > max_depth * 10:
                continue
            visited.add(current_id)

            for dep in self._reverse_edges.get(current_id, []):
                if dep.source_id not in visited:
                    impact = current_impact * dep.strength * 0.8  # Decay per hop
                    if impact > 0.1:  # Threshold
                        dependents.append((dep.source_id, impact))
                        queue.append((dep.source_id, impact))

        # Sort by impact descending
        dependents.sort(key=lambda x: -x[1])
        return dependents

    def get_dependency_path(self, source_id: int, target_id: int, max_depth: int = 5) -> list[Dependency] | None:
        """Find a dependency path from source to target."""
        # BFS
        queue = deque([(source_id, [])])
        visited = set()

        while queue:
            current, path = queue.popleft()

            if current == target_id:
                return path

            if current in visited or len(path) >= max_depth:
                continue
            visited.add(current)

            for dep in self._edges.get(current, []):
                if dep.target_id not in visited:
                    queue.append((dep.target_id, path + [dep]))

        return None

    def analyze_impact(self, memory_id: int) -> ImpactAnalysis:
        """Analyze what would be affected if this memory changed."""
        dependents = self.get_all_dependents(memory_id)

        # Calculate cascade risk
        high_impact = sum(1 for _, score in dependents if score > 0.5)
        cascade_risk = min(1.0, high_impact * 0.2 + len(dependents) * 0.05)

        # Generate recommendation
        if cascade_risk > 0.7:
            recommendation = "High cascade risk. Consider creating new version instead of modifying."
        elif cascade_risk > 0.4:
            recommendation = "Moderate cascade risk. Review dependents before changing."
        else:
            recommendation = "Low cascade risk. Safe to modify."

        return ImpactAnalysis(
            memory_id=memory_id,
            affected_memories=dependents,
            cascade_risk=cascade_risk,
            recommendation=recommendation,
        )

    def propagate_invalidation(self, memory_id: int, reason: str) -> list[int]:
        """Propagate invalidation from a memory to its dependents.

        Returns list of affected memory IDs.
        """
        affected = []
        dependents = self.get_all_dependents(memory_id)

        for dep_id, impact in dependents:
            if impact > 0.3:  # Threshold for invalidation
                mem = self.store.get_memory(dep_id)
                if mem:
                    mem.metadata["invalidated_by"] = memory_id
                    mem.metadata["invalidation_reason"] = reason
                    mem.metadata["invalidation_impact"] = impact
                    mem.updated_at = datetime.now()
                    self.store.update_memory(mem)
                    affected.append(dep_id)

        return affected

    def check_consistency(self, memory_id: int) -> list[dict[str, Any]]:
        """Check for consistency issues in dependencies."""
        issues = []

        # Check for circular dependencies
        cycles = self._find_cycles(memory_id)
        for cycle in cycles:
            issues.append({
                "type": "circular_dependency",
                "severity": "high",
                "cycle": cycle,
                "message": f"Circular dependency detected: {' -> '.join(str(x) for x in cycle)}",
            })

        # Check for unsupported dependencies (dependents with no other support)
        dependents = self.get_dependents(memory_id)
        for dep in dependents:
            other_support = [d for d in self._reverse_edges.get(dep.source_id, [])
                            if d.target_id != memory_id]
            if not other_support:
                issues.append({
                    "type": "single_point_of_failure",
                    "severity": "medium",
                    "dependent_id": dep.source_id,
                    "message": f"Memory {dep.source_id} depends solely on {memory_id}",
                })

        return issues

    def _find_cycles(self, start_id: int, max_depth: int = 5) -> list[list[int]]:
        """Find cycles starting from a node."""
        cycles = []

        def dfs(current: int, path: list[int], depth: int):
            if depth > max_depth:
                return

            if current == start_id and len(path) > 1:
                cycles.append(path + [start_id])
                return

            if current in path:
                return

            for dep in self._edges.get(current, []):
                if dep.target_id not in path:
                    dfs(dep.target_id, path + [current], depth + 1)

        dfs(start_id, [], 0)
        return cycles

    def _dep_type_to_assoc(self, dep_type: DependencyType) -> AssociationType:
        mapping = {
            DependencyType.CAUSAL: AssociationType.CAUSES,
            DependencyType.TEMPORAL: AssociationType.FOLLOWS,
            DependencyType.LOGICAL: AssociationType.ELABORATES,
            DependencyType.COMPOSITIONAL: AssociationType.SUMMARIZES,
            DependencyType.EVIDENTIAL: AssociationType.RELATED,
            DependencyType.SEMANTIC: AssociationType.ELABORATES,
        }
        return mapping.get(dep_type, AssociationType.RELATED)

    def get_graph_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        total_nodes = set(self._edges.keys()) | set(self._reverse_edges.keys())
        total_edges = sum(len(v) for v in self._edges.values())

        return {
            "nodes": len(total_nodes),
            "edges": total_edges,
            "avg_degree": total_edges / len(total_nodes) if total_nodes else 0,
            "max_out_degree": max((len(v) for v in self._edges.values()), default=0),
            "max_in_degree": max((len(v) for v in self._reverse_edges.values()), default=0),
        }


def create_dependency_graph(store: MemoryStore) -> DependencyGraph:
    return DependencyGraph(store)
