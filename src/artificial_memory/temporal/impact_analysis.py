from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.memory.belief import BeliefEngine
from artificial_memory.memory.dependency import (
    DependencyGraph,
)
from artificial_memory.memory.evolution import MemoryEvolutionEngine


@dataclass
class ChangeSimulation:
    """Simulation of a memory change and its effects."""
    change_type: str  # "modify", "delete", "supersede", "archive"
    target_memory_id: int
    proposed_changes: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImpactPrediction:
    """Prediction of what would be affected by a change."""
    simulation: ChangeSimulation
    affected_memories: list[tuple[int, float]] = field(default_factory=list)  # (memory_id, impact_score)
    affected_beliefs: list[int] = field(default_factory=list)
    affected_decisions: list[int] = field(default_factory=list)
    cascade_depth: int = 0
    cascade_risk: float = 0.0
    recommendation: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class DependencyCycle:
    """A circular dependency detected in the graph."""
    cycle: list[int]
    severity: str  # "low", "medium", "high"
    description: str


class ImpactAnalyzer:
    """Advanced impact analysis for memory changes.

    Capabilities:
    - Predict cascading effects of memory changes
    - Detect circular dependencies
    - Identify single points of failure
    - Simulate "what-if" scenarios
    - Recommend safe modification strategies
    """

    def __init__(
        self,
        store: MemoryStore,
        dependency_graph: DependencyGraph,
        evolution_engine: MemoryEvolutionEngine | None = None,
        belief_engine: BeliefEngine | None = None,
    ):
        self.store = store
        self.dependency_graph = dependency_graph
        self.evolution_engine = evolution_engine
        self.belief_engine = belief_engine

    def predict_impact(self, simulation: ChangeSimulation) -> ImpactPrediction:
        """Predict the impact of a proposed change."""
        prediction = ImpactPrediction(simulation=simulation)

        if simulation.change_type == "modify":
            prediction = self._predict_modification_impact(simulation)
        elif simulation.change_type == "delete":
            prediction = self._predict_deletion_impact(simulation)
        elif simulation.change_type == "supersede":
            prediction = self._predict_supersession_impact(simulation)
        elif simulation.change_type == "archive":
            prediction = self._predict_archival_impact(simulation)

        return prediction

    def _predict_modification_impact(self, simulation: ChangeSimulation) -> ImpactPrediction:
        """Predict impact of modifying a memory."""
        prediction = ImpactPrediction(simulation=simulation)

        # Get direct dependents
        dependents = self.dependency_graph.get_all_dependents(
            simulation.target_memory_id, max_depth=3
        )
        prediction.affected_memories = dependents

        # Get affected beliefs
        if self.belief_engine:
            for belief in self.belief_engine._beliefs.values():
                if simulation.target_memory_id in belief.supporting_evidence:
                    prediction.affected_beliefs.append(belief.id)

        # Check affected decisions
        # (Would need decision-memory mapping)

        # Calculate cascade risk
        prediction.cascade_risk = self._calculate_cascade_risk(
            simulation.target_memory_id, dependents
        )

        # Generate recommendation
        prediction.recommendation = self._generate_recommendation(
            "modify", prediction
        )

        return prediction

    def _predict_deletion_impact(self, simulation: ChangeSimulation) -> ImpactPrediction:
        """Predict impact of deleting a memory."""
        prediction = ImpactPrediction(simulation=simulation)

        # Deletion has maximum impact
        dependents = self.dependency_graph.get_all_dependents(
            simulation.target_memory_id, max_depth=5
        )
        prediction.affected_memories = dependents

        # All beliefs that depend on this memory
        if self.belief_engine:
            for belief in self.belief_engine._beliefs.values():
                if simulation.target_memory_id in belief.supporting_evidence:
                    prediction.affected_beliefs.append(belief.id)

        # Maximum cascade risk
        prediction.cascade_risk = 1.0
        prediction.warnings.append(
            "DELETION: This will permanently remove the memory and may break dependent memories/beliefs"
        )
        prediction.recommendation = "Consider superseding or archiving instead of deleting"

        return prediction

    def _predict_supersession_impact(self, simulation: ChangeSimulation) -> ImpactPrediction:
        """Predict impact of superseding a memory."""
        prediction = ImpactPrediction(simulation=simulation)

        # Similar to modification but with explicit supersession
        dependents = self.dependency_graph.get_all_dependents(
            simulation.target_memory_id, max_depth=3
        )
        prediction.affected_memories = dependents

        if self.belief_engine:
            for belief in self.belief_engine._beliefs.values():
                if simulation.target_memory_id in belief.supporting_evidence:
                    prediction.affected_beliefs.append(belief.id)

        prediction.cascade_risk = self._calculate_cascade_risk(
            simulation.target_memory_id, dependents
        )

        prediction.recommendation = self._generate_recommendation(
            "supersede", prediction
        )

        return prediction

    def _predict_archival_impact(self, simulation: ChangeSimulation) -> ImpactPrediction:
        """Predict impact of archiving a memory."""
        prediction = ImpactPrediction(simulation=simulation)

        # Archiving reduces accessibility but preserves data
        dependents = self.dependency_graph.get_all_dependents(
            simulation.target_memory_id, max_depth=2
        )
        prediction.affected_memories = [
            (mid, score * 0.5) for mid, score in dependents  # Reduced impact
        ]

        if self.belief_engine:
            for belief in self.belief_engine._beliefs.values():
                if simulation.target_memory_id in belief.supporting_evidence:
                    # Beliefs can still use archived memories but with lower confidence
                    prediction.affected_beliefs.append(belief.id)

        prediction.cascade_risk = self._calculate_cascade_risk(
            simulation.target_memory_id, dependents
        ) * 0.5

        prediction.recommendation = self._generate_recommendation(
            "archive", prediction
        )

        return prediction

    def _calculate_cascade_risk(
        self,
        memory_id: int,
        dependents: list[tuple[int, float]],
    ) -> float:
        """Calculate cascade risk from dependents."""
        if not dependents:
            return 0.0

        # High impact dependents (score > 0.5)
        high_impact = sum(1 for _, score in dependents if score > 0.5)

        # Total dependents
        total = len(dependents)

        # Risk factors
        risk = (
            min(1.0, high_impact * 0.3) +
            min(1.0, total * 0.05) +
            self._get_memory_criticality(memory_id) * 0.2
        )

        return min(1.0, risk)

    def _get_memory_criticality(self, memory_id: int) -> float:
        """Get criticality score for a memory (0-1)."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            return 0.0

        criticality = 0.0

        # DECISION memories are more critical
        if memory.memory_type.value == "decision":
            criticality += 0.4

        # High importance
        criticality += memory.importance * 0.3

        # High confidence
        criticality += memory.confidence * 0.2

        # Number of dependents
        dep_count = len(self.dependency_graph.get_dependents(memory_id))
        criticality += min(0.2, dep_count * 0.02)

        return min(1.0, criticality)

    def _generate_recommendation(
        self,
        change_type: str,
        prediction: ImpactPrediction,
    ) -> str:
        """Generate a recommendation based on prediction."""
        risk = prediction.cascade_risk

        if risk > 0.7:
            return f"HIGH RISK: This {change_type} will significantly affect dependent memories. Consider creating a new version instead."
        elif risk > 0.4:
            return f"MODERATE RISK: {len(prediction.affected_memories)} dependents will be affected. Review dependents before proceeding."
        elif risk > 0.2:
            return f"LOW RISK: Some dependents affected. Monitor after {change_type}."
        else:
            return f"MINIMAL RISK: Safe to {change_type}."

    def detect_circular_dependencies(self) -> list[DependencyCycle]:
        """Detect all circular dependencies in the graph."""
        cycles = []
        visited = set()

        # Get all nodes
        all_nodes = set(self.dependency_graph._edges.keys()) | \
                    set(self.dependency_graph._reverse_edges.keys())

        for node in all_nodes:
            if node in visited:
                continue

            cycles_from_node = self._find_cycles_from(node, visited)
            for cycle in cycles_from_node:
                cycles.append(DependencyCycle(
                    cycle=cycle,
                    severity=self._assess_cycle_severity(cycle),
                    description=f"Circular dependency: {' -> '.join(str(n) for n in cycle)} -> {cycle[0]}",
                ))

        return cycles

    def _find_cycles_from(
        self,
        start: int,
        visited: set[int],
        path: list[int] | None = None,
        depth: int = 0,
    ) -> list[list[int]]:
        """Find cycles starting from a node using DFS."""
        if path is None:
            path = []

        if depth > 10:  # Max depth
            return []

        if start in path:
            # Found a cycle
            idx = path.index(start)
            return [path[idx:] + [start]]

        if start in visited:
            return []

        visited.add(start)
        path = path + [start]

        cycles = []
        for dep in self.dependency_graph.get_dependencies(start):
            cycles.extend(self._find_cycles_from(dep.target_id, visited, path, depth + 1))

        return cycles

    def _assess_cycle_severity(self, cycle: list[int]) -> str:
        """Assess the severity of a circular dependency."""
        if len(cycle) <= 2:
            return "high"  # Direct mutual dependency
        elif len(cycle) <= 4:
            return "medium"
        else:
            return "low"

    def find_single_points_of_failure(self) -> list[dict[str, Any]]:
        """Find memories that are single points of failure."""
        spofs = []

        all_nodes = set(self.dependency_graph._edges.keys()) | \
                    set(self.dependency_graph._reverse_edges.keys())

        for node_id in all_nodes:
            dependents = self.dependency_graph.get_dependents(node_id)

            for dep in dependents:
                # Check if this dependent has no other support
                other_support = [
                    d for d in self.dependency_graph.get_dependents(dep.source_id)
                    if d.target_id != node_id
                ]

                if not other_support:
                    spofs.append({
                        "memory_id": node_id,
                        "dependent_id": dep.source_id,
                        "severity": "critical",
                        "message": f"Memory {dep.source_id} depends solely on {node_id}",
                    })

        return spofs

    def simulate_scenario(
        self,
        change_type: str,
        target_memory_id: int,
        parameters: dict[str, Any],
    ) -> ImpactPrediction:
        """Run a simulation of a proposed change."""
        simulation = ChangeSimulation(
            change_type=change_type,
            target_memory_id=target_memory_id,
            proposed_changes=parameters,
        )
        return self.predict_impact(simulation)

    def get_safe_modification_strategy(
        self,
        memory_id: int,
    ) -> dict[str, Any]:
        """Get a safe strategy for modifying a memory."""
        prediction = self.simulate_scenario("modify", memory_id, {})

        return {
            "memory_id": memory_id,
            "current_risk": prediction.cascade_risk,
            "dependents": len(prediction.affected_memories),
            "strategy": "supersede" if prediction.cascade_risk > 0.5 else "modify_in_place",
            "recommendation": prediction.recommendation,
            "preparation_steps": [
                "Create superseding memory with updated content",
                "Link superseding memory to original",
                "Update associations to point to new memory",
                "Archive original memory after verification",
            ],
            "rollback_plan": "Original memory preserved at current resolution; can reactivate if needed",
        }


def create_impact_analyzer(
    store: MemoryStore,
    dependency_graph: DependencyGraph,
    evolution_engine: MemoryEvolutionEngine | None = None,
    belief_engine: BeliefEngine | None = None,
) -> ImpactAnalyzer:
    return ImpactAnalyzer(store, dependency_graph, evolution_engine, belief_engine)
