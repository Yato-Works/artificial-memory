from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.core.interfaces import MemoryStore, RecallEngine
from artificial_memory.core.models import Memory
from artificial_memory.memory.evolution import MemoryEvolutionEngine


class CounterfactualOperation(StrEnum):
    """Types of counterfactual operations."""
    REMOVE_MEMORY = "remove_memory"           # What if memory X didn't exist?
    WEAKEN_MEMORY = "weaken_memory"           # What if memory X had lower confidence?
    STRENGTHEN_MEMORY = "strengthen_memory"   # What if memory X had higher confidence?
    CHANGE_RESOLUTION = "change_resolution"   # What if memory X was at different resolution?
    REORDER_TEMPORAL = "reorder_temporal"     # What if memory X happened at different time?


@dataclass
class CounterfactualScenario:
    """A counterfactual scenario to evaluate."""
    id: str
    operation: CounterfactualOperation
    target_memory_id: int
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CounterfactualResult:
    """Result of a counterfactual evaluation."""
    scenario: CounterfactualScenario
    original_context: str
    counterfactual_context: str
    context_diff: dict[str, Any]  # tokens_changed, parts_added, parts_removed, etc.
    affected_memories: list[int] = field(default_factory=list)
    confidence_delta: float = 0.0
    influence_score: float = 0.0  # How much this memory influences outcomes
    evaluated_at: datetime = field(default_factory=datetime.now)


class CounterfactualEngine:
    """Evaluates counterfactual scenarios: "What would change if memory X was different?"

    Enables measuring Memory Influence Score by evaluating:
    - What context changes if memory is removed?
    - What context changes if memory confidence is altered?
    - What context changes if memory resolution is changed?
    """

    def __init__(
        self,
        store: MemoryStore,
        recall_engine: RecallEngine,
        context_builder,
        evolution_engine: MemoryEvolutionEngine,
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.context_builder = context_builder
        self.evolution_engine = evolution_engine

    def evaluate_scenario(self, scenario: CounterfactualScenario) -> CounterfactualResult:
        """Evaluate a counterfactual scenario."""
        # Get original context
        original_memories = self.store.get_memories(
            topic_id=self._get_topic_for_memory(scenario.target_memory_id),
            limit=100
        )

        # This is a simplified evaluation - in production, would simulate
        # the full recall + context building pipeline

        if scenario.operation == CounterfactualOperation.REMOVE_MEMORY:
            return self._evaluate_removal(scenario, original_memories)
        elif scenario.operation == CounterfactualOperation.WEAKEN_MEMORY:
            return self._evaluate_confidence_change(scenario, original_memories, -0.3)
        elif scenario.operation == CounterfactualOperation.STRENGTHEN_MEMORY:
            return self._evaluate_confidence_change(scenario, original_memories, +0.3)
        elif scenario.operation == CounterfactualOperation.CHANGE_RESOLUTION:
            return self._evaluate_resolution_change(scenario, original_memories)
        elif scenario.operation == CounterfactualOperation.REORDER_TEMPORAL:
            return self._evaluate_temporal_change(scenario, original_memories)
        else:
            raise ValueError(f"Unknown operation: {scenario.operation}")

    def _evaluate_removal(
        self,
        scenario: CounterfactualScenario,
        original_memories: list,
    ) -> CounterfactualResult:
        """Evaluate: what if this memory didn't exist?"""
        target_mem = self.store.get_memory(scenario.target_memory_id)
        if not target_mem:
            raise ValueError(f"Memory {scenario.target_memory_id} not found")

        # Build original context
        original_context = self._build_context_for_memories(
            original_memories,
            scenario.parameters.get("query", "")
        )

        # Build counterfactual context (without target memory)
        cf_memories = [m for m in original_memories if m.id != scenario.target_memory_id]
        counterfactual_context = self._build_context_for_memories(
            cf_memories,
            scenario.parameters.get("query", "")
        )

        # Compute diff
        diff = self._compute_context_diff(original_context, counterfactual_context)

        # Calculate influence score
        influence = self._calculate_influence_score(diff, target_mem)

        return CounterfactualResult(
            scenario=scenario,
            original_context=original_context,
            counterfactual_context=counterfactual_context,
            context_diff=diff,
            affected_memories=[scenario.target_memory_id],
            influence_score=influence,
        )

    def _evaluate_confidence_change(
        self,
        scenario: CounterfactualScenario,
        original_memories: list,
        delta: float,
    ) -> CounterfactualResult:
        """Evaluate: what if this memory's confidence changed?"""
        target_mem = self.store.get_memory(scenario.target_memory_id)
        if not target_mem:
            raise ValueError(f"Memory {scenario.target_memory_id} not found")

        original_conf = target_mem.confidence
        new_conf = max(0.0, min(1.0, original_conf + delta))

        # Simulate by temporarily changing confidence
        original_memories_copy = deepcopy(original_memories)
        for m in original_memories_copy:
            if m.id == scenario.target_memory_id:
                m.confidence = new_conf

        original_context = self._build_context_for_memories(
            original_memories,
            scenario.parameters.get("query", "")
        )

        counterfactual_context = self._build_context_for_memories(
            original_memories_copy,
            scenario.parameters.get("query", "")
        )

        diff = self._compute_context_diff(original_context, counterfactual_context)
        influence = self._calculate_influence_score(diff, target_mem) * abs(delta)

        return CounterfactualResult(
            scenario=scenario,
            original_context=original_context,
            counterfactual_context=counterfactual_context,
            context_diff=diff,
            affected_memories=[scenario.target_memory_id],
            confidence_delta=delta,
            influence_score=influence,
        )

    def _evaluate_resolution_change(
        self,
        scenario: CounterfactualScenario,
        original_memories: list,
    ) -> CounterfactualResult:
        """Evaluate: what if this memory had different resolution?"""
        target_res = scenario.parameters.get("target_resolution")
        if not target_res:
            raise ValueError("target_resolution parameter required")

        target_mem = self.store.get_memory(scenario.target_memory_id)
        if not target_mem:
            raise ValueError(f"Memory {scenario.target_memory_id} not found")

        original_memories_copy = deepcopy(original_memories)
        for m in original_memories_copy:
            if m.id == scenario.target_memory_id:
                m.resolution = target_res
                # Also change content to match resolution
                version = self.store.get_memory_version(m.id, target_res)
                if version:
                    m.content = version.content

        original_context = self._build_context_for_memories(
            original_memories,
            scenario.parameters.get("query", "")
        )

        counterfactual_context = self._build_context_for_memories(
            original_memories_copy,
            scenario.parameters.get("query", "")
        )

        diff = self._compute_context_diff(original_context, counterfactual_context)
        influence = self._calculate_influence_score(diff, target_mem)

        return CounterfactualResult(
            scenario=scenario,
            original_context=original_context,
            counterfactual_context=counterfactual_context,
            context_diff=diff,
            affected_memories=[scenario.target_memory_id],
            influence_score=influence,
        )

    def _evaluate_temporal_change(
        self,
        scenario: CounterfactualScenario,
        original_memories: list,
    ) -> CounterfactualResult:
        """Evaluate: what if this memory happened at different time?"""
        # Simplified: just shift valid_from
        target_mem = self.store.get_memory(scenario.target_memory_id)
        if not target_mem:
            raise ValueError(f"Memory {scenario.target_memory_id} not found")

        days_shift = scenario.parameters.get("days_shift", 30)

        original_memories_copy = deepcopy(original_memories)
        for m in original_memories_copy:
            if m.id == scenario.target_memory_id:
                from datetime import timedelta
                if m.valid_from:
                    m.valid_from += timedelta(days=days_shift)
                if m.valid_until:
                    m.valid_until += timedelta(days=days_shift)

        original_context = self._build_context_for_memories(
            original_memories,
            scenario.parameters.get("query", "")
        )

        counterfactual_context = self._build_context_for_memories(
            original_memories_copy,
            scenario.parameters.get("query", "")
        )

        diff = self._compute_context_diff(original_context, counterfactual_context)
        influence = self._calculate_influence_score(diff, target_mem)

        return CounterfactualResult(
            scenario=scenario,
            original_context=original_context,
            counterfactual_context=counterfactual_context,
            context_diff=diff,
            affected_memories=[scenario.target_memory_id],
            influence_score=influence,
        )

    def _build_context_for_memories(
        self,
        memories: list,
        query: str,
    ) -> str:
        """Build context string from memories (simplified)."""
        if not memories:
            return ""

        # Use context builder if available
        if hasattr(self.context_builder, 'build_context'):
            # Need topic_id - extract from first memory
            topic_id = memories[0].topic_id if memories else None
            return self.context_builder.build_context(query, topic_id, max_tokens=4000)

        # Fallback: simple concatenation
        parts = []
        for m in memories:
            parts.append(f"[{m.memory_type.value}] {m.content[:200]}")
        return "\n---\n".join(parts)

    def _compute_context_diff(
        self,
        original: str,
        counterfactual: str,
    ) -> dict[str, Any]:
        """Compute diff between two contexts."""
        orig_tokens = len(original.split())
        cf_tokens = len(counterfactual.split())

        # Simple line-by-line diff
        orig_lines = original.split('\n')
        cf_lines = counterfactual.split('\n')

        added = [cf_line for cf_line in cf_lines if cf_line not in orig_lines]
        removed = [orig_line for orig_line in orig_lines if orig_line not in cf_lines]

        return {
            "original_tokens": orig_tokens,
            "counterfactual_tokens": cf_tokens,
            "token_delta": cf_tokens - orig_tokens,
            "lines_added": len(added),
            "lines_removed": len(removed),
            "added_content": added[:5],  # Sample
            "removed_content": removed[:5],
        }

    def _calculate_influence_score(self, diff: dict[str, Any], memory: Memory) -> float:
        """Calculate memory influence score from context diff."""
        # Base score from token change
        token_change = abs(diff.get("token_delta", 0))
        token_score = min(1.0, token_change / 100)

        # Weight by memory importance
        importance_weight = memory.importance

        # Weight by resolution (higher resolution = more influence)
        res_weight = 1.0 + (memory.resolution.value / 5) * 0.5

        return min(1.0, (token_score * importance_weight * res_weight))

    def _get_topic_for_memory(self, memory_id: int) -> int | None:
        mem = self.store.get_memory(memory_id)
        return mem.topic_id if mem else None

    def run_influence_analysis(
        self,
        query: str,
        topic_id: int,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Run influence analysis for all memories in a topic.

        Returns memories ranked by their influence on the query context.
        """
        memories = self.store.get_memories(topic_id=topic_id, limit=100)

        results = []
        for mem in memories:
            scenario = CounterfactualScenario(
                id=f"infl_{mem.id}",
                operation=CounterfactualOperation.REMOVE_MEMORY,
                target_memory_id=mem.id,
                parameters={"query": query},
                description=f"Influence of memory {mem.id} on '{query}'",
            )

            result = self.evaluate_scenario(scenario)

            results.append({
                "memory_id": mem.id,
                "memory_type": mem.memory_type.value,
                "resolution": mem.resolution.name,
                "importance": mem.importance,
                "confidence": mem.confidence,
                "influence_score": result.influence_score,
                "context_diff": result.context_diff,
            })

        # Sort by influence score
        results.sort(key=lambda x: -x["influence_score"])
        return results[:top_k]


def create_counterfactual_engine(
    store: MemoryStore,
    recall_engine: RecallEngine,
    context_builder,
    evolution_engine: MemoryEvolutionEngine,
) -> CounterfactualEngine:
    return CounterfactualEngine(store, recall_engine, context_builder, evolution_engine)
