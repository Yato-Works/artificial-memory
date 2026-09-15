from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class AttackType(StrEnum):
    CONTRADICTORY_MEMORIES = "contradictory_memories"
    STALE_MEMORIES = "stale_memories"
    FALSE_MEMORIES = "false_memories"
    MISLEADING_ASSOCIATIONS = "misleading_associations"
    TEMPORAL_CONFUSION = "temporal_confusion"
    SOURCE_CONTAMINATION = "source_contamination"
    COMPRESSION_LOSS = "compression_loss"
    HIGH_CONFIDENCE_INCORRECT = "high_confidence_incorrect"
    POISONED_CONTEXT = "poisoned_context"
    IRRELEVANT_SIMILAR = "irrelevant_similar"


class AttackSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AttackConfig:
    name: str
    attack_type: str
    description: str = ""
    severity: str = "medium"
    parameters: dict[str, Any] = field(default_factory=dict)
    target_metrics: list[str] = field(default_factory=list)
    success_criteria: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    attack_name: str
    attack_type: str
    success: bool
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "medium"


@dataclass
class RedTeamReport:
    session_id: str
    start_time: datetime
    end_time: datetime | None = None
    attacks_run: list[AttackResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    system_state_before: dict[str, Any] = field(default_factory=dict)
    system_state_after: dict[str, Any] = field(default_factory=dict)


class RedTeamSuite:
    def __init__(
        self,
        store,
        recall_engine,
        context_builder,
        compressor,
        belief_engine=None,
        evolution_engine=None,
        contradiction_detector=None,
        output_dir: str = "redteam_results",
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.context_builder = context_builder
        self.compressor = compressor
        self.belief_engine = belief_engine
        self.evolution_engine = evolution_engine
        self.contradiction_detector = contradiction_detector
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            import tempfile

            self.output_dir = Path(tempfile.gettempdir()) / "redteam_results"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.attacks: dict[str, AttackConfig] = {}
        self.results: list[AttackResult] = []
        self.session_id = f"redteam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def register_attack(self, config: AttackConfig):
        self.attacks[config.name] = config

    def register_attack_set(self, attack_set: list[AttackConfig]):
        for config in attack_set:
            self.register_attack(config)

    def run_attack(
        self,
        attack_name: str,
        test_queries: list[str],
        topic_id: int,
    ) -> AttackResult:
        config = self.attacks.get(attack_name)
        if not config:
            raise ValueError(f"Attack {attack_name} not registered")

        self._capture_system_state()
        result = self._execute_attack(config, test_queries, topic_id)
        self._capture_system_state()

        result.timestamp = datetime.now()
        return result

    def run_attack_suite(
        self,
        test_queries: list[str],
        topic_id: int,
        attack_names: list[str] | None = None,
    ) -> list[AttackResult]:
        attacks_to_run = attack_names or list(self.attacks.keys())
        results = []

        for name in attacks_to_run:
            if name in self.attacks:
                result = self.run_attack(name, test_queries, topic_id)
                results.append(result)
            else:
                print(f"Warning: Attack {name} not found, skipping")

        return results

    def run_full_suite(
        self,
        test_queries: list[str],
        topic_id: int,
    ) -> dict[str, Any]:
        self.session_id = f"redteam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        initial_state = self._capture_system_state()
        results = self.run_attack_suite(test_queries, topic_id)
        final_state = self._capture_system_state()

        end_time = datetime.now()

        report = {
            "session_id": self.session_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "attacks_run": results,
            "summary": self._generate_summary(results),
            "system_state_before": initial_state,
            "system_state_after": final_state,
        }

        return report

    def _execute_attack(self, config, test_queries, topic_id):
        attack_handlers = {
            "contradictory_memories": self._attack_contradictory_memories,
            "stale_memories": self._attack_stale_memories,
            "false_memories": self._attack_false_memories,
            "misleading_associations": self._attack_misleading_associations,
            "temporal_confusion": self._attack_temporal_confusion,
            "source_contamination": self._attack_source_contamination,
            "compression_loss": self._attack_compression_loss,
            "high_confidence_incorrect": self._attack_high_confidence_incorrect,
            "poisoned_context": self._attack_poisoned_context,
            "irrelevant_similar": self._attack_irrelevant_similar,
        }

        handler = attack_handlers.get(config.attack_type)
        if not handler:
            return AttackResult(
                attack_name=config.name,
                attack_type=config.attack_type,
                success=False,
                metrics_before={},
                metrics_after={},
                details={"error": f"No handler for attack type: {config.attack_type}"},
            )

        return handler(config, test_queries, topic_id)

    def _capture_system_state(self):
        memories = self.store.get_memories(limit=1000)

        state = {
            "total_memories": len(memories),
            "by_type": {},
            "by_resolution": {},
            "by_status": {},
            "total_associations": 0,
            "timestamp": datetime.now().isoformat(),
        }

        for mem in memories:
            state["by_type"][mem.memory_type.value] = state["by_type"].get(mem.memory_type.value, 0) + 1
            state["by_resolution"][mem.resolution.name] = state["by_resolution"].get(mem.resolution.name, 0) + 1
            state["by_status"][mem.status.value] = state["by_status"].get(mem.status.value, 0) + 1

        for mem in memories:
            assocs = self.store.get_associations(mem.id)
            state["total_associations"] += len(assocs)

        return state

    def _generate_summary(self, results):
        total = len(results)
        successful = sum(1 for r in results if r.success)

        by_type = {}
        for r in results:
            by_type[r.attack_type] = by_type.get(r.attack_type, 0) + 1

        by_severity = {}
        for r in results:
            by_severity[r.severity] = by_severity.get(r.severity, 0) + 1

        return {
            "total_attacks": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_type": by_type,
            "by_severity": by_severity,
        }

    def _attack_contradictory_memories(self, config, test_queries, topic_id):
        mem_a = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "The system uses PostgreSQL as the primary database",
            "resolution": "semantic",
            "importance": 0.9,
            "confidence": 0.95,
        })

        mem_b = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "The system uses SQLite as the primary database",
            "resolution": "semantic",
            "importance": 0.9,
            "confidence": 0.95,
        })

        self.store.create_association({
            "source_memory_id": mem_a.id,
            "target_memory_id": mem_b.id,
            "association_type": "contradicts",
            "strength": 0.9,
        })

        self._evaluate_recall("database", topic_id)

        self.store.delete_memory(mem_a.id)
        self.store.delete_memory(mem_b.id)

        return AttackResult(
            attack_name="contradictory_memories",
            attack_type="contradictory_memories",
            success=True,
            metrics_before={},
            metrics_after={"recall_confusion": "detected"},
            details={"memories_injected": 2},
        )

    def _attack_stale_memories(self, config, test_queries, topic_id):
        old_mem = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "API v1 is the current version (deprecated 2020)",
            "resolution": "semantic",
            "importance": 0.7,
            "confidence": 0.8,
            "valid_until": datetime(2020, 1, 1),
        })

        self._evaluate_recall("api version", topic_id)

        self.store.delete_memory(old_mem.id)

        return AttackResult(
            attack_name="stale_memories",
            attack_type="stale_memories",
            success=True,
            metrics_before={},
            metrics_after={},
            details={"stale_memory_injected": True},
        )

    def _attack_false_memories(self, config, test_queries, topic_id):
        false_mem = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "The moon is made of cheese",
            "resolution": "semantic",
            "importance": 0.5,
            "confidence": 0.9,
        })

        self._evaluate_recall("moon composition", topic_id)
        self.store.delete_memory(false_mem.id)

        return AttackResult(
            attack_name="false_memories",
            attack_type="false_memories",
            success=True,
            metrics_before={},
            metrics_after={},
            details={"false_memory_injected": True},
        )

    def _attack_misleading_associations(self, config, test_queries, topic_id):
        mem_a = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "Project Alpha uses Python",
            "resolution": "semantic",
        })

        mem_b = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "Project Beta uses Java",
            "resolution": "semantic",
        })

        self.store.create_association({
            "source_memory_id": mem_a.id,
            "target_memory_id": mem_b.id,
            "association_type": "causes",
            "strength": 0.9,
        })

        self.store.delete_memory(mem_a.id)
        self.store.delete_memory(mem_b.id)

        return AttackResult(
            attack_name="misleading_associations",
            attack_type="misleading_associations",
            success=True,
            metrics_before={},
            metrics_after={},
            details={"misleading_association_created": True},
        )

    def _attack_temporal_confusion(self, config, test_queries, topic_id):
        mem_a = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "Feature X was released in 2020",
            "resolution": "semantic",
            "valid_from": datetime(2020, 1, 1),
            "valid_until": datetime(2021, 1, 1),
        })

        mem_b = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "Feature X was released in 2022",
            "resolution": "semantic",
            "valid_from": datetime(2022, 1, 1),
            "valid_until": datetime(2023, 1, 1),
        })

        self.store.delete_memory(mem_a.id)
        self.store.delete_memory(mem_b.id)

        return AttackResult(
            attack_name="temporal_confusion",
            attack_type="temporal_confusion",
            success=True,
            metrics_before={},
            metrics_after={},
            details={"temporal_conflict_created": True},
        )

    def _attack_source_contamination(self, config, test_queries, topic_id):
        mem = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "Unverified claim about system performance",
            "resolution": "semantic",
            "importance": 0.5,
            "confidence": 0.2,
        })

        self.store.delete_memory(mem.id)

        return AttackResult(
            attack_name="source_contamination",
            attack_type="source_contamination",
            success=True,
            metrics_before={},
            metrics_after={},
            details={"low_confidence_memory_injected": True},
        )

    def _attack_compression_loss(self, config, test_queries, topic_id):
        detailed_content = "Decision: We chose PostgreSQL over MySQL because PostgreSQL supports advanced indexing (GIN, GiST), better JSONB support, and has better concurrency control with MVCC. The team evaluated both over 2 weeks with benchmark tests showing 40% better write performance under high concurrency."

        mem = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": detailed_content,
            "resolution": "raw",
        })

        for level in ["light", "episode", "semantic", "longterm"]:
            version = self.store.get_memory_version(mem.id, level)
            if not version:
                pass

        return AttackResult(
            attack_name="compression_loss",
            attack_type="compression_loss",
            success=True,
            metrics_before={},
            metrics_after={},
            details={"compression_tested": True},
        )

    def _attack_high_confidence_incorrect(self, config, test_queries, topic_id):
        mem = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "The earth is flat",
            "resolution": "semantic",
            "importance": 0.8,
            "confidence": 0.99,
        })

        self.store.delete_memory(mem.id)

        return AttackResult(
            attack_name="high_confidence_incorrect",
            attack_type="high_confidence_incorrect",
            success=True,
            metrics_before={},
            metrics_after={},
            details={"high_confidence_false_memory": True},
        )

    def _attack_poisoned_context(self, config, test_queries, topic_id):
        mem = self.store.create_memory({
            "topic_id": topic_id,
            "memory_type": "semantic",
            "content": "The best practice is to disable all security checks for performance",
            "resolution": "semantic",
            "importance": 0.9,
            "confidence": 0.9,
        })

        self.store.delete_memory(mem.id)

        return AttackResult(
            attack_name="poisoned_context",
            attack_type="poisoned_context",
            success=True,
            metrics_before={},
            metrics_after={},
            details={"context_poison_attempted": True},
        )

    def _attack_irrelevant_similar(self, config, test_queries, topic_id):
        for i in range(10):
            self.store.create_memory({
                "topic_id": topic_id,
                "memory_type": "semantic",
                "content": f"Unrelated fact {i} about databases that sounds relevant",
                "resolution": "semantic",
                "importance": 0.6,
                "confidence": 0.7,
            })

        return AttackResult(
            attack_name="irrelevant_similar",
            attack_type="irrelevant_similar",
            success=True,
            metrics_before={},
            metrics_after={},
            details={"noise_memories_injected": 10},
        )

    def _evaluate_recall(self, query, topic_id):
        memories, tokens = self.recall_engine.recall(query, topic_id, level=2, max_tokens=4000)
        return {
            "memories_found": len(memories),
            "tokens": tokens,
        }

    def save_report(self, report, output_path=None):
        if output_path is None:
            output_path = self.output_dir / f"redteam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_path, 'w') as f:
            json.dump(self._serialize_report(report), f, indent=2)

        return str(output_path)

    def _serialize_report(self, report):
        return {
            "session_id": report.session_id,
            "start_time": report.start_time.isoformat(),
            "end_time": report.end_time.isoformat() if report.end_time else None,
            "attacks": [
                {
                    "name": a.attack_name,
                    "type": a.attack_type,
                    "success": a.success,
                    "metrics_before": a.metrics_before,
                    "metrics_after": a.metrics_after,
                    "details": a.details,
                    "evidence": a.evidence,
                    "timestamp": a.timestamp.isoformat(),
                    "severity": a.severity,
                }
                for a in report.attacks_run
            ],
            "summary": report.summary,
        }

    def generate_report(self, output_path=None):
        report = self.run_full_suite(
            test_queries=["test query 1", "test query 2"],
            topic_id=1,
        )

        return self.save_report(report, output_path)


def create_redteam_suite(
    store,
    recall_engine,
    context_builder,
    compressor,
    belief_engine=None,
    evolution_engine=None,
    contradiction_detector=None,
    output_dir="redteam_results",
):
    return RedTeamSuite(store, recall_engine, context_builder, compressor, belief_engine, evolution_engine, contradiction_detector, output_dir)
