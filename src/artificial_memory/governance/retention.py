from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from artificial_memory.core.models import Memory, MemoryStatus, ResolutionLevel


class RetentionAction(StrEnum):
    ARCHIVE = "archive"
    COMPRESS = "compress"
    DELETE = "delete"
    ANONYMIZE = "anonymize"
    NOTIFY = "notify"
    REVIEW = "review"


class RetentionTriggerType(StrEnum):
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"
    ACCESS_BASED = "access_based"
    IMPORTANCE_BASED = "importance_based"
    COMPOSITE = "composite"


@dataclass
class RetentionRule:
    rule_id: str
    name: str
    description: str
    trigger_type: RetentionTriggerType
    condition: dict[str, Any]
    action: RetentionAction
    priority: int = 100
    enabled: bool = True
    grace_period_days: int = 0
    applies_to: dict[str, Any] = field(default_factory=dict)  # memory_type, topic, tenant, etc.
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetentionPolicy:
    policy_id: str
    name: str
    description: str
    rules: list[RetentionRule] = field(default_factory=list)
    default_action: str = "review"
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetentionActionPlan:
    plan_id: str
    memory_id: int
    rule_id: str
    action: RetentionAction
    scheduled_at: datetime
    status: str = "pending"  # pending, executing, completed, failed, cancelled
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)


class RetentionPolicyEngine:
    """Manages retention policies and automated memory lifecycle management.

    Features:
    - Time-based retention (age-based archival/deletion)
    - Size-based retention (storage quotas)
    - Access-based retention (LRU-style)
    - Importance-based retention
    - Composite rules with multiple triggers
    - Grace periods and notifications
    - Automated scheduling and execution
    """

    def __init__(
        self,
        store,
        compressor=None,
        consolidation_engine=None,
    ):
        self.store = store
        self.compressor = compressor
        self.consolidation_engine = consolidation_engine

        self.policies: dict[str, RetentionPolicy] = {}
        self.action_plans: dict[str, RetentionActionPlan] = {}
        self.execution_history: list[dict[str, Any]] = []

        self._register_default_policies()

    def _register_default_policies(self):
        # Default time-based policy
        self.register_policy(RetentionPolicy(
            policy_id="default_time_based",
            name="Default Time-Based Retention",
            description="Archive memories older than 7 years",
            rules=[
                RetentionRule(
                    rule_id="archive_after_7_years",
                    name="Archive After 7 Years",
                    description="Move memories older than 7 years to archive",
                    trigger_type=RetentionTriggerType.TIME_BASED,
                    condition={"max_age_days": 2555},  # 7 years
                    action=RetentionAction.ARCHIVE,
                    priority=10,
                ),
                RetentionRule(
                    rule_id="compress_after_1_year",
                    name="Compress After 1 Year",
                    description="Compress memories older than 1 year",
                    trigger_type=RetentionTriggerType.TIME_BASED,
                    condition={"max_age_days": 365},
                    action=RetentionAction.COMPRESS,
                    priority=20,
                ),
                RetentionRule(
                    rule_id="delete_after_20_years",
                    name="Delete After 20 Years",
                    description="Delete memories older than 20 years",
                    trigger_type=RetentionTriggerType.TIME_BASED,
                    condition={"max_age_days": 7300},
                    action=RetentionAction.DELETE,
                    priority=5,
                ),
            ],
        ))

        # Access-based policy
        self.register_policy(RetentionPolicy(
            policy_id="access_based_retention",
            name="Access-Based Retention",
            description="Archive memories not accessed in 2 years",
            rules=[
                RetentionRule(
                    rule_id="archive_unaccessed_2_years",
                    name="Archive Unaccessed Memories",
                    description="Archive memories not accessed in 2 years",
                    trigger_type=RetentionTriggerType.ACCESS_BASED,
                    condition={"max_inactive_days": 730},
                    action=RetentionAction.ARCHIVE,
                    priority=30,
                ),
            ],
        ))

        # Importance-based policy
        self.register_policy(RetentionPolicy(
            policy_id="importance_based_retention",
            name="Importance-Based Retention",
            description="Keep high-importance memories longer",
            rules=[
                RetentionRule(
                    rule_id="keep_critical_forever",
                    name="Keep Critical Forever",
                    description="Never delete critical importance memories",
                    trigger_type=RetentionTriggerType.IMPORTANCE_BASED,
                    condition={"min_importance": 0.9},
                    action=RetentionAction.REVIEW,
                    priority=5,
                ),
                RetentionRule(
                    rule_id="archive_low_importance",
                    name="Archive Low Importance",
                    description="Archive low importance memories after 1 year",
                    trigger_type=RetentionTriggerType.COMPOSITE,
                    condition={
                        "max_importance": 0.3,
                        "max_age_days": 365,
                    },
                    action=RetentionAction.ARCHIVE,
                    priority=25,
                ),
            ],
        ))

        # Size-based policy
        self.register_policy(RetentionPolicy(
            policy_id="size_based_retention",
            name="Size-Based Retention",
            description="Enforce storage quotas by archiving old low-importance memories",
            rules=[
                RetentionRule(
                    rule_id="archive_when_near_quota",
                    name="Archive When Near Quota",
                    description="Archive old low-importance memories when storage near limit",
                    trigger_type=RetentionTriggerType.SIZE_BASED,
                    condition={"storage_usage_pct": 80},
                    action=RetentionAction.ARCHIVE,
                    priority=15,
                ),
            ],
        ))

    def register_policy(self, policy: RetentionPolicy) -> None:
        self.policies[policy.policy_id] = policy

    def unregister_policy(self, policy_id: str) -> bool:
        if policy_id in self.policies:
            del self.policies[policy_id]
            return True
        return False

    def get_policy(self, policy_id: str) -> RetentionPolicy | None:
        return self.policies.get(policy_id)

    def list_policies(self, enabled_only: bool = True) -> list[RetentionPolicy]:
        policies = list(self.policies.values())
        if not enabled_only:
            return policies
        return [p for p in policies if p.enabled]

    def evaluate_memory(self, memory: Memory) -> list[RetentionRule]:
        """Find all rules that apply to a memory."""
        applicable = []

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            for rule in policy.rules:
                if not rule.enabled:
                    continue

                if self._matches_applicability(rule, memory):
                    if self._evaluates_true(rule.condition, memory):
                        applicable.append(rule)

        # Sort by priority (lower = higher priority)
        applicable.sort(key=lambda r: r.priority)
        return applicable

    def _matches_applicability(self, rule: RetentionRule, memory: Memory) -> bool:
        applies_to = rule.applies_to

        if "memory_type" in applies_to:
            if memory.memory_type.value not in applies_to["memory_type"]:
                return False

        if "topic" in applies_to:
            if str(memory.topic_id) not in applies_to["topic"]:
                return False

        if "tenant" in applies_to:
            if str(memory.topic_id) not in applies_to["tenant"]:
                return False

        if "min_importance" in applies_to:
            if memory.importance < applies_to["min_importance"]:
                return False

        if "max_importance" in applies_to:
            if memory.importance > applies_to["max_importance"]:
                return False

        return True

    def _evaluates_true(self, condition: dict[str, Any], memory: Memory) -> bool:
        now = datetime.now()

        for key, expected in condition.items():
            if key == "max_age_days":
                age_days = (now - memory.created_at).days
                if age_days < expected:
                    return False

            elif key == "max_inactive_days":
                last_accessed = memory.last_accessed or memory.created_at
                inactive_days = (now - last_accessed).days
                if inactive_days < expected:
                    return False

            elif key == "min_importance":
                if memory.importance < expected:
                    return False

            elif key == "max_importance":
                if memory.importance > expected:
                    return False

            elif key == "min_confidence":
                if memory.confidence < expected:
                    return False

            elif key == "max_age_days":
                age_days = (now - memory.created_at).days
                if age_days < expected:
                    return False

            elif key == "storage_usage_pct":
                # Would need to check global storage
                pass

            elif key == "max_inactive_days":
                last_accessed = memory.last_accessed or memory.created_at
                inactive_days = (now - last_accessed).days
                if inactive_days < expected:
                    return False

        return True

    def create_action_plan(self, memory: Memory, rule: RetentionRule) -> RetentionActionPlan:
        plan_id = str(uuid.uuid4())

        # Calculate when to execute (grace period)
        execute_at = datetime.now() + timedelta(days=rule.grace_period_days)

        plan = RetentionActionPlan(
            plan_id=plan_id,
            memory_id=memory.id,
            rule_id=rule.rule_id,
            action=rule.action,
            scheduled_at=execute_at,
            details={
                "rule_name": rule.name,
                "memory_content_preview": memory.content[:100],
            },
        )

        self.action_plans[plan_id] = plan
        return plan

    def execute_action_plan(self, plan: RetentionActionPlan) -> dict[str, Any]:
        if plan.status != "pending":
            return {"success": False, "error": f"Plan status is {plan.status}"}

        memory = self.store.get_memory(plan.memory_id)
        if not memory:
            return {"success": False, "error": "Memory not found"}

        try:
            if plan.action == RetentionAction.ARCHIVE:
                return self._archive_memory(plan, memory)
            elif plan.action == RetentionAction.COMPRESS:
                return self._compress_memory(plan, memory)
            elif plan.action == RetentionAction.DELETE:
                return self._delete_memory(plan, memory)
            elif plan.action == RetentionAction.ANONYMIZE:
                return self._anonymize_memory(plan, memory)
            elif plan.action == RetentionAction.NOTIFY:
                return self._notify_stakeholders(plan, memory)
            elif plan.action == RetentionAction.REVIEW:
                return self._schedule_review(plan, memory)

            return {"success": False, "error": f"Unknown action: {plan.action}"}

        except Exception as e:
            plan.status = "failed"
            plan.result = {"error": str(e)}
            return {"success": False, "error": str(e)}

    def _archive_memory(self, plan: RetentionActionPlan, memory) -> dict[str, Any]:
        original_status = memory.status
        memory.status = MemoryStatus.ARCHIVED
        memory.updated_at = datetime.now()
        self.store.update_memory(memory)

        plan.status = "completed"
        plan.executed_at = datetime.now()
        plan.result = {"previous_status": original_status.value, "new_status": "archived"}

        return {"success": True, "action": "archived"}

    def _compress_memory(self, plan: RetentionActionPlan, memory) -> dict[str, Any]:
        if not self.compressor:
            return {"success": False, "error": "Compressor not available"}

        len(memory.content) // 3
        compressed_content, meta = self.compressor.compress_long_term(memory.content, {})

        # Store version
        from artificial_memory.core.models import MemoryVersion
        version = MemoryVersion(
            memory_id=memory.id,
            resolution=ResolutionLevel.LONG_TERM,
            content=compressed_content,
            compression_ratio=meta.get("compression_ratio"),
            created_at=datetime.now(),
            source="retention_compression",
        )
        self.store.add_memory_version(version)

        # Update memory
        memory.content = compressed_content
        memory.resolution = ResolutionLevel.LONG_TERM
        memory.updated_at = datetime.now()
        self.store.update_memory(memory)

        plan.status = "completed"
        plan.executed_at = datetime.now()
        plan.result = {"compression_ratio": meta.get("compression_ratio")}

        return {"success": True, "action": "compressed", "ratio": meta.get("compression_ratio")}

    def _delete_memory(self, plan: RetentionActionPlan, memory) -> dict[str, Any]:
        self.store.delete_memory(memory.id)

        plan.status = "completed"
        plan.executed_at = datetime.now()
        plan.result = {"deleted": True}

        return {"success": True, "action": "deleted"}

    def _anonymize_memory(self, plan: RetentionActionPlan, memory) -> dict[str, Any]:
        # Replace sensitive info with placeholders
        import re
        content = memory.content

        # Simple PII patterns
        content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', content)
        content = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]', content)
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
        content = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', content)

        memory.content = content
        memory.updated_at = datetime.now()
        self.store.update_memory(memory)

        plan.status = "completed"
        plan.executed_at = datetime.now()
        plan.result = {"anonymized": True}

        return {"success": True, "action": "anonymized"}

    def _notify_stakeholders(self, plan: RetentionActionPlan, memory) -> dict[str, Any]:
        # In production, would send notifications
        plan.status = "completed"
        plan.executed_at = datetime.now()
        plan.result = {"notified": True}

        return {"success": True, "action": "notified"}

    def _schedule_review(self, plan: RetentionActionPlan, memory) -> dict[str, Any]:
        plan.status = "completed"
        plan.executed_at = datetime.now()
        plan.result = {"review_scheduled": True, "review_date": (datetime.now() + timedelta(days=30)).isoformat()}

        return {"success": True, "action": "review_scheduled"}

    def run_retention_cycle(self, topic_id: int | None = None, limit: int = 1000) -> dict[str, Any]:
        """Run a full retention cycle."""
        memories = self.store.get_memories(topic_id=topic_id, limit=limit)

        results = {
            "evaluated": 0,
            "plans_created": 0,
            "actions_executed": 0,
            "errors": 0,
        }

        for memory in memories:
            results["evaluated"] += 1

            applicable_rules = self.evaluate_memory(memory)
            if not applicable_rules:
                continue

            # Apply highest priority rule
            rule = applicable_rules[0]
            plan = self.create_action_plan(memory, rule)

            if rule.grace_period_days == 0:
                result = self.execute_action_plan(plan)
                if result.get("success"):
                    results["actions_executed"] += 1
            else:
                results["plans_created"] += 1

        return results

    def execute_pending_plans(self) -> dict[str, int]:
        executed = 0
        failed = 0

        for plan_id, plan in list(self.action_plans.items()):
            if plan.status != "pending":
                continue

            if plan.scheduled_at <= datetime.now():
                result = self.execute_action_plan(plan)
                if result.get("success"):
                    executed += 1
                else:
                    failed += 1

        return {"executed": executed, "failed": failed, "remaining": len(self.action_plans)}

    def get_pending_plans(self, limit: int = 100) -> list[dict[str, Any]]:
        plans = [p for p in self.action_plans.values() if p.status == "pending"]
        plans.sort(key=lambda p: p.scheduled_at)
        return [self._plan_to_dict(p) for p in plans[:limit]]

    def _plan_to_dict(self, plan) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "memory_id": plan.memory_id,
            "rule_id": plan.rule_id,
            "action": plan.action.value,
            "scheduled_at": plan.scheduled_at.isoformat(),
            "status": plan.status,
            "details": plan.details,
            "created_at": plan.created_at.isoformat(),
        }

    def get_execution_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.execution_history[-limit:]

    def save_state(self, filepath: str):
        state = {
            "policies": {k: self._policy_to_dict(v) for k, v in self.policies.items()},
            "action_plans": {k: self._plan_to_dict(v) for k, v in self.action_plans.items()},
            "execution_history": self.execution_history[-1000:],
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _policy_to_dict(self, policy) -> dict[str, Any]:
        return {
            "policy_id": policy.policy_id,
            "name": policy.name,
            "description": policy.description,
            "rules": [self._rule_to_dict(r) for r in policy.rules],
            "default_action": policy.default_action,
            "enabled": policy.enabled,
            "created_at": policy.created_at.isoformat(),
            "updated_at": policy.updated_at.isoformat(),
            "version": policy.version,
        }

    def _rule_to_dict(self, rule) -> dict[str, Any]:
        return {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "description": rule.description,
            "trigger_type": rule.trigger_type.value,
            "condition": rule.condition,
            "action": rule.action.value,
            "priority": rule.priority,
            "enabled": rule.enabled,
            "grace_period_days": rule.grace_period_days,
            "applies_to": rule.applies_to,
        }


def create_retention_policy_engine(
    store,
    compressor=None,
    consolidation_engine=None,
) -> RetentionPolicyEngine:
    return RetentionPolicyEngine(store, compressor, consolidation_engine)
