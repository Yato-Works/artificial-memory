from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class PolicyDomain(StrEnum):
    MEMORY = "memory"
    RECALL = "recall"
    COMPRESSION = "compression"
    FEDERATION = "federation"
    RETENTION = "retention"
    SECURITY = "security"
    AUDIT = "audit"


class PolicyAction(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    LOG = "log"
    ALERT = "alert"
    QUARANTINE = "quarantine"


class ViolationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class GovernanceRule:
    rule_id: str
    name: str
    description: str
    domain: PolicyDomain
    condition: dict[str, Any]
    action: PolicyAction
    severity: ViolationSeverity = ViolationSeverity.WARNING
    priority: int = 100
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyViolation:
    violation_id: str
    rule_id: str
    rule_name: str
    domain: PolicyDomain
    severity: ViolationSeverity
    memory_id: int | None = None
    topic_id: int | None = None
    tenant_id: str | None = None
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: datetime | None = None
    resolution: str = ""


@dataclass
class GovernancePolicy:
    policy_id: str
    name: str
    description: str
    domain: PolicyDomain
    rules: list[str] = field(default_factory=list)  # Rule IDs
    enabled: bool = True
    priority: int = 100
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class GovernanceEngine:
    """Central governance engine for enforcing policies across all domains."""

    def __init__(
        self,
        store,
        trust_engine=None,
        tenancy_manager=None,
    ):
        self.store = store
        self.trust_engine = trust_engine
        self.tenancy_manager = tenancy_manager

        self.rules: dict[str, GovernanceRule] = {}
        self.policies: dict[str, GovernancePolicy] = {}
        self.violations: list[PolicyViolation] = []

        self._register_default_rules()

    def _register_default_rules(self):
        # Memory protection rules
        self.register_rule(GovernanceRule(
            rule_id="memory_max_size",
            name="Maximum Memory Size",
            description="Prevent individual memories from exceeding size limit",
            domain=PolicyDomain.MEMORY,
            condition={"max_size_mb": 10},
            action=PolicyAction.DENY,
            severity=ViolationSeverity.ERROR,
            priority=10,
        ))

        self.register_rule(GovernanceRule(
            rule_id="memory_confidence_minimum",
            name="Minimum Confidence Threshold",
            description="Reject memories with confidence below threshold",
            domain=PolicyDomain.MEMORY,
            condition={"min_confidence": 0.3},
            action=PolicyAction.REQUIRE_APPROVAL,
            severity=ViolationSeverity.WARNING,
            priority=20,
        ))

        # Recall protection rules
        self.register_rule(GovernanceRule(
            rule_id="recall_token_limit",
            name="Recall Token Limit",
            description="Enforce token budget on recall operations",
            domain=PolicyDomain.RECALL,
            condition={"max_tokens": 8000},
            action=PolicyAction.DENY,
            severity=ViolationSeverity.ERROR,
            priority=10,
        ))

        # Compression rules
        self.register_rule(GovernanceRule(
            rule_id="compression_ratio_limit",
            name="Maximum Compression Ratio",
            description="Prevent over-compression that loses meaning",
            domain=PolicyDomain.COMPRESSION,
            condition={"max_ratio": 20.0},
            action=PolicyAction.REQUIRE_APPROVAL,
            severity=ViolationSeverity.WARNING,
            priority=30,
        ))

        # Federation rules
        self.register_rule(GovernanceRule(
            rule_id="federation_trust_minimum",
            name="Minimum Federation Trust",
            description="Require minimum trust level for federation",
            domain=PolicyDomain.FEDERATION,
            condition={"min_trust_level": "low"},
            action=PolicyAction.DENY,
            severity=ViolationSeverity.ERROR,
            priority=10,
        ))

        # Retention rules
        self.register_rule(GovernanceRule(
            rule_id="retention_max_age",
            name="Maximum Retention Age",
            description="Archive memories older than retention period",
            domain=PolicyDomain.RETENTION,
            condition={"max_age_days": 2555},  # 7 years
            action=PolicyAction.LOG,
            severity=ViolationSeverity.INFO,
            priority=50,
        ))

        # Security rules
        self.register_rule(GovernanceRule(
            rule_id="pii_detection",
            name="PII Detection",
            description="Detect potential PII in memories",
            domain=PolicyDomain.SECURITY,
            condition={"patterns": ["ssn", "credit_card", "api_key", "password"]},
            action=PolicyAction.QUARANTINE,
            severity=ViolationSeverity.CRITICAL,
            priority=5,
        ))

        self.register_rule(GovernanceRule(
            rule_id="injection_attempt",
            name="Injection Attempt Detection",
            description="Detect prompt injection or manipulation attempts",
            domain=PolicyDomain.SECURITY,
            condition={"patterns": ["ignore previous", "system prompt", "override"]},
            action=PolicyAction.QUARANTINE,
            severity=ViolationSeverity.CRITICAL,
            priority=5,
        ))

    def register_rule(self, rule: GovernanceRule) -> None:
        self.rules[rule.rule_id] = rule

    def unregister_rule(self, rule_id: str) -> bool:
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def get_rule(self, rule_id: str):
        return self.rules.get(rule_id)

    def list_rules(self, domain: str | None = None, enabled: bool = True) -> list[GovernanceRule]:
        rules = [r for r in self.rules.values() if r.enabled == enabled]
        if domain:
            rules = [r for r in rules if r.domain.value == domain]
        return sorted(rules, key=lambda r: r.priority)

    def evaluate(self, context: dict[str, Any]) -> list[PolicyViolation]:
        """Evaluate all rules against a context."""
        violations = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            if self._evaluate_condition(rule.condition, context):
                violation = PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    domain=rule.domain,
                    severity=rule.severity,
                    memory_id=context.get("memory_id"),
                    topic_id=context.get("topic_id"),
                    tenant_id=context.get("tenant_id"),
                    description=f"Rule '{rule.name}' triggered: {rule.description}",
                    details={
                        "condition": rule.condition,
                        "action": rule.action.value,
                        "context_keys": list(context.keys()),
                    },
                )
                self.violations.append(violation)
                violations.append(violation)

        return violations

    def _evaluate_condition(self, condition: dict[str, Any], context: dict[str, Any]) -> bool:
        for key, expected in condition.items():
            if key == "max_tokens":
                actual = context.get("tokens", 0)
                if actual > expected:
                    return True
            elif key == "min_confidence":
                actual = context.get("confidence", 1.0)
                if actual < expected:
                    return True
            elif key == "max_size_mb":
                actual = context.get("size_mb", 0)
                if actual > expected:
                    return True
            elif key == "min_trust_level":
                trust_order = {"untrusted": 0, "low": 1, "medium": 2, "high": 3, "implicit": 4}
                actual = trust_order.get(context.get("trust_level", "medium"), 2)
                expected_val = trust_order.get(expected, 2)
                if actual < expected_val:
                    return True
            elif key == "patterns":
                content = context.get("content", "").lower()
                for pattern in expected:
                    if pattern.lower() in content.lower():
                        return True
            elif key == "max_age_days":
                created = context.get("created_at")
                if created:
                    age = (datetime.now() - created).days
                    if age > expected:
                        return True
            elif key == "max_ratio":
                actual = context.get("compression_ratio", 1.0)
                if actual > expected:
                    return True

        return False

    def record_violation(self, violation: PolicyViolation) -> None:
        self.violations.append(violation)

    def get_violations(
        self,
        domain: str | None = None,
        severity: str | None = None,
        resolved: bool | None = None,
        limit: int = 100,
    ) -> list[PolicyViolation]:
        violations = self.violations

        if domain:
            violations = [v for v in violations if v.domain.value == domain]
        if severity:
            violations = [v for v in violations if v.severity.value == severity]
        if resolved is not None:
            violations = [v for v in violations if v.resolved == resolved]

        return sorted(violations, key=lambda v: v.detected_at, reverse=True)[:limit]

    def resolve_violation(self, violation_id: str, resolution: str) -> bool:
        for v in self.violations:
            if v.violation_id == violation_id:
                v.resolved = True
                v.resolved_at = datetime.now()
                v.resolution = resolution
                return True
        return False

    def get_compliance_report(self) -> dict[str, Any]:
        total = len(self.violations)
        by_domain = {}
        by_severity = {}
        by_resolved = {"resolved": 0, "unresolved": 0}

        for v in self.violations:
            domain = v.domain.value
            severity = v.severity.value

            by_domain[domain] = by_domain.get(domain, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

            if v.resolved:
                by_resolved["resolved"] += 1
            else:
                by_resolved["unresolved"] += 1

        return {
            "total_violations": total,
            "by_domain": by_domain,
            "by_severity": by_severity,
            "resolution_status": by_resolved,
            "resolution_rate": by_resolved["resolved"] / total if total > 0 else 0,
        }


def create_governance_engine(
    store,
    trust_engine=None,
    tenancy_manager=None,
) -> GovernanceEngine:
    return GovernanceEngine(store, trust_engine, tenancy_manager)
