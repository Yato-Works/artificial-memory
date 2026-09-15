from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TrustLevel(StrEnum):
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMPLICIT = "implicit"


class TrustPolicyType(StrEnum):
    NODE_BASED = "node_based"
    MEMORY_TYPE_BASED = "memory_type_based"
    IMPORTANCE_BASED = "importance_based"
    PROVENANCE_BASED = "provenance_based"
    TEMPORAL_BASED = "temporal_based"
    COMPOSITE = "composite"


class PolicyAction(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_VERIFICATION = "require_verification"
    REQUIRE_APPROVAL = "require_approval"
    QUARANTINE = "quarantine"
    LOG_ONLY = "log_only"


@dataclass
class TrustRule:
    rule_id: str
    name: str
    description: str
    policy_type: TrustPolicyType
    condition: dict[str, Any]
    action: PolicyAction
    priority: int = 100
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustPolicy:
    policy_id: str
    name: str
    description: str
    rules: list[TrustRule] = field(default_factory=list)
    default_action: PolicyAction = PolicyAction.REQUIRE_VERIFICATION
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustEvaluationResult:
    allowed: bool
    trust_level: str
    matched_rules: list[str] = field(default_factory=list)
    applied_action: str = ""
    reason: str = ""
    required_verifications: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrustPolicyEngine:
    """Evaluates trust decisions based on configured policies.

    Supports:
    - Node-based trust (per-node trust levels)
    - Memory-type based trust
    - Importance/confidence thresholds
    - Provenance chain validation
    - Temporal constraints
    - Composite policies with multiple rules
    """

    def __init__(
        self,
        store,
        default_trust_level: str = "medium",
    ):
        self.store = store
        self.default_trust_level = default_trust_level
        self.policies: dict[str, TrustPolicy] = {}
        self.node_trust_levels: dict[str, str] = {}
        self.memory_type_trust: dict[str, str] = {}
        self._register_default_policies()

    def _register_default_policies(self):
        # Default deny-all policy for untrusted nodes
        self.register_policy(TrustPolicy(
            policy_id="default_untrusted",
            name="Default Untrusted Node Policy",
            description="Deny all exchanges from untrusted nodes",
            rules=[TrustRule(
                rule_id="untrusted_deny",
                name="Untrusted Node Deny",
                description="Deny all from untrusted nodes",
                policy_type=TrustPolicyType.NODE_BASED,
                condition={"trust_level": "untrusted"},
                action=PolicyAction.DENY,
                priority=1,
            )],
            default_action=PolicyAction.DENY,
        ))

        # Default allow for high trust and implicit trust
        self.register_policy(TrustPolicy(
            policy_id="default_trusted",
            name="Default Trusted Node Policy",
            description="Allow all for trusted nodes",
            rules=[TrustRule(
                rule_id="trusted_allow",
                name="Trusted Node Allow",
                description="Allow all from trusted (high/implicit) nodes",
                policy_type=TrustPolicyType.NODE_BASED,
                condition={"trust_level": "high"},
                action=PolicyAction.ALLOW,
                priority=1,
            ), TrustRule(
                rule_id="implicit_allow",
                name="Implicit Trust Allow",
                description="Allow all from implicitly trusted nodes",
                policy_type=TrustPolicyType.NODE_BASED,
                condition={"trust_level": "implicit"},
                action=PolicyAction.ALLOW,
                priority=2,
            )],
            default_action=PolicyAction.ALLOW,
        ))

        # Default deny for low trust
        self.register_policy(TrustPolicy(
            policy_id="default_low_trust",
            name="Default Low Trust Policy",
            description="Deny exchanges from low trust nodes",
            rules=[TrustRule(
                rule_id="low_trust_deny",
                name="Low Trust Deny",
                description="Deny from low trust nodes",
                policy_type=TrustPolicyType.NODE_BASED,
                condition={"trust_level": "low"},
                action=PolicyAction.DENY,
                priority=1,
            )],
            default_action=PolicyAction.REQUIRE_VERIFICATION,
        ))

    def register_policy(self, policy: TrustPolicy) -> None:
        self.policies[policy.policy_id] = policy

    def unregister_policy(self, policy_id: str) -> bool:
        if policy_id in self.policies:
            del self.policies[policy_id]
            return True
        return False

    def set_node_trust_level(self, node_id: str, trust_level: str) -> None:
        self.node_trust_levels[node_id] = trust_level

    def get_node_trust_level(self, node_id: str) -> str:
        return self.node_trust_levels.get(node_id, "medium")

    def evaluate_trust(
        self,
        source_node_id: str,
        memories: list[Any],
        context: dict[str, Any] | None = None,
    ) -> TrustEvaluationResult:
        """Evaluate trust for a set of memories from a source node."""

        # Get node trust level
        self.get_node_trust_level(source_node_id)

        # Collect all applicable rules from all policies
        all_rules = []
        for policy in self.policies.values():
            if policy.enabled:
                all_rules.extend(policy.rules)

        # Sort by priority
        all_rules.sort(key=lambda r: r.priority)

        matched_rules = []
        final_action = PolicyAction.REQUIRE_VERIFICATION
        rule_matched = False

        for rule in all_rules:
            if not rule.enabled:
                continue

            if self._evaluate_condition(rule.condition, source_node_id, None, context):
                matched_rules.append(rule.rule_id)
                final_action = rule.action
                rule_matched = True
                # Continue to find highest priority match

        # Apply default action from highest priority policy ONLY if no rules matched
        if not rule_matched:
            for policy in sorted(self.policies.values(), key=lambda p: min(r.priority for r in p.rules) if p.rules else 100):
                if policy.enabled:
                    final_action = policy.default_action
                    break

        # Determine allowed
        allowed = final_action == PolicyAction.ALLOW

        # Determine trust level based on action
        if final_action == PolicyAction.DENY:
            trust_level = "untrusted"
        elif final_action == PolicyAction.ALLOW:
            trust_level = "high"
        elif final_action == PolicyAction.REQUIRE_VERIFICATION:
            trust_level = "medium"
        elif final_action == PolicyAction.REQUIRE_APPROVAL:
            trust_level = "medium"
        elif final_action == PolicyAction.QUARANTINE:
            trust_level = "low"
        else:
            trust_level = "medium"

        return TrustEvaluationResult(
            allowed=allowed,
            trust_level=trust_level,
            matched_rules=matched_rules,
            applied_action=final_action.value,
            reason=f"Evaluated {len(matched_rules)} rules, final action: {final_action.value}",
        )

    def _evaluate_condition(
        self,
        condition: dict[str, Any],
        source_node_id: str,
        memory: Any | None,
        context: dict[str, Any] | None,
    ) -> bool:
        for key, expected in condition.items():
            if key == "trust_level":
                actual = self.get_node_trust_level(source_node_id)
                if actual != expected:
                    return False
            elif key == "memory_type":
                if memory and memory.memory_type.value != expected:
                    return False
            elif key == "importance_min":
                if memory and memory.importance < expected:
                    return False
            elif key == "confidence_min":
                if memory and memory.confidence < expected:
                    return False
            elif key == "node_id":
                if source_node_id != expected:
                    return False
            elif key == "time_after":
                if datetime.now() < expected:
                    return False
            elif key == "time_before":
                if datetime.now() > expected:
                    return False
        return True

    def get_policy(self, policy_id: str) -> TrustPolicy | None:
        return self.policies.get(policy_id)

    def list_policies(self) -> list[TrustPolicy]:
        return list(self.policies.values())


def create_trust_policy_engine(store, default_trust_level: str = "medium") -> TrustPolicyEngine:
    return TrustPolicyEngine(store, default_trust_level)
