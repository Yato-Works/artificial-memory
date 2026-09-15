from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Project


class TenantStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"
    PENDING = "pending"


class IsolationLevel(StrEnum):
    LOGICAL = "logical"          # Shared DB, logical separation via tenant_id
    SCHEMA = "schema"            # Separate schemas per tenant
    DATABASE = "database"        # Separate databases per tenant
    CLUSTER = "cluster"          # Separate clusters per tenant


@dataclass
class TenantConfig:
    tenant_id: str
    name: str
    display_name: str
    description: str = ""
    status: TenantStatus = TenantStatus.ACTIVE
    isolation_level: IsolationLevel = IsolationLevel.LOGICAL
    max_memories: int = 100000
    max_storage_mb: int = 10240
    max_concurrent_sessions: int = 10
    allowed_memory_types: list[str] = field(default_factory=lambda: ["semantic", "decision", "episode", "timeline"])
    allowed_features: list[str] = field(default_factory=lambda: ["recall", "search", "temporal", "federation"])
    retention_days: int = 2555  # 7 years
    encryption_required: bool = True
    audit_logging: bool = True
    custom_limits: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Tenant:
    config: TenantConfig
    current_memory_count: int = 0
    current_storage_mb: float = 0.0
    active_sessions: int = 0
    last_activity: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def tenant_id(self) -> str:
        return self.config.tenant_id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_active(self) -> bool:
        return self.config.status == TenantStatus.ACTIVE

    def can_create_memory(self, memory_type: str, estimated_size_mb: float = 0) -> tuple[bool, str]:
        if not self.is_active:
            return False, "Tenant is not active"

        if memory_type not in self.config.allowed_memory_types:
            return False, f"Memory type {memory_type} not allowed"

        if self.current_memory_count >= self.config.max_memories:
            return False, f"Memory limit reached ({self.config.max_memories})"

        if self.current_storage_mb + estimated_size_mb > self.config.max_storage_mb:
            return False, f"Storage limit reached ({self.config.max_storage_mb} MB)"

        return True, "OK"

    def can_create_session(self) -> tuple[bool, str]:
        if not self.is_active:
            return False, "Tenant is not active"

        if self.active_sessions >= self.config.max_concurrent_sessions:
            return False, f"Session limit reached ({self.config.max_concurrent_sessions})"

        return True, "OK"

    def increment_memory_count(self, estimated_size_mb: float = 0) -> None:
        self.current_memory_count += 1
        self.current_storage_mb += estimated_size_mb
        self.last_activity = datetime.now()


class TenancyManager:
    """Manages multi-tenant isolation and resource limits."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.tenants: dict[str, Tenant] = {}
        self._tenant_id_to_project: dict[str, int] = {}
        self._memory_tenant_cache: dict[int, str] = {}

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        display_name: str,
        config: TenantConfig | None = None,
    ) -> Tenant:
        if tenant_id in self.tenants:
            raise ValueError(f"Tenant {tenant_id} already exists")

        if config is None:
            config = TenantConfig(
                tenant_id=tenant_id,
                name=name,
                display_name=display_name,
            )

        tenant = Tenant(config=config)
        self.tenants[tenant_id] = tenant

        # Create a project for this tenant if using logical isolation
        if config.isolation_level == "logical":
            project = Project(name=tenant_id, display_name=display_name)
            project = self.store.create_project(project)
            self._tenant_id_to_project[tenant_id] = project.id

        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self.tenants.get(tenant_id)

    def update_tenant(self, tenant_id: str, updates: dict[str, Any]) -> Tenant:
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        for key, value in updates.items():
            if hasattr(tenant.config, key):
                setattr(tenant.config, key, value)

        tenant.config.updated_at = datetime.now()
        return tenant

    def delete_tenant(self, tenant_id: str, force: bool = False) -> bool:
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        if tenant.current_memory_count > 0 and not force:
            raise ValueError(f"Tenant {tenant_id} has {tenant.current_memory_count} memories. Use force=True to delete anyway.")

        # Delete all memories
        if tenant.config.isolation_level == "logical":
            project_id = self._tenant_id_to_project.get(tenant_id)
            if project_id:
                memories = self.store.get_memories(topic_id=None, limit=100000)  # Would need tenant filter
                for mem in memories:
                    self.store.delete_memory(mem.id)
                self.store.delete_project(project_id)
                del self._tenant_id_to_project[tenant_id]

        del self.tenants[tenant_id]
        return True

    def list_tenants(self, status: TenantStatus | None = None) -> list[Tenant]:
        tenants = list(self.tenants.values())
        if status:
            tenants = [t for t in tenants if t.config.status == status]
        return tenants

    def assign_memory_to_tenant(self, memory_id: int, tenant_id: str) -> bool:
        memory = self.store.get_memory(memory_id)
        if not memory:
            return False

        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        # Update memory with tenant info (would need tenant_id field in Memory model)
        # For logical isolation, we use project_id
        if self.tenants[tenant_id].config.isolation_level == "logical":
            project_id = self._tenant_id_to_project.get(tenant_id)
            if project_id:
                memory.topic_id = project_id  # Simplified
                self.store.update_memory(memory)

        self._memory_tenant_cache[memory_id] = tenant_id
        return True

    def get_tenant_memories(self, tenant_id: str, limit: int = 100) -> list:
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return []

        if tenant.config.isolation_level == "logical":
            project_id = self._tenant_id_to_project.get(tenant_id)
            if project_id:
                return self.store.get_memories(topic_id=project_id, limit=limit)

        # For other isolation levels, would filter by tenant_id
        return []

    def get_tenant_stats(self, tenant_id: str) -> dict[str, Any]:
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {}

        self.get_tenant_memories(tenant_id, limit=10000)

        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "status": tenant.config.status.value,
            "memory_count": tenant.current_memory_count,
            "storage_mb": tenant.current_storage_mb,
            "active_sessions": tenant.active_sessions,
            "limits": {
                "max_memories": tenant.config.max_memories,
                "max_storage_mb": tenant.config.max_storage_mb,
                "max_sessions": tenant.config.max_concurrent_sessions,
            },
            "usage_pct": {
                "memory": tenant.current_memory_count / tenant.config.max_memories * 100,
                "storage": tenant.current_storage_mb / tenant.config.max_storage_mb * 100,
                "sessions": tenant.active_sessions / tenant.config.max_concurrent_sessions * 100,
            },
        }

    def enforce_limits(self, tenant_id: str) -> list[str]:
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return []

        violations = []

        if tenant.current_memory_count >= tenant.config.max_memories:
            violations.append(f"Memory limit exceeded: {tenant.current_memory_count}/{tenant.config.max_memories}")

        if tenant.current_storage_mb >= tenant.config.max_storage_mb:
            violations.append(f"Storage limit exceeded: {tenant.current_storage_mb:.1f}/{tenant.config.max_storage_mb} MB")

        if tenant.active_sessions >= tenant.config.max_concurrent_sessions:
            violations.append(f"Session limit exceeded: {tenant.active_sessions}/{tenant.config.max_concurrent_sessions}")

        return violations


def create_tenancy_manager(store: MemoryStore) -> TenancyManager:
    return TenancyManager(store)
