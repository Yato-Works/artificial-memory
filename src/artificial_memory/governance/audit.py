from __future__ import annotations

import gzip
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any


class AuditEventType(StrEnum):
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_ACCESSED = "memory_accessed"
    MEMORY_RECALLED = "memory_recalled"
    MEMORY_COMPRESSED = "memory_compressed"
    MEMORY_EXPANDED = "memory_expanded"
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"
    MESSAGE_LOGGED = "message_logged"
    RECALL_PERFORMED = "recall_performed"
    RECALL_EXPANDED = "recall_expanded"
    CONTEXT_BUILT = "context_built"
    FEDERATION_EXCHANGE_REQUESTED = "federation_exchange_requested"
    FEDERATION_EXCHANGE_APPROVED = "federation_exchange_approved"
    FEDERATION_EXCHANGE_REJECTED = "federation_exchange_rejected"
    FEDERATION_PACKAGE_SENT = "federation_package_sent"
    FEDERATION_PACKAGE_RECEIVED = "federation_package_received"
    BELIEF_CREATED = "belief_created"
    BELIEF_UPDATED = "belief_updated"
    BELIEF_SUPERSEDED = "belief_superseded"
    CONTRADICTION_DETECTED = "contradiction_detected"
    CONTRADICTION_RESOLVED = "contradiction_resolved"
    DECISION_MADE = "decision_made"
    DECISION_REVISED = "decision_revised"
    CONSOLIDATION_RUN = "consolidation_run"
    MEMORY_ARCHIVED = "memory_archived"
    MEMORY_REACTIVATED = "memory_reactivated"
    POLICY_VIOLATION = "policy_violation"
    POLICY_VIOLATION_RESOLVED = "policy_violation_resolved"
    TENANT_CREATED = "tenant_created"
    TENANT_UPDATED = "tenant_updated"
    TENANT_DELETED = "tenant_deleted"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    ADMIN_ACTION = "admin_action"
    SYSTEM_ERROR = "system_error"
    SECURITY_EVENT = "security_event"


class AuditSeverity(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    action: str = ""
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None
    trace_id: str | None = None
    span_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "description": self.description,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }


class AuditLogger:
    """Comprehensive audit logging with structured events, filtering, and export."""

    def __init__(
        self,
        store,
        output_dir: str = "audit_logs",
        max_file_size_mb: int = 100,
        max_files: int = 100,
        compression_enabled: bool = True,
        flush_interval_seconds: int = 60,
    ):
        self.store = store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_files = max_files
        self.compression_enabled = compression_enabled
        self.flush_interval = flush_interval_seconds

        self.current_file: Path | None = None
        self.current_file_size = 0
        self.buffer: list[AuditEvent] = []
        self.buffer_lock = Lock()
        self._init_current_file()

        # Index for fast querying
        self.index: dict[str, list[str]] = defaultdict(list)
        self.event_buffer: list[AuditEvent] = []

        # Stats
        self.stats = {
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "events_by_severity": defaultdict(int),
            "events_by_tenant": defaultdict(int),
            "bytes_written": 0,
        }

    def _init_current_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = self.output_dir / f"audit_{timestamp}.jsonl"
        self.current_file_size = 0

    def log_event(
        self,
        event_type: AuditEventType,
        severity: str = "info",
        user_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        action: str = "",
        description: str = "",
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> str:
        """Log an audit event."""

        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=AuditSeverity(severity),
            user_id=user_id,
            tenant_id=tenant_id,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            description=description,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            trace_id=trace_id,
            span_id=span_id,
        )

        with self.buffer_lock:
            self.buffer.append(event)
            self.event_buffer.append(event)
            self.stats["total_events"] += 1
            self.stats["events_by_type"][event_type.value] += 1
            self.stats["events_by_severity"][severity] += 1
            if tenant_id:
                self.stats["events_by_tenant"][tenant_id] += 1

        if len(self.buffer) >= 100:
            self.flush()

        return event.event_id

    def flush(self):
        with self.buffer_lock:
            if not self.buffer:
                return

            events_to_write = self.buffer[:]
            self.buffer.clear()

        self._write_events(events_to_write)

    def _write_events(self, events: list[AuditEvent]):
        if not events:
            return

        if self.current_file_size >= self.max_file_size_bytes:
            self._rotate_file()

        lines = []
        for event in events:
            line = json.dumps(event.to_dict(), separators=(',', ':'))
            lines.append(line)

        content = "\n".join(lines) + "\n"
        content_bytes = content.encode('utf-8')

        if self.compression_enabled:
            content_bytes = gzip.compress(content_bytes)

        with open(self.current_file, 'ab') as f:
            f.write(content_bytes)

        self.current_file_size += len(content_bytes)
        self.stats["bytes_written"] += len(content_bytes)

        for event in events:
            self.index[event.event_type.value].append(event.event_id)

    def _rotate_file(self):
        if self.current_file.exists() and not self.current_file.suffix == '.gz':
            gz_path = self.current_file.with_suffix('.jsonl.gz')
            with open(self.current_file, 'rb') as f_in:
                with gzip.open(gz_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            self.current_file.unlink(missing_ok=True)

        self._init_current_file()

        self._cleanup_old_files()

    def _cleanup_old_files(self):
        files = sorted(self.output_dir.glob("audit_*.jsonl*"), key=lambda f: f.stat().st_mtime)
        while len(files) > self.max_files:
            oldest = files.pop(0)
            oldest.unlink(missing_ok=True)

    def query_events(
        self,
        event_types: list[str] | None = None,
        severity: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        results = []

        for event in reversed(self.event_buffer):
            if len(results) >= 1000:
                break
            if not self._matches_filter(event, event_types, severity, tenant_id, user_id, start_time, end_time):
                continue
            results.append(event.to_dict())

        return results[:limit]

    def _matches_filter(
        self,
        event,
        event_types: list[str] | None,
        severity: str | None,
        tenant_id: str | None,
        user_id: str | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> bool:
        if event_types and event.event_type.value not in event_types:
            return False
        if severity and event.severity.value != severity:
            return False
        if tenant_id and event.tenant_id != tenant_id:
            return False
        if user_id and event.user_id != user_id:
            return False
        if start_time and event.timestamp < start_time:
            return False
        if end_time and event.timestamp > end_time:
            return False
        return True

    def get_stats(self) -> dict[str, Any]:
        return {
            **self.stats,
            "buffer_size": len(self.buffer),
            "event_buffer_size": len(self.event_buffer),
            "index_size": sum(len(v) for v in self.index.values()),
        }

    def export_events(
        self,
        output_path: str,
        event_types: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> str:
        events = self.query_events(
            event_types=event_types,
            start_time=start_time,
            end_time=end_time,
            limit=1000000,
        )

        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            for event in events:
                f.write(json.dumps(event.to_dict()) + '\n')

        return str(output_path)

    def get_recent_events(
        self,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        events = []
        for event in reversed(self.event_buffer):
            if event_type and event.event_type.value != event_type:
                continue
            events.append(event.to_dict())
            if len(events) >= limit:
                break
        return events

    def shutdown(self):
        self.flush()


def create_audit_logger(
    store,
    output_dir: str = "audit_logs",
    max_file_size_mb: int = 100,
    max_files: int = 100,
) -> AuditLogger:
    from .audit import AuditLogger
    return AuditLogger(store, output_dir, max_file_size_mb, max_files)
