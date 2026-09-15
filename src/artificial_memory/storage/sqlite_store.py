from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from artificial_memory.core.models import (
    Association,
    AssociationType,
    BeliefState,
    CompressionEvent,
    CompressionMethod,
    ContextIR,
    Conversation,
    ConversationStatus,
    Decision,
    EvolutionEventRecord,
    IRType,
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    Message,
    MessageRole,
    Project,
    RecallEvent,
    RecallLevel,
    ResolutionLevel,
    TokenUsage,
    Topic,
)


class SQLiteMemoryStore:
    """SQLite implementation of MemoryStore."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._apply_schema_if_needed()

    def _apply_schema_if_needed(self) -> None:
        # Check if tables already exist
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='projects'"
        )
        if cursor.fetchone() is None:
            self._apply_schema()
        else:
            self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Lightweight forward migrations for existing databases.

        - Creates belief_states / evolution_events tables if missing (P0-4, P0-5)
        - Adds policy_version / algorithm_version to memory_versions (P0-7)
        """
        with self._transaction() as conn:
            # New tables (safe if they already exist)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS belief_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    belief_key TEXT NOT NULL UNIQUE,
                    proposition TEXT NOT NULL,
                    status TEXT DEFAULT 'accepted',
                    confidence REAL DEFAULT 0.5,
                    supporting_evidence TEXT,
                    contradicting_evidence TEXT,
                    source_memories TEXT,
                    valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    valid_until TIMESTAMP,
                    revision INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_belief_states_key ON belief_states(belief_key);
                CREATE INDEX IF NOT EXISTS idx_belief_states_status ON belief_states(status);

                CREATE TABLE IF NOT EXISTS evolution_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    operation TEXT NOT NULL,
                    source_memory_ids TEXT,
                    target_memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL,
                    description TEXT,
                    old_content TEXT,
                    new_content TEXT,
                    metadata_json TEXT,
                    triggered_by TEXT DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_evolution_events_memory ON evolution_events(memory_id);
                CREATE INDEX IF NOT EXISTS idx_evolution_events_operation ON evolution_events(operation);
                CREATE INDEX IF NOT EXISTS idx_evolution_events_created ON evolution_events(created_at);
            """)

            # Add new columns to memory_versions if missing
            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(memory_versions)").fetchall()
            }
            if "policy_version" not in existing_columns:
                conn.execute(
                    "ALTER TABLE memory_versions ADD COLUMN policy_version TEXT DEFAULT 'decision-v1'"
                )
            if "algorithm_version" not in existing_columns:
                conn.execute(
                    "ALTER TABLE memory_versions ADD COLUMN algorithm_version TEXT DEFAULT 'rule-compressor-v1'"
                )

    def _apply_schema(self) -> None:
        schema_path = Path(__file__).parent.parent.parent.parent / "scripts" / "schema.sql"
        if schema_path.exists():
            with open(schema_path) as f:
                self._conn.executescript(f.read())
        else:
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

    @contextmanager
    def _transaction(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _row_to_project(self, row: sqlite3.Row) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            display_name=row["display_name"],
            description=row["description"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            is_active=bool(row["is_active"]),
        )

    def _row_to_topic(self, row: sqlite3.Row) -> Topic:
        return Topic(
            id=row["id"],
            project_id=row["project_id"],
            parent_id=row["parent_id"],
            name=row["name"],
            path=row["path"],
            description=row["description"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_conversation(self, row: sqlite3.Row) -> Conversation:
        metadata = {}
        if row["metadata_json"]:
            try:
                metadata = yaml.safe_load(row["metadata_json"]) or {}
            except Exception:
                pass
        return Conversation(
            id=row["id"],
            topic_id=row["topic_id"],
            title=row["title"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            message_count=row["message_count"],
            token_count=row["token_count"],
            metadata=metadata,
            status=ConversationStatus(row["status"]),
        )

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        metadata = {}
        if row["metadata_json"]:
            try:
                metadata = yaml.safe_load(row["metadata_json"]) or {}
            except Exception:
                pass
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            token_count=row["token_count"],
            sequence_num=row["sequence_num"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=metadata,
        )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            topic_id=row["topic_id"],
            memory_type=MemoryType(row["memory_type"]),
            content=row["content"],
            resolution=ResolutionLevel(row["resolution"]),
            importance=row["importance"],
            confidence=row["confidence"],
            status=MemoryStatus(row["status"]),
            valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            is_current=bool(row["is_current"]),
            source_conversation_id=row["source_conversation_id"],
            source_message_id=row["source_message_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            access_count=row["access_count"],
        )

    def _row_to_memory_version(self, row: sqlite3.Row) -> MemoryVersion:
        return MemoryVersion(
            id=row["id"],
            memory_id=row["memory_id"],
            resolution=ResolutionLevel(row["resolution"]),
            content=row["content"],
            compression_ratio=row["compression_ratio"],
            created_at=datetime.fromisoformat(row["created_at"]),
            source=row["source"],
            policy_version=row["policy_version"] if "policy_version" in row.keys() else "decision-v1",
            algorithm_version=row["algorithm_version"] if "algorithm_version" in row.keys() else "rule-compressor-v1",
        )

    def _row_to_decision(self, row: sqlite3.Row) -> Decision:
        rejected = []
        if row["rejected_options"]:
            try:
                rejected = json.loads(row["rejected_options"])
            except Exception:
                pass
        return Decision(
            id=row["id"],
            topic_id=row["topic_id"],
            memory_id=row["memory_id"],
            decision_text=row["decision_text"],
            rejected_options=rejected,
            reason=row["reason"],
            confidence=row["confidence"],
            decided_at=datetime.fromisoformat(row["decided_at"]),
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            is_current=bool(row["is_current"]),
        )

    def _row_to_association(self, row: sqlite3.Row) -> Association:
        return Association(
            id=row["id"],
            source_memory_id=row["source_memory_id"],
            target_memory_id=row["target_memory_id"],
            association_type=AssociationType(row["association_type"]),
            strength=row["strength"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_recall(self, row: sqlite3.Row) -> RecallEvent:
        return RecallEvent(
            id=row["id"],
            query=row["query"],
            topic_id=row["topic_id"],
            recall_level=RecallLevel(row["recall_level"]),
            memories_retrieved=row["memories_retrieved"],
            tokens_returned=row["tokens_returned"],
            latency_ms=row["latency_ms"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_compression(self, row: sqlite3.Row) -> CompressionEvent:
        return CompressionEvent(
            id=row["id"],
            source_memory_id=row["source_memory_id"],
            target_memory_id=row["target_memory_id"],
            from_resolution=ResolutionLevel(row["from_resolution"]),
            to_resolution=ResolutionLevel(row["to_resolution"]),
            original_tokens=row["original_tokens"],
            compressed_tokens=row["compressed_tokens"],
            compression_ratio=row["compression_ratio"],
            method=CompressionMethod(row["method"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_context_ir(self, row: sqlite3.Row) -> ContextIR:
        return ContextIR(
            id=row["id"],
            conversation_id=row["conversation_id"],
            ir_type=IRType(row["ir_type"]),
            ir_key=row["ir_key"],
            ir_value=row["ir_value"],
            sequence_num=row["sequence_num"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_token_usage(self, row: sqlite3.Row) -> TokenUsage:
        return TokenUsage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            operation=row["operation"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            total_tokens=row["total_tokens"],
            model=row["model"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # Project operations
    def create_project(self, project: Project) -> Project:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO projects (name, display_name, description, created_at, updated_at, is_active)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (project.name, project.display_name, project.description,
                 project.created_at.isoformat(), project.updated_at.isoformat(),
                 int(project.is_active))
            )
            project.id = cursor.lastrowid
        return project

    def get_project(self, project_id: int) -> Project | None:
        row = self._conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._row_to_project(row) if row else None

    def get_project_by_name(self, name: str) -> Project | None:
        row = self._conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self, active_only: bool = True) -> list[Project]:
        query = "SELECT * FROM projects"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY name"
        rows = self._conn.execute(query).fetchall()
        return [self._row_to_project(r) for r in rows]

    def update_project(self, project: Project) -> Project:
        project.updated_at = datetime.now()
        with self._transaction() as conn:
            conn.execute(
                """UPDATE projects SET display_name = ?, description = ?, updated_at = ?, is_active = ?
                   WHERE id = ?""",
                (project.display_name, project.description, project.updated_at.isoformat(),
                 int(project.is_active), project.id)
            )
        return project

    # Topic operations
    def create_topic(self, topic: Topic) -> Topic:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO topics (project_id, parent_id, name, path, description, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (topic.project_id, topic.parent_id, topic.name, topic.path,
                 topic.description, topic.created_at.isoformat(), topic.updated_at.isoformat())
            )
            topic.id = cursor.lastrowid
        return topic

    def get_topic(self, topic_id: int) -> Topic | None:
        row = self._conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        return self._row_to_topic(row) if row else None

    def get_topic_by_path(self, path: str) -> Topic | None:
        row = self._conn.execute("SELECT * FROM topics WHERE path = ?", (path,)).fetchone()
        return self._row_to_topic(row) if row else None

    def list_topics(self, project_id: int | None = None, parent_id: int | None = None) -> list[Topic]:
        query = "SELECT * FROM topics WHERE 1=1"
        params = []
        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)
        if parent_id is not None:
            query += " AND parent_id = ?"
            params.append(parent_id)
        query += " ORDER BY path"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_topic(r) for r in rows]

    def update_topic(self, topic: Topic) -> Topic:
        topic.updated_at = datetime.now()
        with self._transaction() as conn:
            conn.execute(
                """UPDATE topics SET parent_id = ?, name = ?, path = ?, description = ?, updated_at = ?
                   WHERE id = ?""",
                (topic.parent_id, topic.name, topic.path, topic.description,
                 topic.updated_at.isoformat(), topic.id)
            )
        return topic

    # Conversation operations
    def create_conversation(self, conversation: Conversation) -> Conversation:
        metadata_json = yaml.dump(conversation.metadata) if conversation.metadata else None
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO conversations (topic_id, title, started_at, ended_at, message_count, token_count, metadata_json, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (conversation.topic_id, conversation.title, conversation.started_at.isoformat(),
                 conversation.ended_at.isoformat() if conversation.ended_at else None,
                 conversation.message_count, conversation.token_count,
                 metadata_json, conversation.status.value)
            )
            conversation.id = cursor.lastrowid
        return conversation

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        row = self._conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return self._row_to_conversation(row) if row else None

    def list_conversations(self, topic_id: int | None = None, status: ConversationStatus | None = None) -> list[Conversation]:
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        if topic_id is not None:
            query += " AND topic_id = ?"
            params.append(topic_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY started_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_conversation(r) for r in rows]

    def update_conversation(self, conversation: Conversation) -> Conversation:
        metadata_json = yaml.dump(conversation.metadata) if conversation.metadata else None
        with self._transaction() as conn:
            conn.execute(
                """UPDATE conversations SET title = ?, ended_at = ?, message_count = ?, token_count = ?, metadata_json = ?, status = ?
                   WHERE id = ?""",
                (conversation.title, conversation.ended_at.isoformat() if conversation.ended_at else None,
                 conversation.message_count, conversation.token_count, metadata_json,
                 conversation.status.value, conversation.id)
            )
        return conversation

    def end_conversation(self, conversation_id: int) -> Conversation:
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.ended_at = datetime.now()
            conversation.status = ConversationStatus.COMPLETED
            return self.update_conversation(conversation)
        raise ValueError(f"Conversation {conversation_id} not found")

    # Message operations
    def add_message(self, message: Message) -> Message:
        metadata_json = yaml.dump(message.metadata) if message.metadata else None
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO messages (conversation_id, role, content, token_count, sequence_num, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (message.conversation_id, message.role.value, message.content,
                 message.token_count, message.sequence_num, message.created_at.isoformat(), metadata_json)
            )
            message.id = cursor.lastrowid

            # Update conversation message count
            conn.execute(
                "UPDATE conversations SET message_count = message_count + 1 WHERE id = ?",
                (message.conversation_id,)
            )
        return message

    def get_messages(self, conversation_id: int, limit: int | None = None, offset: int = 0) -> list[Message]:
        query = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence_num"
        params = [conversation_id]
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_message(r) for r in rows]

    def get_message_count(self, conversation_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return row["cnt"] if row else 0

    # Memory operations
    def create_memory(self, memory: Memory) -> Memory:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO memories (topic_id, memory_type, content, resolution, importance, confidence, status,
                   valid_from, valid_until, is_current, source_conversation_id, source_message_id, created_at, updated_at, last_accessed, access_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (memory.topic_id, memory.memory_type.value, memory.content, memory.resolution.value,
                 memory.importance, memory.confidence, memory.status.value,
                 memory.valid_from.isoformat() if memory.valid_from else None,
                 memory.valid_until.isoformat() if memory.valid_until else None,
                 int(memory.is_current), memory.source_conversation_id, memory.source_message_id,
                 memory.created_at.isoformat(), memory.updated_at.isoformat(),
                 memory.last_accessed.isoformat() if memory.last_accessed else None,
                 memory.access_count)
            )
            memory.id = cursor.lastrowid
        return memory

    def get_memory(self, memory_id: int) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def get_memories(
        self,
        topic_id: int | None = None,
        memory_type: MemoryType | None = None,
        resolution: ResolutionLevel | None = None,
        status: MemoryStatus | None = None,
        is_current: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        if topic_id is not None:
            query += " AND topic_id = ?"
            params.append(topic_id)
        if memory_type is not None:
            query += " AND memory_type = ?"
            params.append(memory_type.value)
        if resolution is not None:
            query += " AND resolution = ?"
            params.append(resolution.value)
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        if is_current is not None:
            query += " AND is_current = ?"
            params.append(int(is_current))
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def update_memory(self, memory: Memory) -> Memory:
        memory.updated_at = datetime.now()
        with self._transaction() as conn:
            conn.execute(
                """UPDATE memories SET topic_id = ?, memory_type = ?, content = ?, resolution = ?, importance = ?,
                   confidence = ?, status = ?, valid_from = ?, valid_until = ?, is_current = ?,
                   source_conversation_id = ?, source_message_id = ?, updated_at = ?, last_accessed = ?, access_count = ?
                   WHERE id = ?""",
                (memory.topic_id, memory.memory_type.value, memory.content, memory.resolution.value,
                 memory.importance, memory.confidence, memory.status.value,
                 memory.valid_from.isoformat() if memory.valid_from else None,
                 memory.valid_until.isoformat() if memory.valid_until else None,
                 int(memory.is_current), memory.source_conversation_id, memory.source_message_id,
                 memory.updated_at.isoformat(),
                 memory.last_accessed.isoformat() if memory.last_accessed else None,
                 memory.access_count, memory.id)
            )
        return memory

    def delete_memory(self, memory_id: int) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    # Memory version operations
    def add_memory_version(self, version: MemoryVersion) -> MemoryVersion:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO memory_versions (memory_id, resolution, content, compression_ratio, created_at, source, policy_version, algorithm_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (version.memory_id, version.resolution.value, version.content,
                 version.compression_ratio, version.created_at.isoformat(), version.source,
                 version.policy_version, version.algorithm_version)
            )
            version.id = cursor.lastrowid
        return version

    def get_memory_versions(self, memory_id: int) -> list[MemoryVersion]:
        rows = self._conn.execute(
            "SELECT * FROM memory_versions WHERE memory_id = ? ORDER BY resolution", (memory_id,)
        ).fetchall()
        return [self._row_to_memory_version(r) for r in rows]

    def get_memory_version(self, memory_id: int, resolution: ResolutionLevel) -> MemoryVersion | None:
        row = self._conn.execute(
            "SELECT * FROM memory_versions WHERE memory_id = ? AND resolution = ?",
            (memory_id, resolution.value)
        ).fetchone()
        return self._row_to_memory_version(row) if row else None

    # Decision operations
    def create_decision(self, decision: Decision) -> Decision:
        rejected_json = json.dumps(decision.rejected_options) if decision.rejected_options else None
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO decisions (topic_id, memory_id, decision_text, rejected_options, reason, confidence,
                   decided_at, valid_from, valid_until, is_current)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (decision.topic_id, decision.memory_id, decision.decision_text, rejected_json,
                 decision.reason, decision.confidence, decision.decided_at.isoformat(),
                 decision.valid_from.isoformat(),
                 decision.valid_until.isoformat() if decision.valid_until else None,
                 int(decision.is_current))
            )
            decision.id = cursor.lastrowid
        return decision

    def get_decisions(self, topic_id: int, current_only: bool = True) -> list[Decision]:
        query = "SELECT * FROM decisions WHERE topic_id = ?"
        params = [topic_id]
        if current_only:
            query += " AND is_current = 1"
        query += " ORDER BY decided_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_decision(r) for r in rows]

    def update_decision(self, decision: Decision) -> Decision:
        rejected_json = json.dumps(decision.rejected_options) if decision.rejected_options else None
        with self._transaction() as conn:
            conn.execute(
                """UPDATE decisions SET memory_id = ?, decision_text = ?, rejected_options = ?, reason = ?,
                   confidence = ?, valid_from = ?, valid_until = ?, is_current = ?
                   WHERE id = ?""",
                (decision.memory_id, decision.decision_text, rejected_json, decision.reason,
                 decision.confidence, decision.valid_from.isoformat(),
                 decision.valid_until.isoformat() if decision.valid_until else None,
                 int(decision.is_current), decision.id)
            )
        return decision

    # Association operations
    def create_association(self, association: Association) -> Association:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO associations (source_memory_id, target_memory_id, association_type, strength, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (association.source_memory_id, association.target_memory_id,
                 association.association_type.value, association.strength, association.created_at.isoformat())
            )
            association.id = cursor.lastrowid
        return association

    def get_associations(self, memory_id: int, association_type: AssociationType | None = None) -> list[Association]:
        query = "SELECT * FROM associations WHERE source_memory_id = ? OR target_memory_id = ?"
        params = [memory_id, memory_id]
        if association_type:
            query += " AND association_type = ?"
            params.append(association_type.value)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_association(r) for r in rows]

    def get_related_memories(self, memory_id: int, min_strength: float = 0.3) -> list[tuple[Memory, Association]]:
        query = """
            SELECT m.*, a.* FROM memories m
            JOIN associations a ON (a.source_memory_id = m.id OR a.target_memory_id = m.id)
            WHERE (a.source_memory_id = ? OR a.target_memory_id = ?) AND a.strength >= ? AND m.id != ?
        """
        params = [memory_id, memory_id, min_strength, memory_id]
        rows = self._conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            assoc = Association(
                id=row["id"],
                source_memory_id=row["source_memory_id"],
                target_memory_id=row["target_memory_id"],
                association_type=AssociationType(row["association_type"]),
                strength=row["strength"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            results.append((memory, assoc))
        return results

    # Recall event operations
    def log_recall(self, recall: RecallEvent) -> RecallEvent:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO recalls (query, topic_id, recall_level, memories_retrieved, tokens_returned, latency_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (recall.query, recall.topic_id, recall.recall_level.value,
                 recall.memories_retrieved, recall.tokens_returned, recall.latency_ms,
                 recall.created_at.isoformat())
            )
            recall.id = cursor.lastrowid
        return recall

    def get_recent_recalls(self, topic_id: int | None = None, limit: int = 50) -> list[RecallEvent]:
        query = "SELECT * FROM recalls WHERE 1=1"
        params = []
        if topic_id is not None:
            query += " AND topic_id = ?"
            params.append(topic_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_recall(r) for r in rows]

    # Compression event operations
    def log_compression(self, event: CompressionEvent) -> CompressionEvent:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO compression_events (source_memory_id, target_memory_id, from_resolution, to_resolution,
                   original_tokens, compressed_tokens, compression_ratio, method, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event.source_memory_id, event.target_memory_id, event.from_resolution.value,
                 event.to_resolution.value, event.original_tokens, event.compressed_tokens,
                 event.compression_ratio, event.method.value, event.created_at.isoformat())
            )
            event.id = cursor.lastrowid
        return event

    def get_compression_history(self, memory_id: int) -> list[CompressionEvent]:
        rows = self._conn.execute(
            "SELECT * FROM compression_events WHERE source_memory_id = ? ORDER BY created_at",
            (memory_id,)
        ).fetchall()
        return [self._row_to_compression(r) for r in rows]

    # Context IR operations
    def add_context_ir(self, ir: ContextIR) -> ContextIR:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO context_ir (conversation_id, ir_type, ir_key, ir_value, sequence_num, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ir.conversation_id, ir.ir_type.value, ir.ir_key, ir.ir_value,
                 ir.sequence_num, ir.created_at.isoformat())
            )
            ir.id = cursor.lastrowid
        return ir

    def get_context_ir(self, conversation_id: int) -> list[ContextIR]:
        rows = self._conn.execute(
            "SELECT * FROM context_ir WHERE conversation_id = ? ORDER BY sequence_num",
            (conversation_id,)
        ).fetchall()
        return [self._row_to_context_ir(r) for r in rows]

    # Token usage operations
    def log_token_usage(self, usage: TokenUsage) -> TokenUsage:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO token_usage (conversation_id, operation, input_tokens, output_tokens, total_tokens, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (usage.conversation_id, usage.operation, usage.input_tokens,
                 usage.output_tokens, usage.total_tokens, usage.model, usage.created_at.isoformat())
            )
            usage.id = cursor.lastrowid
        return usage

    def get_token_usage(self, conversation_id: int | None = None, operation: str | None = None) -> list[TokenUsage]:
        query = "SELECT * FROM token_usage WHERE 1=1"
        params = []
        if conversation_id is not None:
            query += " AND conversation_id = ?"
            params.append(conversation_id)
        if operation is not None:
            query += " AND operation = ?"
            params.append(operation)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_token_usage(r) for r in rows]

    # ==================== Belief state persistence (P0-4) ====================

    def _row_to_belief_state(self, row: sqlite3.Row) -> BeliefState:
        def _loads(value: str | None) -> list[int]:
            if not value:
                return []
            try:
                return list(json.loads(value))
            except Exception:
                return []
        return BeliefState(
            id=row["id"],
            belief_key=row["belief_key"],
            proposition=row["proposition"],
            status=row["status"],
            confidence=row["confidence"],
            supporting_evidence=_loads(row["supporting_evidence"]),
            contradicting_evidence=_loads(row["contradicting_evidence"]),
            source_memories=_loads(row["source_memories"]),
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            revision=row["revision"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def upsert_belief_state(self, belief: BeliefState) -> BeliefState:
        """Insert or update a persisted belief (keyed by belief_key hash)."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO belief_states
                   (belief_key, proposition, status, confidence, supporting_evidence,
                    contradicting_evidence, source_memories, valid_from, valid_until,
                    revision, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(belief_key) DO UPDATE SET
                     proposition = excluded.proposition,
                     status = excluded.status,
                     confidence = excluded.confidence,
                     supporting_evidence = excluded.supporting_evidence,
                     contradicting_evidence = excluded.contradicting_evidence,
                     source_memories = excluded.source_memories,
                     valid_from = excluded.valid_from,
                     valid_until = excluded.valid_until,
                     revision = excluded.revision,
                     updated_at = excluded.updated_at""",
                (
                    belief.belief_key, belief.proposition, belief.status,
                    belief.confidence,
                    json.dumps(belief.supporting_evidence),
                    json.dumps(belief.contradicting_evidence),
                    json.dumps(belief.source_memories),
                    belief.valid_from.isoformat(),
                    belief.valid_until.isoformat() if belief.valid_until else None,
                    belief.revision,
                    belief.created_at.isoformat(),
                    belief.updated_at.isoformat(),
                )
            )
            if belief.id is None:
                belief.id = cursor.lastrowid
        return belief

    def get_all_belief_states(self) -> list[BeliefState]:
        rows = self._conn.execute("SELECT * FROM belief_states ORDER BY id").fetchall()
        return [self._row_to_belief_state(r) for r in rows]

    def get_belief_state_by_key(self, belief_key: str) -> BeliefState | None:
        row = self._conn.execute(
            "SELECT * FROM belief_states WHERE belief_key = ?", (belief_key,)
        ).fetchone()
        return self._row_to_belief_state(row) if row else None

    def get_belief_state(self, belief_id: int) -> BeliefState | None:
        row = self._conn.execute(
            "SELECT * FROM belief_states WHERE id = ?", (belief_id,)
        ).fetchone()
        return self._row_to_belief_state(row) if row else None

    # ==================== Evolution event persistence (P0-5) ====================

    def add_evolution_event(self, event: EvolutionEventRecord) -> EvolutionEventRecord:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO evolution_events
                   (memory_id, operation, source_memory_ids, target_memory_id,
                    description, old_content, new_content, metadata_json,
                    triggered_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.memory_id, event.operation,
                    json.dumps(event.source_memory_ids),
                    event.target_memory_id, event.description,
                    event.old_content, event.new_content,
                    json.dumps(event.metadata), event.triggered_by,
                    event.created_at.isoformat(),
                )
            )
            event.id = cursor.lastrowid
        return event

    def get_evolution_events(
        self,
        memory_id: int | None = None,
        operation: str | None = None,
        limit: int = 100,
    ) -> list[EvolutionEventRecord]:
        query = "SELECT * FROM evolution_events WHERE 1=1"
        params: list[Any] = []
        if memory_id is not None:
            query += " AND memory_id = ?"
            params.append(memory_id)
        if operation is not None:
            query += " AND operation = ?"
            params.append(operation)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        events = []
        for row in rows:
            try:
                source_ids = json.loads(row["source_memory_ids"]) if row["source_memory_ids"] else []
                metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            except Exception:
                source_ids, metadata = [], {}
            events.append(EvolutionEventRecord(
                id=row["id"],
                memory_id=row["memory_id"],
                operation=row["operation"],
                source_memory_ids=source_ids,
                target_memory_id=row["target_memory_id"],
                description=row["description"] or "",
                old_content=row["old_content"] or "",
                new_content=row["new_content"] or "",
                metadata=metadata,
                triggered_by=row["triggered_by"] or "auto",
                created_at=datetime.fromisoformat(row["created_at"]),
            ))
        return events

    # Health/management
    def health_check(self) -> bool:
        try:
            self._conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SQLiteMemoryStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
