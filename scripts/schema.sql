-- Artificial Memory SQLite Schema v0.1
-- Based on Planning.txt sections 25, 38, 50

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- Projects table (top-level topics)
CREATE TABLE projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Topics table (hierarchical topics within projects)
CREATE TABLE topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,  -- e.g., "Projects/Artificial-Memory/Architecture"
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_topics_project ON topics(project_id);
CREATE INDEX idx_topics_path ON topics(path);
CREATE INDEX idx_topics_parent ON topics(parent_id);

-- Conversations table (each conversation session)
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    title TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    metadata_json TEXT,  -- YAML/JSON: tone, style, importance, etc.
    status TEXT DEFAULT 'active'  -- active, completed, archived
);

CREATE INDEX idx_conversations_topic ON conversations(topic_id);
CREATE INDEX idx_conversations_started ON conversations(started_at);

-- Messages table (raw conversation messages)
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    sequence_num INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT  -- YAML/JSON: filler, hesitation markers, etc.
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_sequence ON messages(conversation_id, sequence_num);

-- Memories table (core memory units at various resolutions)
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    memory_type TEXT NOT NULL,  -- 'current', 'timeline', 'decision', 'episode', 'semantic', 'conversation_style'
    content TEXT NOT NULL,  -- Markdown content
    resolution INTEGER NOT NULL DEFAULT 0,  -- 0=raw, 1=light, 2=episode, 3=semantic, 4=longterm, 5=deeplongterm
    importance REAL DEFAULT 0.5,  -- 0.0 to 1.0
    confidence REAL DEFAULT 0.9,  -- 0.0 to 1.0
    status TEXT DEFAULT 'active',  -- 'active', 'dormant', 'compressed', 'archived', 'deep_archived'
    valid_from TIMESTAMP,
    valid_until TIMESTAMP,
    is_current BOOLEAN DEFAULT 1,
    source_conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    source_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX idx_memories_topic ON memories(topic_id);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_resolution ON memories(resolution);
CREATE INDEX idx_memories_status ON memories(status);
CREATE INDEX idx_memories_current ON memories(is_current) WHERE is_current = 1;
CREATE INDEX idx_memories_accessed ON memories(last_accessed);

-- Memory versions table (multiple resolutions for same memory)
CREATE TABLE memory_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    resolution INTEGER NOT NULL,
    content TEXT NOT NULL,
    compression_ratio REAL,  -- tokens_original / tokens_compressed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT,  -- 'auto', 'manual', 'recall_expansion'
    policy_version TEXT DEFAULT 'decision-v1',  -- P0-7: compression policy that produced this version
    algorithm_version TEXT DEFAULT 'rule-compressor-v1'  -- P0-7: compression algorithm version
);

CREATE INDEX idx_memory_versions_memory ON memory_versions(memory_id);
CREATE INDEX idx_memory_versions_resolution ON memory_versions(resolution);

-- Decisions table (important decisions extracted from conversations)
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL,
    decision_text TEXT NOT NULL,
    rejected_options TEXT,  -- JSON array
    reason TEXT,
    confidence REAL DEFAULT 0.8,
    decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMP,
    is_current BOOLEAN DEFAULT 1
);

CREATE INDEX idx_decisions_topic ON decisions(topic_id);
CREATE INDEX idx_decisions_current ON decisions(is_current) WHERE is_current = 1;

-- Associations table (memory-to-memory relationships)
CREATE TABLE associations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    association_type TEXT NOT NULL,  -- 'related', 'causes', 'follows', 'contradicts', 'elaborates', 'summarizes'
    strength REAL DEFAULT 0.5,  -- 0.0 to 1.0
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_associations_source ON associations(source_memory_id);
CREATE INDEX idx_associations_target ON associations(target_memory_id);
CREATE INDEX idx_associations_type ON associations(association_type);

-- Recalls table (tracking recall events for analytics)
CREATE TABLE recalls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    recall_level INTEGER NOT NULL,  -- 0=current, 1=longterm, 2=episode, 3=light, 4=raw
    memories_retrieved INTEGER DEFAULT 0,
    tokens_returned INTEGER DEFAULT 0,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_recalls_topic ON recalls(topic_id);
CREATE INDEX idx_recalls_created ON recalls(created_at);

-- Compression events table (tracking compression pipeline)
CREATE TABLE compression_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL,
    from_resolution INTEGER NOT NULL,
    to_resolution INTEGER NOT NULL,
    original_tokens INTEGER,
    compressed_tokens INTEGER,
    compression_ratio REAL,
    method TEXT,  -- 'light', 'episode', 'semantic', 'longterm'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_compression_source ON compression_events(source_memory_id);

-- Context IR table (intermediate representation units)
CREATE TABLE context_ir (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    ir_type TEXT NOT NULL,  -- 'tone', 'state', 'decision', 'reasoning', 'temporal', 'knowledge', 'event'
    ir_key TEXT NOT NULL,  -- e.g., 'c', 'h', 'q', 'r', 'd', 'i', 't', 's', 'k', 'de'
    ir_value TEXT,  -- associated value
    sequence_num INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_context_ir_conversation ON context_ir(conversation_id);
CREATE INDEX idx_context_ir_type ON context_ir(ir_type);

-- Token usage tracking
CREATE TABLE token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    operation TEXT NOT NULL,  -- 'chat', 'compress', 'recall', 'compile', 'optimize'
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_token_usage_conversation ON token_usage(conversation_id);
CREATE INDEX idx_token_usage_operation ON token_usage(operation);
CREATE INDEX idx_token_usage_created ON token_usage(created_at);

-- Belief states table (P0-4: beliefs persisted separately from evidence).
-- V1 stores evidence id lists as JSON (JSONB in Postgres); V2 may normalize.
CREATE TABLE belief_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_key TEXT NOT NULL UNIQUE,  -- proposition hash for deduplication
    proposition TEXT NOT NULL,
    status TEXT DEFAULT 'accepted',  -- 'accepted', 'contested', 'rejected', 'superseded'
    confidence REAL DEFAULT 0.5,
    supporting_evidence TEXT,  -- JSON array of memory ids
    contradicting_evidence TEXT,  -- JSON array of memory ids
    source_memories TEXT,  -- JSON array of memory ids
    valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMP,
    revision INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_belief_states_key ON belief_states(belief_key);
CREATE INDEX idx_belief_states_status ON belief_states(status);

-- Evolution events table (P0-5: every meaningful memory change is persisted).
CREATE TABLE evolution_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    operation TEXT NOT NULL,  -- 'revise', 'merge', 'split', 'contradict', 'supersede', 'reactivate', 'heal'
    source_memory_ids TEXT,  -- JSON array of memory ids (for merge/split)
    target_memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL,
    description TEXT,
    old_content TEXT,
    new_content TEXT,
    metadata_json TEXT,  -- JSON object
    triggered_by TEXT DEFAULT 'auto',  -- 'auto', 'user', 'integrity_check', 'recall'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_evolution_events_memory ON evolution_events(memory_id);
CREATE INDEX idx_evolution_events_operation ON evolution_events(operation);
CREATE INDEX idx_evolution_events_created ON evolution_events(created_at);

-- Views for common queries
CREATE VIEW active_memories AS
SELECT * FROM memories WHERE status = 'active' AND is_current = 1;

CREATE VIEW current_project_state AS
SELECT p.name as project, t.path as topic, m.memory_type, m.content, m.resolution, m.importance
FROM memories m
JOIN topics t ON m.topic_id = t.id
JOIN projects p ON t.project_id = p.id
WHERE m.is_current = 1 AND m.status = 'active'
ORDER BY p.name, t.path, m.memory_type, m.resolution;

CREATE VIEW memory_hierarchy AS
SELECT 
    m.id,
    m.topic_id,
    t.path as topic_path,
    m.memory_type,
    m.resolution,
    m.status,
    m.importance,
    mv.resolution as version_resolution,
    mv.compression_ratio
FROM memories m
LEFT JOIN memory_versions mv ON m.id = mv.memory_id
JOIN topics t ON m.topic_id = t.id
ORDER BY m.topic_id, m.memory_type, m.resolution;