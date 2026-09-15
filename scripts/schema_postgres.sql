-- Artificial Memory PostgreSQL/pgvector Schema v0.1
-- Compatible with PostgreSQL 14+ and pgvector 0.5+

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Projects table (top-level topics)
CREATE TABLE projects (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Topics table (hierarchical topics within projects)
CREATE TABLE topics (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    parent_id BIGINT REFERENCES topics(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_topics_project ON topics(project_id);
CREATE INDEX idx_topics_path ON topics(path);
CREATE INDEX idx_topics_parent ON topics(parent_id);

-- Conversations table (each conversation session)
CREATE TABLE conversations (
    id BIGSERIAL PRIMARY KEY,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    title TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    message_count INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    metadata_json JSONB,
    status TEXT DEFAULT 'active'
);

CREATE INDEX idx_conversations_topic ON conversations(topic_id);
CREATE INDEX idx_conversations_started ON conversations(started_at);
CREATE INDEX idx_conversations_status ON conversations(status);

-- Messages table (raw conversation messages)
CREATE TABLE messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    sequence_num INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata_json JSONB
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_sequence ON messages(conversation_id, sequence_num);

-- Memories table (core memory units at various resolutions)
CREATE TABLE memories (
    id BIGSERIAL PRIMARY KEY,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    resolution INTEGER NOT NULL DEFAULT 0,
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.9,
    status TEXT DEFAULT 'active',
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    is_current BOOLEAN DEFAULT TRUE,
    source_conversation_id BIGINT REFERENCES conversations(id) ON DELETE SET NULL,
    source_message_id BIGINT REFERENCES messages(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ,
    access_count INTEGER DEFAULT 0,
    -- Vector embedding for semantic search (384 dimensions for all-MiniLM-L6-v2)
    embedding VECTOR(384)
);

CREATE INDEX idx_memories_topic ON memories(topic_id);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_resolution ON memories(resolution);
CREATE INDEX idx_memories_status ON memories(status);
CREATE INDEX idx_memories_current ON memories(is_current) WHERE is_current = TRUE;
CREATE INDEX idx_memories_accessed ON memories(last_accessed);
CREATE INDEX idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Memory versions table (multiple resolutions for same memory)
CREATE TABLE memory_versions (
    id BIGSERIAL PRIMARY KEY,
    memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    resolution INTEGER NOT NULL,
    content TEXT NOT NULL,
    compression_ratio REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    source TEXT
);

CREATE INDEX idx_memory_versions_memory ON memory_versions(memory_id);
CREATE INDEX idx_memory_versions_resolution ON memory_versions(resolution);

-- Decisions table (important decisions extracted from conversations)
CREATE TABLE decisions (
    id BIGSERIAL PRIMARY KEY,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    memory_id BIGINT REFERENCES memories(id) ON DELETE SET NULL,
    decision_text TEXT NOT NULL,
    rejected_options JSONB,
    reason TEXT,
    confidence REAL DEFAULT 0.8,
    decided_at TIMESTAMPTZ DEFAULT NOW(),
    valid_from TIMESTAMPTZ DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    is_current BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_decisions_topic ON decisions(topic_id);
CREATE INDEX idx_decisions_current ON decisions(is_current) WHERE is_current = TRUE;

-- Associations table (memory-to-memory relationships)
CREATE TABLE associations (
    id BIGSERIAL PRIMARY KEY,
    source_memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    association_type TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_associations_source ON associations(source_memory_id);
CREATE INDEX idx_associations_target ON associations(target_memory_id);
CREATE INDEX idx_associations_type ON associations(association_type);

-- Recalls table (tracking recall events for analytics)
CREATE TABLE recalls (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    topic_id BIGINT REFERENCES topics(id) ON DELETE SET NULL,
    recall_level INTEGER NOT NULL,
    memories_retrieved INTEGER DEFAULT 0,
    tokens_returned INTEGER DEFAULT 0,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_recalls_topic ON recalls(topic_id);
CREATE INDEX idx_recalls_created ON recalls(created_at);

-- Compression events table (tracking compression pipeline)
CREATE TABLE compression_events (
    id BIGSERIAL PRIMARY KEY,
    source_memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id BIGINT REFERENCES memories(id) ON DELETE SET NULL,
    from_resolution INTEGER NOT NULL,
    to_resolution INTEGER NOT NULL,
    original_tokens INTEGER,
    compressed_tokens INTEGER,
    compression_ratio REAL,
    method TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_compression_source ON compression_events(source_memory_id);

-- Context IR table (intermediate representation units)
CREATE TABLE context_ir (
    id BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    ir_type TEXT NOT NULL,
    ir_key TEXT NOT NULL,
    ir_value TEXT,
    sequence_num INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_context_ir_conversation ON context_ir(conversation_id);
CREATE INDEX idx_context_ir_type ON context_ir(ir_type);

-- Token usage tracking
CREATE TABLE token_usage (
    id BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT REFERENCES conversations(id) ON DELETE SET NULL,
    operation TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    model TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_token_usage_conversation ON token_usage(conversation_id);
CREATE INDEX idx_token_usage_operation ON token_usage(operation);
CREATE INDEX idx_token_usage_created ON token_usage(created_at);

-- Vector index metadata table (for tracking FAISS/pgvector index state)
CREATE TABLE vector_index_metadata (
    id BIGSERIAL PRIMARY KEY,
    index_name TEXT NOT NULL UNIQUE,
    dimension INTEGER NOT NULL,
    total_vectors BIGINT DEFAULT 0,
    index_type TEXT DEFAULT 'hnsw',
    m INTEGER DEFAULT 16,
    ef_construction INTEGER DEFAULT 64,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default vector index metadata
INSERT INTO vector_index_metadata (index_name, dimension, total_vectors)
VALUES ('memories', 384, 0)
ON CONFLICT (index_name) DO NOTHING;

-- Views for common queries
CREATE VIEW active_memories AS
SELECT * FROM memories WHERE status = 'active' AND is_current = TRUE;

CREATE VIEW current_project_state AS
SELECT p.name as project, t.path as topic, m.memory_type, m.content, m.resolution, m.importance
FROM memories m
JOIN topics t ON m.topic_id = t.id
JOIN projects p ON t.project_id = p.id
WHERE m.is_current = TRUE AND m.status = 'active'
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

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_topics_updated_at BEFORE UPDATE ON topics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_memories_updated_at BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vector_index_metadata_updated_at BEFORE UPDATE ON vector_index_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();