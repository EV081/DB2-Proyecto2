CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS items (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    modality TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT,
    histogram JSONB DEFAULT '{}'::jsonb,
    embedding vector(8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    modality TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO items (title, modality, metadata, embedding)
VALUES
    ('Item demo audio', 'audio_text', '{"source": "mock"}', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'),
    ('Item demo tienda', 'image_text', '{"source": "mock"}', '[0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]')
ON CONFLICT DO NOTHING;
