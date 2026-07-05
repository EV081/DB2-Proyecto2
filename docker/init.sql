-- Schema para Musica (audio y letra) + Fashion (imágenes y descripción)

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS search_logs CASCADE;
DROP TABLE IF EXISTS songs CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS codebooks CASCADE;
DROP TABLE IF EXISTS items CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;


-- App Busqueda Musical Inteligente
CREATE TABLE songs (
    id           SERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    artist       TEXT,
    genre        TEXT,
    lyrics_path  TEXT,
    audio_path   TEXT,
    lyrics_text  TEXT,
    lyrics_tsv   tsvector GENERATED ALWAYS AS
                 (to_tsvector('english', coalesce(lyrics_text, ''))) STORED,
    lyrics_hist  JSONB NOT NULL DEFAULT '{}'::jsonb,
    audio_hist   JSONB NOT NULL DEFAULT '{}'::jsonb,
    audio_emb    vector(500),
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_songs_lyrics_tsv_gin
    ON songs USING gin (lyrics_tsv);
CREATE INDEX idx_songs_lyrics_tsv_gist
    ON songs USING gist (lyrics_tsv);
CREATE INDEX idx_songs_metadata_gin
    ON songs USING gin (metadata);
CREATE INDEX idx_songs_audio_emb_hnsw
    ON songs USING hnsw (audio_emb vector_cosine_ops);

-- App Multimodal (Fashion)
CREATE TABLE products (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT,
    subcategory     TEXT,
    image_path      TEXT,
    description     TEXT,
    description_tsv tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', coalesce(description, ''))) STORED,
    desc_hist       JSONB NOT NULL DEFAULT '{}'::jsonb,
    image_hist      JSONB NOT NULL DEFAULT '{}'::jsonb,
    image_emb       vector(1024),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_products_desc_tsv_gin
    ON products USING gin (description_tsv);
CREATE INDEX idx_products_desc_tsv_gist
    ON products USING gist (description_tsv);
CREATE INDEX idx_products_metadata_gin
    ON products USING gin (metadata);
CREATE INDEX idx_products_image_emb_hnsw
    ON products USING hnsw (image_emb vector_cosine_ops);

-- Codebooks
CREATE TABLE codebooks (
    id             SERIAL PRIMARY KEY,
    app            TEXT NOT NULL,        -- 'music' | 'fashion'
    modality       TEXT NOT NULL,        -- 'text' | 'audio' | 'image'
    codebook_size  INTEGER NOT NULL,
    bag_of_words   JSONB,                
    centroids_path TEXT,                 -- audio/imagen: path a binario .npy
    index_dir      TEXT,                 -- SPIMI
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (app, modality)
);

-- Logs de busqueda para benchmark
CREATE TABLE search_logs (
    id           SERIAL PRIMARY KEY,
    app          TEXT NOT NULL,           -- 'music' | 'fashion'
    modality     TEXT NOT NULL,           -- 'text' | 'audio' | 'image'
    engine       TEXT NOT NULL,           -- 'spimi' | 'pgvector' | 'gin' | 'gist'
    query        TEXT,
    latency_ms   DOUBLE PRECISION,
    n_results    INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_search_logs_grouping
    ON search_logs (app, modality, engine);
