from __future__ import annotations
import json
from math import sqrt
from typing import Any, Iterable
from sqlalchemy import text
from src.db.database import get_session


# Conversion histograma -> vector denso (para pgvector)
def hist_to_dense(
    hist: dict[str, int],
    codebook_keys: list[str],
    normalize: bool = True,
) -> list[float]:
    vec = [float(hist.get(k, 0)) for k in codebook_keys]
    if normalize:
        n = sqrt(sum(v * v for v in vec))
        if n > 0:
            vec = [v / n for v in vec]
    return vec


def _vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


# Ajuste dinamico de dimension de columnas pgvector
_HNSW_INDEX_BY_COLUMN = {
    ("songs", "audio_emb"): "idx_songs_audio_emb_hnsw",
    ("products", "image_emb"): "idx_products_image_emb_hnsw",
}


def _current_emb_dim(session, table: str, column: str) -> int | None:
    row = session.execute(
        text(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS type_name
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = :table AND a.attname = :column
            """
        ),
        {"table": table, "column": column},
    ).first()
    if row is None:
        return None
    type_name = row[0]
    # 'vector(1000)' -> 1000
    if "(" in type_name and ")" in type_name:
        try:
            return int(type_name.split("(")[1].split(")")[0])
        except (IndexError, ValueError):
            return None
    return None


def ensure_emb_column(table: str, column: str, dim: int) -> bool:
    if dim <= 0:
        raise ValueError(f"dim debe ser positiva, recibido {dim}")
    with get_session() as session:
        current = _current_emb_dim(session, table, column)
        if current == dim:
            return False
        index_name = _HNSW_INDEX_BY_COLUMN.get((table, column))
        if index_name:
            session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
        session.execute(text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}"))
        session.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} vector({dim})"))
        if index_name:
            session.execute(
                text(
                    f"CREATE INDEX {index_name} ON {table} "
                    f"USING hnsw ({column} vector_cosine_ops)"
                )
            )
        session.commit()
        return True


# Reset / truncate
def reset_table(table: str) -> None:
    with get_session() as session:
        session.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
        session.commit()


# Inserts
def _insert_one_returning_id(sql: str, params: dict) -> int:
    with get_session() as session:
        row = session.execute(text(sql + " RETURNING id"), params).one()
        session.commit()
        return int(row[0])


def save_song(
    title: str,
    artist: str | None,
    genre: str | None,
    lyrics_path: str | None,
    audio_path: str | None,
    lyrics_text: str | None,
    lyrics_hist: dict[str, int],
    audio_hist: dict[str, int],
    audio_emb: list[float] | None,
    metadata: dict[str, Any] | None = None,
) -> int:
    params = _song_params(
        title, artist, genre, lyrics_path, audio_path,
        lyrics_text, lyrics_hist, audio_hist,
        audio_emb, metadata,
    )
    return _insert_one_returning_id(_SONG_INSERT_SQL, params)


_SONG_INSERT_SQL = """
INSERT INTO songs (
    title, artist, genre, lyrics_path, audio_path,
    lyrics_text, lyrics_hist, audio_hist,
    audio_emb, metadata
) VALUES (
    :title, :artist, :genre, :lyrics_path, :audio_path,
    :lyrics_text,
    CAST(:lyrics_hist AS jsonb),
    CAST(:audio_hist AS jsonb),
    CAST(:audio_emb AS vector),
    CAST(:metadata AS jsonb)
)
"""

_PRODUCT_INSERT_SQL = """
INSERT INTO products (
    name, category, subcategory, image_path, description,
    desc_hist, image_hist, image_emb, metadata
) VALUES (
    :name, :category, :subcategory, :image_path, :description,
    CAST(:desc_hist AS jsonb),
    CAST(:image_hist AS jsonb),
    CAST(:image_emb AS vector),
    CAST(:metadata AS jsonb)
)
"""


def _song_params(
    title: str, artist: str | None, genre: str | None,
    lyrics_path: str | None, audio_path: str | None,
    lyrics_text: str | None,
    lyrics_hist: dict[str, int], audio_hist: dict[str, int],
    audio_emb: list[float] | None,
    metadata: dict[str, Any] | None,
) -> dict:
    return {
        "title": title, "artist": artist, "genre": genre,
        "lyrics_path": lyrics_path, "audio_path": audio_path,
        "lyrics_text": lyrics_text,
        "lyrics_hist": json.dumps(lyrics_hist or {}),
        "audio_hist": json.dumps(audio_hist or {}),
        "audio_emb": _vector_literal(audio_emb) if audio_emb else None,
        "metadata": json.dumps(metadata or {}),
    }


def _product_params(
    name: str, category: str | None, subcategory: str | None,
    image_path: str | None, description: str | None,
    desc_hist: dict[str, int], image_hist: dict[str, int],
    image_emb: list[float] | None,
    metadata: dict[str, Any] | None,
) -> dict:
    return {
        "name": name, "category": category, "subcategory": subcategory,
        "image_path": image_path, "description": description,
        "desc_hist": json.dumps(desc_hist or {}),
        "image_hist": json.dumps(image_hist or {}),
        "image_emb": _vector_literal(image_emb) if image_emb else None,
        "metadata": json.dumps(metadata or {}),
    }


def save_songs_batch(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_session() as session:
        session.execute(text(_SONG_INSERT_SQL), rows)
        session.commit()
    return len(rows)


def save_products_batch(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_session() as session:
        session.execute(text(_PRODUCT_INSERT_SQL), rows)
        session.commit()
    return len(rows)


def save_product(
    name: str,
    category: str | None,
    subcategory: str | None,
    image_path: str | None,
    description: str | None,
    desc_hist: dict[str, int],
    image_hist: dict[str, int],
    image_emb: list[float] | None,
    metadata: dict[str, Any] | None = None,
) -> int:
    params = _product_params(
        name, category, subcategory, image_path, description,
        desc_hist, image_hist, image_emb, metadata,
    )
    return _insert_one_returning_id(_PRODUCT_INSERT_SQL, params)


def save_codebook(
    app: str,
    modality: str,
    codebook_size: int,
    bag_of_words: list[str] | None = None,
    centroids_path: str | None = None,
    index_dir: str | None = None,
) -> int:
    params = {
        "app": app,
        "modality": modality,
        "codebook_size": codebook_size,
        "bag_of_words": json.dumps(bag_of_words) if bag_of_words is not None else None,
        "centroids_path": centroids_path,
        "index_dir": index_dir,
    }
    with get_session() as session:
        row = session.execute(
            text(
                """
                INSERT INTO codebooks (
                    app, modality, codebook_size,
                    bag_of_words, centroids_path, index_dir
                ) VALUES (
                    :app, :modality, :codebook_size,
                    CAST(:bag_of_words AS jsonb), :centroids_path, :index_dir
                )
                ON CONFLICT (app, modality) DO UPDATE SET
                    codebook_size = EXCLUDED.codebook_size,
                    bag_of_words = EXCLUDED.bag_of_words,
                    centroids_path = EXCLUDED.centroids_path,
                    index_dir = EXCLUDED.index_dir,
                    created_at = NOW()
                RETURNING id
                """
            ),
            params,
        ).one()
        session.commit()
        return int(row[0])


def load_codebook(app: str, modality: str) -> dict | None:
    with get_session() as session:
        row = session.execute(
            text(
                """
                SELECT codebook_size, bag_of_words, centroids_path, index_dir
                FROM codebooks WHERE app = :app AND modality = :modality
                """
            ),
            {"app": app, "modality": modality},
        ).first()
        if row is None:
            return None
        return {
            "codebook_size": int(row[0]),
            "bag_of_words": row[1],
            "centroids_path": row[2],
            "index_dir": row[3],
        }


def log_search(
    app: str,
    modality: str,
    engine: str,
    query: str | None,
    latency_ms: float,
    n_results: int,
) -> None:
    try:
        with get_session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO search_logs (app, modality, engine, query, latency_ms, n_results)
                    VALUES (:app, :modality, :engine, :query, :latency_ms, :n_results)
                    """
                ),
                {
                    "app": app,
                    "modality": modality,
                    "engine": engine,
                    "query": query,
                    "latency_ms": latency_ms,
                    "n_results": n_results,
                },
            )
            session.commit()
    except Exception:
        pass


__all__ = [
    "hist_to_dense",
    "ensure_emb_column",
    "reset_table",
    "save_song",
    "save_songs_batch",
    "save_product",
    "save_products_batch",
    "_song_params",
    "_product_params",
    "save_codebook",
    "load_codebook",
    "log_search",
]