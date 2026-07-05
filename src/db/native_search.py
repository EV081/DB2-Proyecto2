from __future__ import annotations
import re
from typing import Any, Iterable
from sqlalchemy import text
from src.db.database import get_session


def _vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row._mapping)


def _to_or_tsquery(query: str) -> str:
    # Convierte "hello world foo" en "hello | world | foo" para semantica OR.
    # Filtra caracteres no alfanumericos (evita errores de sintaxis en to_tsquery).
    words = re.findall(r"[a-z0-9]+", query.lower())
    if not words:
        return "unlikelywordxyz"
    return " | ".join(words)


_SONG_TEXT_COLS = "s.id, s.title, s.artist, s.genre, s.lyrics_text"
_SONG_AUDIO_COLS = "s.id, s.title, s.artist, s.genre"
_PRODUCT_COLS = "p.id, p.name, p.category, p.subcategory, p.description"


# Musica - Lyrics (texto)
def _search_songs_lyrics_fts(query: str, limit: int, index_kind: str) -> list[dict]:
    ts_query = _to_or_tsquery(query)
    sql = f"""
        SELECT
            {_SONG_TEXT_COLS},
            ts_rank(s.lyrics_tsv, to_tsquery('english', :ts_query)) AS score,
            :engine AS engine
        FROM songs s
        WHERE s.lyrics_tsv @@ to_tsquery('english', :ts_query)
        ORDER BY score DESC, s.id ASC
        LIMIT :limit
    """
    engine = f"postgres_{index_kind}_full_text"
    with get_session() as session:
        rows = session.execute(text(sql), {"ts_query": ts_query, "limit": limit, "engine": engine}).all()
    return [_row_to_dict(r) for r in rows]


def search_songs_lyrics_gin(query: str, limit: int = 10) -> list[dict]:
    return _search_songs_lyrics_fts(query, limit, "gin")


def search_songs_lyrics_gist(query: str, limit: int = 10) -> list[dict]:
    return _search_songs_lyrics_fts(query, limit, "gist")


def search_songs_audio_pgvector(query_emb: list[float], limit: int = 10) -> list[dict]:
    vec = _vector_literal(query_emb)
    sql = f"""
        SELECT
            {_SONG_AUDIO_COLS},
            1.0 - (s.audio_emb <=> CAST(:vec AS vector)) AS score,
            'pgvector_hnsw_cosine' AS engine
        FROM songs s
        WHERE s.audio_emb IS NOT NULL
        ORDER BY s.audio_emb <=> CAST(:vec AS vector)
        LIMIT :limit
    """
    with get_session() as session:
        rows = session.execute(text(sql), {"vec": vec, "limit": limit}).all()
    return [_row_to_dict(r) for r in rows]


# Fashion - Descripcion (texto)
def _search_products_desc_fts(query: str, limit: int, index_kind: str) -> list[dict]:
    ts_query = _to_or_tsquery(query)
    sql = f"""
        SELECT
            {_PRODUCT_COLS},
            ts_rank(p.description_tsv, to_tsquery('english', :ts_query)) AS score,
            :engine AS engine
        FROM products p
        WHERE p.description_tsv @@ to_tsquery('english', :ts_query)
        ORDER BY score DESC, p.id ASC
        LIMIT :limit
    """
    engine = f"postgres_{index_kind}_full_text"
    with get_session() as session:
        rows = session.execute(text(sql), {"ts_query": ts_query, "limit": limit, "engine": engine}).all()
    return [_row_to_dict(r) for r in rows]


def search_products_desc_gin(query: str, limit: int = 10) -> list[dict]:
    return _search_products_desc_fts(query, limit, "gin")


def search_products_desc_gist(query: str, limit: int = 10) -> list[dict]:
    return _search_products_desc_fts(query, limit, "gist")


def search_products_image_pgvector(query_emb: list[float], limit: int = 10) -> list[dict]:
    vec = _vector_literal(query_emb)
    sql = f"""
        SELECT
            {_PRODUCT_COLS},
            1.0 - (p.image_emb <=> CAST(:vec AS vector)) AS score,
            'pgvector_hnsw_cosine' AS engine
        FROM products p
        WHERE p.image_emb IS NOT NULL
        ORDER BY p.image_emb <=> CAST(:vec AS vector)
        LIMIT :limit
    """
    with get_session() as session:
        rows = session.execute(text(sql), {"vec": vec, "limit": limit}).all()
    return [_row_to_dict(r) for r in rows]


__all__ = [
    "search_songs_lyrics_gin",
    "search_songs_lyrics_gist",
    "search_songs_audio_pgvector",
    "search_products_desc_gin",
    "search_products_desc_gist",
    "search_products_image_pgvector",
]