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


__all__ = [
    "hist_to_dense",
    "save_song",
    "save_songs_batch",
    "save_product",
    "save_products_batch",
    "_song_params",
    "_product_params",
]
