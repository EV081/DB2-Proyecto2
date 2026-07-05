from datetime import datetime
from typing import Any, Optional
from sqlalchemy import Column, DateTime, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None


def _vector_col(dim: int) -> Column:
    if Vector is None:
        return Column(Text, nullable=True)
    return Column(Vector(dim), nullable=True)


def _jsonb_col(default_factory=dict) -> Column:
    return Column(JSONB, nullable=False, server_default="{}")


class Song(SQLModel, table=True):
    __tablename__ = "songs"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    artist: Optional[str] = None
    genre: Optional[str] = None
    lyrics_path: Optional[str] = None
    audio_path: Optional[str] = None
    lyrics_text: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    lyrics_hist: dict[str, Any] = Field(default_factory=dict, sa_column=_jsonb_col())
    audio_hist: dict[str, Any] = Field(default_factory=dict, sa_column=_jsonb_col())
    audio_emb: Any | None = Field(default=None, sa_column=_vector_col(500))
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False),
    )


class Product(SQLModel, table=True):
    __tablename__ = "products"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    image_path: Optional[str] = None
    description: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    desc_hist: dict[str, Any] = Field(default_factory=dict, sa_column=_jsonb_col())
    image_hist: dict[str, Any] = Field(default_factory=dict, sa_column=_jsonb_col())
    image_emb: Any | None = Field(default=None, sa_column=_vector_col(1024))
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False),
    )


class Codebook(SQLModel, table=True):
    __tablename__ = "codebooks"
    __table_args__ = (UniqueConstraint("app", "modality"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    app: str = Field(index=True)
    modality: str = Field(index=True)
    codebook_size: int = Field(sa_column=Column(Integer, nullable=False))
    bag_of_words: Optional[list] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    centroids_path: Optional[str] = None
    index_dir: Optional[str] = None
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False),
    )


class SearchLog(SQLModel, table=True):
    __tablename__ = "search_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    app: str = Field(index=True)
    modality: str = Field(index=True)
    engine: str = Field(index=True)
    query: Optional[str] = None
    latency_ms: Optional[float] = None
    n_results: Optional[int] = None
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False),
    )
