from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from sqlalchemy import text

from src.db import native_search as ns
from src.db.database import get_session
from src.db.storage import hist_to_dense, load_codebook, log_search
from src.engine._utils import counts_to_hist
from src.engine.inverted_index import InvertedIndex
from src.engine.text_pipeline import prepare_query as _text_query_tf
from src.extraction.audio_mfcc import extract_mfcc_features
from src.extraction.image_sift import extract_sift_features
from src.ml.quantizer import VectorQuantizer


# ----------------------------------------------------------------------------
# Cache de indices SPIMI y de centroides
# ----------------------------------------------------------------------------
_SPIMI_CACHE: dict[tuple[str, str], InvertedIndex] = {}
_CENTROIDS_CACHE: dict[tuple[str, str], VectorQuantizer] = {}
_BAG_CACHE: dict[tuple[str, str], list[str]] = {}


def _get_codebook(app: str, modality: str) -> dict:
    cb = load_codebook(app, modality)
    if cb is None:
        raise LookupError(f"No hay codebook persistido para {app}/{modality}")
    return cb


def _get_spimi(app: str, modality: str) -> InvertedIndex:
    key = (app, modality)
    if key not in _SPIMI_CACHE:
        cb = _get_codebook(app, modality)
        idx_dir = Path(cb["index_dir"]) / "final"
        idx = InvertedIndex(
            postings_path=idx_dir / "final.postings",
            vocab_path=idx_dir / "vocab.json",
            meta_path=idx_dir / "meta.json",
        ).open()
        _SPIMI_CACHE[key] = idx
    return _SPIMI_CACHE[key]


def _get_bag(app: str, modality: str = "text") -> list[str]:
    key = (app, modality)
    if key not in _BAG_CACHE:
        cb = _get_codebook(app, modality)
        bag = cb["bag_of_words"]
        if not bag:
            raise LookupError(f"Codebook {app}/{modality} sin bag_of_words")
        _BAG_CACHE[key] = sorted(bag)
    return _BAG_CACHE[key]


def _get_quantizer(app: str, modality: str) -> VectorQuantizer:
    key = (app, modality)
    if key not in _CENTROIDS_CACHE:
        cb = _get_codebook(app, modality)
        path = cb["centroids_path"]
        if not path:
            raise LookupError(f"Codebook {app}/{modality} sin centroids_path")
        centroids = np.load(path)
        _CENTROIDS_CACHE[key] = VectorQuantizer([c for c in centroids])
    return _CENTROIDS_CACHE[key]


def clear_caches() -> None:
    for idx in _SPIMI_CACHE.values():
        idx.close()
    _SPIMI_CACHE.clear()
    _CENTROIDS_CACHE.clear()
    _BAG_CACHE.clear()


# ----------------------------------------------------------------------------
# Query -> histograma / embedding por modalidad
# ----------------------------------------------------------------------------
def _audio_query_hist(app: str, audio_path: Path) -> tuple[dict[str, int], list[str]]:
    vq = _get_quantizer(app, "audio")
    mfcc = extract_mfcc_features(str(audio_path))
    if mfcc.size == 0:
        return {}, [f"a_{i:04d}" for i in range(vq.n_centroids)]
    counts = vq.histogram(mfcc)
    hist = counts_to_hist(counts, prefix="a")
    keys = [f"a_{i:04d}" for i in range(vq.n_centroids)]
    return hist, keys


def _image_query_hist(app: str, image_path: Path) -> tuple[dict[str, int], list[str]]:
    vq = _get_quantizer(app, "image")
    desc = extract_sift_features(str(image_path))
    if desc.size == 0:
        return {}, [f"v_{i:04d}" for i in range(vq.n_centroids)]
    counts = vq.histogram(desc)
    hist = counts_to_hist(counts, prefix="v")
    keys = [f"v_{i:04d}" for i in range(vq.n_centroids)]
    return hist, keys


def _spimi_enrich(
    ranking: list[tuple[str, float]],
    table: str,
    extras: list[str],
    path_col: str,
) -> list[dict]:
    # Deduplicar por stem: si varios chunks de la misma cancion/producto aparecen
    # en el ranking, quedarnos solo con el mejor score.
    best_by_stem: dict[str, float] = {}
    order: list[str] = []
    for doc_id, score in ranking:
        stem = doc_id.split("#", 1)[0]
        if stem not in best_by_stem:
            order.append(stem)
            best_by_stem[stem] = float(score)
        else:
            if float(score) > best_by_stem[stem]:
                best_by_stem[stem] = float(score)

    stems = order
    stem_expr = f"regexp_replace({path_col}, '^.*/|\\.[^./]*$', '', 'g')"
    cols = ", ".join(extras)
    sql = f"""
        SELECT id, {cols}, {stem_expr} AS stem
        FROM {table}
        WHERE {stem_expr} = ANY(:stems)
    """
    with get_session() as session:
        rows = session.execute(text(sql), {"stems": stems}).all()
    lookup = {r._mapping["stem"]: r._mapping for r in rows}
    out: list[dict] = []
    for stem in stems:
        row = lookup.get(stem)
        entry: dict[str, Any] = {"id": row["id"] if row else stem}
        for col in extras:
            entry[col] = row[col] if row else None
        entry["score"] = best_by_stem[stem]
        entry["engine"] = "spimi"
        out.append(entry)
    return out


def _attach_media_urls(results: list[dict], app: str) -> list[dict]:
    if app == "music":
        url_field, endpoint = "audio_url", "/api/music/media/"
    elif app == "fashion":
        url_field, endpoint = "image_url", "/api/fashion/media/"
    else:
        return results
    for r in results:
        rid = r.get("id")
        if isinstance(rid, int):
            r[url_field] = f"{endpoint}{rid}"
    return results


# ----------------------------------------------------------------------------
# Routers de motores -> funcion nativa o SPIMI
# ----------------------------------------------------------------------------
_NATIVE_MUSIC_LYRICS = {
    "gin": ns.search_songs_lyrics_gin,
    "gist": ns.search_songs_lyrics_gist,
}
_NATIVE_MUSIC_AUDIO = {
    "pgvector": ns.search_songs_audio_pgvector,
}
_NATIVE_FASHION_DESC = {
    "gin": ns.search_products_desc_gin,
    "gist": ns.search_products_desc_gist,
}
_NATIVE_FASHION_IMAGE = {
    "pgvector": ns.search_products_image_pgvector,
}


# ----------------------------------------------------------------------------
# Entradas publicas: una por (app, modalidad)
# ----------------------------------------------------------------------------
_SONG_COLS = ["title", "artist", "genre", "lyrics_text"]
_PRODUCT_COLS = ["name", "category", "subcategory", "description"]


def search_music_lyrics(query_text: str, engine: str, k: int = 10) -> dict[str, Any]:
    t0 = time.perf_counter()
    if engine == "spimi":
        idx = _get_spimi("music", "text")
        bag = _get_bag("music", "text")
        tf = _text_query_tf(query_text, codebook=set(bag))
        ranking = idx.search_topk(tf, k=k * 3)
        results = _spimi_enrich(ranking, "songs", _SONG_COLS, "lyrics_path")[:k]
    elif engine in _NATIVE_MUSIC_LYRICS:
        fn = _NATIVE_MUSIC_LYRICS[engine]
        results = fn(query_text, limit=k)
    else:
        raise ValueError(f"engine no soportado: {engine}")
    _attach_media_urls(results, "music")
    latency = (time.perf_counter() - t0) * 1000
    log_search("music", "text", engine, query_text, latency, len(results))
    return {
        "app": "music", "modality": "text", "engine": engine,
        "query": query_text, "k": k,
        "latency_ms": round(latency, 3),
        "results": results,
    }


def search_music_audio(audio_path: Path, engine: str, k: int = 10) -> dict[str, Any]:
    t0 = time.perf_counter()
    hist, keys = _audio_query_hist("music", audio_path)
    if engine == "spimi":
        idx = _get_spimi("music", "audio")
        ranking = idx.search_topk(hist, k=k * 3)
        # Audio search: drop lyrics_text from response (irrelevant to the query)
        results = _spimi_enrich(ranking, "songs", ["title", "artist", "genre"], "audio_path")[:k]
    elif engine in _NATIVE_MUSIC_AUDIO:
        fn = _NATIVE_MUSIC_AUDIO[engine]
        emb = hist_to_dense(hist, keys)
        results = fn(emb, limit=k)
    else:
        raise ValueError(f"engine no soportado: {engine}")
    _attach_media_urls(results, "music")
    latency = (time.perf_counter() - t0) * 1000
    log_search("music", "audio", engine, str(audio_path.name), latency, len(results))
    return {
        "app": "music", "modality": "audio", "engine": engine,
        "query": str(audio_path.name), "k": k,
        "latency_ms": round(latency, 3),
        "results": results,
    }


def search_fashion_desc(query_text: str, engine: str, k: int = 10) -> dict[str, Any]:
    t0 = time.perf_counter()
    if engine == "spimi":
        idx = _get_spimi("fashion", "text")
        bag = _get_bag("fashion", "text")
        tf = _text_query_tf(query_text, codebook=set(bag))
        ranking = idx.search_topk(tf, k=k * 3)
        results = _spimi_enrich(ranking, "products", _PRODUCT_COLS, "image_path")[:k]
    elif engine in _NATIVE_FASHION_DESC:
        fn = _NATIVE_FASHION_DESC[engine]
        results = fn(query_text, limit=k)
    else:
        raise ValueError(f"engine no soportado: {engine}")
    _attach_media_urls(results, "fashion")
    latency = (time.perf_counter() - t0) * 1000
    log_search("fashion", "text", engine, query_text, latency, len(results))
    return {
        "app": "fashion", "modality": "text", "engine": engine,
        "query": query_text, "k": k,
        "latency_ms": round(latency, 3),
        "results": results,
    }


def search_fashion_image(image_path: Path, engine: str, k: int = 10) -> dict[str, Any]:
    t0 = time.perf_counter()
    hist, keys = _image_query_hist("fashion", image_path)
    if engine == "spimi":
        idx = _get_spimi("fashion", "image")
        ranking = idx.search_topk(hist, k=k * 3)
        results = _spimi_enrich(ranking, "products", _PRODUCT_COLS, "image_path")[:k]
    elif engine in _NATIVE_FASHION_IMAGE:
        fn = _NATIVE_FASHION_IMAGE[engine]
        emb = hist_to_dense(hist, keys)
        results = fn(emb, limit=k)
    else:
        raise ValueError(f"engine no soportado: {engine}")
    _attach_media_urls(results, "fashion")
    latency = (time.perf_counter() - t0) * 1000
    log_search("fashion", "image", engine, str(image_path.name), latency, len(results))
    return {
        "app": "fashion", "modality": "image", "engine": engine,
        "query": str(image_path.name), "k": k,
        "latency_ms": round(latency, 3),
        "results": results,
    }


__all__ = [
    "clear_caches",
    "search_music_lyrics",
    "search_music_audio",
    "search_fashion_desc",
    "search_fashion_image",
]