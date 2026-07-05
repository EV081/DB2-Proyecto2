from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from src.api.search_services import (
    clear_caches,
    search_fashion_desc,
    search_fashion_image,
    search_music_audio,
    search_music_lyrics,
)
from src.db.database import get_session


# Carga de ground truth desde DB
def _load_song_labels(label_col: str) -> dict[str, str]:
    with get_session() as session:
        rows = session.execute(text(
            f"SELECT title, {label_col} FROM songs WHERE {label_col} IS NOT NULL"
        )).all()
    return {r[0]: r[1] for r in rows}


def _load_product_labels(label_col: str) -> dict[str, str]:
    with get_session() as session:
        rows = session.execute(text(
            f"SELECT name, {label_col} FROM products WHERE {label_col} IS NOT NULL"
        )).all()
    return {r[0]: r[1] for r in rows}


def _label_counts(labels: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lab in labels.values():
        counts[lab] = counts.get(lab, 0) + 1
    return counts


# Recall@K
def _metrics_at_k_for_query(
    query_id: str,
    query_label: str,
    retrieved_ids: list[str],
    labels: dict[str, str],
    counts: dict[str, int],
    k: int,
) -> tuple[float, float, float] | None:
    relevant_total = counts.get(query_label, 0) - 1
    if relevant_total <= 0:
        return None
    retrieved_top_k = [rid for rid in retrieved_ids[:k] if rid != query_id]
    if not retrieved_top_k:
        return None
    tp = sum(1 for rid in retrieved_top_k if labels.get(rid) == query_label)
    recall = tp / relevant_total
    precision = tp / len(retrieved_top_k)
    recall_norm = tp / min(k, relevant_total)
    return recall, precision, recall_norm


def _f1(recall: float, precision: float) -> float:
    if recall + precision == 0.0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def _aggregate(triples: list[tuple[float, float, float]]) -> dict:
    if not triples:
        return {"recall_at_k": None, "precision_at_k": None, "recall_norm_at_k": None,
                "f1_at_k": None, "evaluated": 0}
    recall = sum(r for r, _, _ in triples) / len(triples)
    precision = sum(p for _, p, _ in triples) / len(triples)
    recall_norm = sum(rn for _, _, rn in triples) / len(triples)
    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "recall_norm_at_k": round(recall_norm, 4),
        "f1_at_k": round(_f1(recall, precision), 4),
        "evaluated": len(triples),
    }


def _eval_engine_text(
    sample_queries: list[tuple[str, str, str]],   # (query_id, query_text, query_label)
    search_fn,
    engine: str,
    k: int,
    labels: dict[str, str],
    counts: dict[str, int],
) -> dict:
    pairs: list[tuple[float, float, float]] = []
    for qid, qtext, qlabel in sample_queries:
        try:
            resp = search_fn(qtext, engine, k * 2)
        except Exception:
            continue
        retrieved_ids = [r.get("id") or r.get("title") or r.get("name")
                         for r in resp.get("results", [])]
        retrieved_ids = [str(x) for x in retrieved_ids if x is not None]
        m = _metrics_at_k_for_query(qid, qlabel, retrieved_ids, labels, counts, k)
        if m is not None:
            pairs.append(m)
    return _aggregate(pairs)


def _eval_engine_file(
    sample_queries: list[tuple[str, Path, str]],  # (query_id, file_path, label)
    search_fn,
    engine: str,
    k: int,
    labels: dict[str, str],
    counts: dict[str, int],
) -> dict:
    pairs: list[tuple[float, float, float]] = []
    for qid, qpath, qlabel in sample_queries:
        try:
            resp = search_fn(qpath, engine, k * 2)
        except Exception:
            continue
        retrieved_ids = [r.get("id") or r.get("title") or r.get("name")
                         for r in resp.get("results", [])]
        retrieved_ids = [str(x) for x in retrieved_ids if x is not None]
        m = _metrics_at_k_for_query(qid, qlabel, retrieved_ids, labels, counts, k)
        if m is not None:
            pairs.append(m)
    return _aggregate(pairs)


# Sampling de queries del DB
def _short_query(text_full: str, max_words: int = 6) -> str:
    words = text_full.split()
    return " ".join(words[:max_words])


def _sample_song_text_queries(n: int, label_col: str, seed: int):
    with get_session() as session:
        rows = session.execute(text(
            f"SELECT id, lyrics_text, {label_col} FROM songs "
            f"WHERE lyrics_text IS NOT NULL AND {label_col} IS NOT NULL"
        )).all()
    if not rows:
        return [], {}, {}
    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    labels = {str(r[0]): r[2] for r in rows}
    counts = _label_counts(labels)
    queries = [(str(r[0]), _short_query(r[1]), r[2]) for r in sampled]
    return queries, labels, counts


def _sample_song_audio_queries(n: int, label_col: str, seed: int):
    with get_session() as session:
        rows = session.execute(text(
            f"SELECT id, audio_path, {label_col} FROM songs "
            f"WHERE audio_path IS NOT NULL AND {label_col} IS NOT NULL"
        )).all()
    rows = [r for r in rows if r[1] and Path(r[1]).exists()]
    if not rows:
        return [], {}, {}
    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    labels = {str(r[0]): r[2] for r in rows}
    counts = _label_counts(labels)
    queries = [(str(r[0]), Path(r[1]), r[2]) for r in sampled]
    return queries, labels, counts


def _sample_product_text_queries(n: int, label_col: str, seed: int):
    with get_session() as session:
        rows = session.execute(text(
            f"SELECT id, description, {label_col} FROM products "
            f"WHERE description IS NOT NULL AND {label_col} IS NOT NULL"
        )).all()
    if not rows:
        return [], {}, {}
    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    labels = {str(r[0]): r[2] for r in rows}
    counts = _label_counts(labels)
    queries = [(str(r[0]), r[1][:200], r[2]) for r in sampled]
    return queries, labels, counts


def _sample_product_image_queries(n: int, label_col: str, seed: int):
    with get_session() as session:
        rows = session.execute(text(
            f"SELECT id, image_path, {label_col} FROM products "
            f"WHERE image_path IS NOT NULL AND {label_col} IS NOT NULL"
        )).all()
    rows = [r for r in rows if r[1] and Path(r[1]).exists()]
    if not rows:
        return [], {}, {}
    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    labels = {str(r[0]): r[2] for r in rows}
    counts = _label_counts(labels)
    queries = [(str(r[0]), Path(r[1]), r[2]) for r in sampled]
    return queries, labels, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@K Fase 4")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--music-label", default="genre",
                        choices=["artist", "genre"])
    parser.add_argument("--fashion-label", default="subcategory",
                        choices=["category", "subcategory"])
    parser.add_argument("--out-json", type=str, default="recall_fase4.json")
    args = parser.parse_args()

    report: dict = {"k": args.k, "n_queries": args.queries, "seed": args.seed,
                    "music_label": args.music_label,
                    "fashion_label": args.fashion_label}

    print(f"\n[music/lyrics] ground truth = {args.music_label}")
    qs, labels, counts = _sample_song_text_queries(args.queries, args.music_label, args.seed)
    if not qs:
        print("  (skip) no hay ground truth disponible")
    else:
        report["music_lyrics"] = {}
        for engine in ("spimi", "gin", "gist"):
            res = _eval_engine_text(qs, search_music_lyrics, engine, args.k, labels, counts)
            report["music_lyrics"][engine] = res
            print(f"  {engine:10s}  recall@{args.k} = {res['recall_at_k']}  "
                  f"precision@{args.k} = {res['precision_at_k']}  "
                  f"recall_norm@{args.k} = {res['recall_norm_at_k']}  "
                  f"f1@{args.k} = {res['f1_at_k']}  "
                  f"(evaluadas {res['evaluated']})")

    print(f"\n[music/audio] ground truth = {args.music_label}")
    qs, labels, counts = _sample_song_audio_queries(args.queries, args.music_label, args.seed)
    if not qs:
        print("  (skip) no hay ground truth disponible o paths invalidos")
    else:
        report["music_audio"] = {}
        for engine in ("spimi", "pgvector"):
            res = _eval_engine_file(qs, search_music_audio, engine, args.k, labels, counts)
            report["music_audio"][engine] = res
            print(f"  {engine:10s}  recall@{args.k} = {res['recall_at_k']}  "
                  f"precision@{args.k} = {res['precision_at_k']}  "
                  f"recall_norm@{args.k} = {res['recall_norm_at_k']}  "
                  f"f1@{args.k} = {res['f1_at_k']}  "
                  f"(evaluadas {res['evaluated']})")

    print(f"\n[fashion/description] ground truth = {args.fashion_label}")
    qs, labels, counts = _sample_product_text_queries(args.queries, args.fashion_label, args.seed)
    if not qs:
        print("  (skip) no hay ground truth disponible")
    else:
        report["fashion_desc"] = {}
        for engine in ("spimi", "gin", "gist"):
            res = _eval_engine_text(qs, search_fashion_desc, engine, args.k, labels, counts)
            report["fashion_desc"][engine] = res
            print(f"  {engine:10s}  recall@{args.k} = {res['recall_at_k']}  "
                  f"precision@{args.k} = {res['precision_at_k']}  "
                  f"recall_norm@{args.k} = {res['recall_norm_at_k']}  "
                  f"f1@{args.k} = {res['f1_at_k']}  "
                  f"(evaluadas {res['evaluated']})")

    print(f"\n[fashion/image] ground truth = {args.fashion_label}")
    qs, labels, counts = _sample_product_image_queries(args.queries, args.fashion_label, args.seed)
    if not qs:
        print("  (skip) no hay ground truth disponible o paths invalidos")
    else:
        report["fashion_image"] = {}
        for engine in ("spimi", "pgvector"):
            res = _eval_engine_file(qs, search_fashion_image, engine, args.k, labels, counts)
            report["fashion_image"][engine] = res
            print(f"  {engine:10s}  recall@{args.k} = {res['recall_at_k']}  "
                  f"precision@{args.k} = {res['precision_at_k']}  "
                  f"recall_norm@{args.k} = {res['recall_norm_at_k']}  "
                  f"f1@{args.k} = {res['f1_at_k']}  "
                  f"(evaluadas {res['evaluated']})")

    clear_caches()
    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReporte: {args.out_json}")


if __name__ == "__main__":
    main()