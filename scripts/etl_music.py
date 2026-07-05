from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.db.storage import (
    _song_params,
    ensure_emb_column,
    hist_to_dense,
    reset_table,
    save_codebook,
    save_songs_batch,
)
from src.engine.audio_pipeline import index_audio_corpus
from src.engine.text_pipeline import index_text_corpus
from src.extraction.audio_mfcc import extract_mfcc_features
from src.extraction.text_tfidf import extract_tfidf_features


def _collect_lyrics(lyrics_dir: Path | None, lyrics_ext: str) -> list[tuple[str, Path]]:
    if lyrics_dir is None or not lyrics_dir.exists():
        return []
    return sorted((p.stem, p) for p in lyrics_dir.glob(f"*{lyrics_ext}"))


def _collect_audio(audio_dir: Path | None, audio_exts: tuple[str, ...]) -> list[tuple[str, Path]]:
    if audio_dir is None or not audio_dir.exists():
        return []
    seen: dict[str, Path] = {}
    for ext in audio_exts:
        for p in audio_dir.glob(f"*{ext}"):
            seen.setdefault(p.stem, p)
    return sorted(seen.items())


def _aggregate_lyrics_hist(
    lyrics_path: Path, codebook: set[str],
) -> tuple[dict[str, int], str]:
    chunks_tf = extract_tfidf_features(str(lyrics_path))
    agg: dict[str, int] = {}
    for tf in chunks_tf:
        for term, count in tf.items():
            if term in codebook:
                agg[term] = agg.get(term, 0) + count
    raw_text = lyrics_path.read_text(encoding="utf-8", errors="ignore")
    return agg, raw_text


def _load_metadata_csv(paths: list[Path] | Path | None) -> dict[str, dict]:
    if paths is None:
        return {}
    if isinstance(paths, Path):
        paths = [paths]
    out: dict[str, dict] = {}
    for path in paths:
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = (row.get("stem") or "").strip()
                if not stem:
                    continue
                out[stem] = {
                    "title": (row.get("title") or stem).strip(),
                    "artist": (row.get("artist") or "").strip() or None,
                    "genre": (row.get("genre") or "").strip() or None,
                }
    return out


def _audio_hist(
    audio_path: Path, vq, n_mfcc: int,
    cache_dir: Path | None = None,
) -> dict[str, int]:
    mfcc = None
    if cache_dir is not None:
        cached = cache_dir / f"{audio_path.stem}.npy"
        if cached.exists():
            mfcc = np.load(cached, mmap_mode=None)
    if mfcc is None:
        try:
            mfcc = extract_mfcc_features(str(audio_path), n_mfcc=n_mfcc)
        except Exception as e:
            print(f"  [SKIP audio persist] {audio_path.name}: {e.__class__.__name__}: {e}")
            return {}
    if mfcc.size == 0:
        return {}
    counts = vq.histogram(mfcc)
    return {f"a_{i:04d}": int(c) for i, c in enumerate(counts) if c > 0}


def run(
    lyrics_dir: Path | None, audio_dir: Path | None,
    codebook_text: int, codebook_audio: int,
    index_dir: Path,
    lyrics_ext: str, audio_exts: tuple[str, ...],
    n_mfcc: int, reset: bool,
    metadata_csv: list[Path] | Path | None = None,
    max_audio_samples: int = 50_000,
    codebooks_dir: Path = Path("codebooks"),
    app_name: str = "music",
) -> dict:
    if lyrics_dir is None and audio_dir is None:
        raise SystemExit("Debes pasar al menos --lyrics-dir o --audio-dir")

    metadata = _load_metadata_csv(metadata_csv)
    lyrics_items = _collect_lyrics(lyrics_dir, lyrics_ext)
    audio_items = _collect_audio(audio_dir, audio_exts)
    if not lyrics_items and not audio_items:
        raise SystemExit(
            f"No se encontraron archivos en {lyrics_dir} o {audio_dir}."
        )
    print(f"Canciones: {len(lyrics_items)} con lyrics, {len(audio_items)} con audio")

    index_dir.mkdir(parents=True, exist_ok=True)
    codebooks_dir.mkdir(parents=True, exist_ok=True)
    text_index_dir = index_dir / "text"
    audio_index_dir = index_dir / "audio"

    lyrics_paths = [str(lp) for _, lp in lyrics_items]
    audio_paths = [str(ap) for _, ap in audio_items]

    text_bag: list[str] = []
    text_codebook: set[str] = set()
    if lyrics_paths:
        t0 = time.perf_counter()
        print("[1/4] Codebook textual + indice SPIMI...")
        text_idx, text_codebook = index_text_corpus(
            file_paths=lyrics_paths,
            codebook_size=codebook_text,
            index_dir=text_index_dir,
        )
        text_idx.close()
        text_bag = sorted(text_codebook)
        print(f"      codebook texto: {len(text_bag)} palabras  "
              f"({time.perf_counter() - t0:.1f}s)")
    else:
        print("[1/4] (skip) sin lyrics_dir, no se construye codebook textual")

    centroids: list = []
    audio_keys: list[str] = []
    audio_vq = None
    centroids_path = None
    if audio_paths:
        t0 = time.perf_counter()
        print("[2/4] Codebook acustico + indice SPIMI...")
        audio_idx, audio_vq = index_audio_corpus(
            file_paths=audio_paths,
            codebook_size=codebook_audio,
            index_dir=audio_index_dir,
            n_mfcc=n_mfcc,
            max_samples=max_audio_samples,
        )
        audio_idx.close()
        centroids = list(audio_vq.centroids)
        audio_keys = [f"a_{i:04d}" for i in range(len(centroids))]
        centroids_path = codebooks_dir / f"{app_name}_audio_centroids.npy"
        np.save(centroids_path, np.array(centroids))
        print(f"      codebook audio: {len(centroids)} centroides  "
              f"({time.perf_counter() - t0:.1f}s)")
    else:
        print("[2/4] (skip) sin audio_dir, no se construye codebook acustico")

    print("[3/4] Ajustando schema y limpiando tabla songs...")
    if reset:
        reset_table("songs")
    if centroids:
        ensure_emb_column("songs", "audio_emb", len(centroids))

    total_items = len(lyrics_items) + len(audio_items)
    print(f"[4/4] Persistiendo {total_items} canciones en BD...")
    audio_cache = audio_index_dir / "_features"
    if not audio_cache.exists():
        audio_cache = None
    inserted = 0
    batch: list[dict] = []
    batch_size = 500
    t0 = time.perf_counter()
    log_every = max(500, total_items // 40)

    def _flush(force: bool = False) -> None:
        nonlocal inserted, batch
        if batch and (force or len(batch) >= batch_size):
            inserted += save_songs_batch(batch)
            batch = []

    # Loop 1: canciones con lyrics (Spotify)
    for i, (stem, lp) in enumerate(lyrics_items, 1):
        lyrics_hist, lyrics_text = _aggregate_lyrics_hist(lp, text_codebook)
        meta = metadata.get(stem, {})
        batch.append(_song_params(
            title=meta.get("title") or stem,
            artist=meta.get("artist"),
            genre=meta.get("genre"),
            lyrics_path=str(lp),
            audio_path=None,
            lyrics_text=lyrics_text,
            lyrics_hist=lyrics_hist,
            audio_hist={},
            audio_emb=None,
            metadata={"source": "etl_music"},
        ))
        _flush()
        if i % log_every == 0:
            print(f"  [persist lyrics] {i}/{len(lyrics_items)}  ({time.perf_counter() - t0:.1f}s)")
    _flush(force=True)

    # Loop 2: canciones con audio (FMA)
    for i, (stem, ap) in enumerate(audio_items, 1):
        if audio_vq is None:
            break
        audio_hist = _audio_hist(ap, audio_vq, n_mfcc=n_mfcc, cache_dir=audio_cache)
        audio_emb = hist_to_dense(audio_hist, audio_keys) if audio_keys else None
        meta = metadata.get(stem, {})
        batch.append(_song_params(
            title=meta.get("title") or stem,
            artist=meta.get("artist"),
            genre=meta.get("genre"),
            lyrics_path=None,
            audio_path=str(ap),
            lyrics_text=None,
            lyrics_hist={},
            audio_hist=audio_hist,
            audio_emb=audio_emb,
            metadata={"source": "etl_music"},
        ))
        _flush()
        if i % log_every == 0:
            print(f"  [persist audio] {i}/{len(audio_items)}  ({time.perf_counter() - t0:.1f}s)")
    _flush(force=True)

    print(f"      insertadas {inserted}/{total_items} en {time.perf_counter() - t0:.1f}s")

    if text_bag:
        bag_path = codebooks_dir / f"{app_name}_text_bag.json"
        bag_path.write_text(json.dumps(text_bag, ensure_ascii=False), encoding="utf-8")
        save_codebook(
            app=app_name, modality="text",
            codebook_size=len(text_bag),
            bag_of_words=text_bag,
            index_dir=str(text_index_dir),
        )
    if centroids:
        save_codebook(
            app=app_name, modality="audio",
            codebook_size=len(centroids),
            centroids_path=str(centroids_path) if centroids_path else None,
            index_dir=str(audio_index_dir),
        )

    return {
        "n_items": total_items,
        "with_lyrics": len(lyrics_items),
        "with_audio": len(audio_items),
        "inserted": inserted,
        "codebook_text": len(text_bag),
        "codebook_audio": len(centroids),
        "text_index_dir": str(text_index_dir) if text_bag else None,
        "audio_index_dir": str(audio_index_dir) if centroids else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Musica (App 2)")
    parser.add_argument("--lyrics-dir", type=Path, default=None,
                        help="Directorio con letras (.txt). Opcional.")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="Directorio con audio. Opcional.")
    parser.add_argument("--codebook-text", type=int, default=1000)
    parser.add_argument("--codebook-audio", type=int, default=500)
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/music"))
    parser.add_argument("--codebooks-dir", type=Path, default=Path("codebooks"))
    parser.add_argument("--app-name", default="music")
    parser.add_argument("--lyrics-ext", default=".txt")
    parser.add_argument(
        "--audio-exts", default=".mp3,.wav,.flac,.m4a,.ogg",
        help="Coma-separated lista de extensiones de audio aceptadas",
    )
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--max-audio-samples", type=int, default=50_000,
                        help="Subsample de vectores MFCC para entrenar KMeans (0 = sin limite)")
    parser.add_argument("--reset", action="store_true",
                        help="TRUNCATE songs antes de insertar")
    parser.add_argument("--metadata-csv", type=Path, action="append", default=None,
                        help="CSV opcional con columnas stem,title,artist,genre. Repetible.")
    args = parser.parse_args()

    result = run(
        lyrics_dir=args.lyrics_dir,
        audio_dir=args.audio_dir,
        codebook_text=args.codebook_text,
        codebook_audio=args.codebook_audio,
        index_dir=args.index_dir,
        lyrics_ext=args.lyrics_ext,
        audio_exts=tuple(args.audio_exts.split(",")),
        n_mfcc=args.n_mfcc,
        reset=args.reset,
        metadata_csv=args.metadata_csv,
        max_audio_samples=args.max_audio_samples,
        codebooks_dir=args.codebooks_dir,
        app_name=args.app_name,
    )
    print("\n" + json.dumps(result, indent=2))