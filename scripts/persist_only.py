from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.db.storage import (
    ensure_emb_column,
    hist_to_dense,
    reset_table,
    save_codebook,
    save_song,
)
from src.engine.audio_pipeline import _extract_mfcc_worker  # worker top-level
from src.extraction.text_tfidf import extract_tfidf_features
from src.ml.quantizer import VectorQuantizer


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


def _load_text_codebook(index_dir: Path, codebooks_dir: Path, app: str) -> list[str] | None:
    bag_path = codebooks_dir / f"{app}_text_bag.json"
    if bag_path.exists():
        return sorted(json.loads(bag_path.read_text(encoding="utf-8")))
    vocab_path = index_dir / "text" / "final" / "vocab.json"
    if not vocab_path.exists():
        return None
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    return sorted(data.keys())


def _load_audio_centroids(codebooks_dir: Path, app: str) -> tuple[np.ndarray, Path] | tuple[None, None]:
    p = codebooks_dir / f"{app}_audio_centroids.npy"
    if not p.exists():
        return None, None
    return np.load(p), p


def run(
    app: str,
    lyrics_dir: Path | None,
    audio_dir: Path | None,
    index_dir: Path,
    metadata_csv: list[Path] | Path | None,
    lyrics_ext: str,
    audio_exts: tuple[str, ...],
    reset: bool,
    n_mfcc: int = 13,
    codebooks_dir: Path = Path("codebooks"),
) -> dict:
    text_bag = _load_text_codebook(index_dir, codebooks_dir, app)
    centroids, centroids_path = _load_audio_centroids(codebooks_dir, app)
    if text_bag is None and centroids is None:
        sys.exit(
            f"No encontre codebooks en {codebooks_dir} ni vocab.json en {index_dir}/text/final. "
            "Necesitas correr el ETL completo (etl_music.py) al menos una vez."
        )
    text_codebook = set(text_bag) if text_bag else set()
    audio_vq = VectorQuantizer(centroids) if centroids is not None else None
    audio_keys = (
        [f"a_{i:04d}" for i in range(len(centroids))] if centroids is not None else []
    )
    if text_bag:
        print(f"  codebook texto cargado: {len(text_bag)} palabras")
    if centroids is not None:
        print(f"  codebook audio cargado: {len(centroids)} centroides")

    lyrics_items = _collect_lyrics(lyrics_dir, lyrics_ext)
    audio_items = _collect_audio(audio_dir, audio_exts)
    if not lyrics_items and not audio_items:
        sys.exit(f"No hay archivos en {lyrics_dir} ni {audio_dir}")
    total_items = len(lyrics_items) + len(audio_items)
    print(f"  items: {len(lyrics_items)} con lyrics, {len(audio_items)} con audio")

    metadata = _load_metadata_csv(metadata_csv)

    if reset:
        print("  TRUNCATE songs...")
        reset_table("songs")
    if centroids is not None:
        ensure_emb_column("songs", "audio_emb", len(centroids))

    # Pre-computar histogramas de audio en paralelo (lo mas lento)
    audio_hists: dict[str, dict[str, int]] = {}
    if audio_vq is not None and audio_items:
        args = [(str(ap), n_mfcc) for _, ap in audio_items]
        n_workers = max(1, (cpu_count() or 1) - 1)
        log_every = max(50, len(args) // 20)
        print(
            f"  [audio quant] {len(args)} archivos en {n_workers} workers paralelos"
        )
        n_done = 0
        t0 = time.perf_counter()
        with Pool(n_workers) as pool:
            for fp_str, mfcc, err in pool.imap_unordered(
                _extract_mfcc_worker, args, chunksize=10,
            ):
                n_done += 1
                stem = Path(fp_str).stem
                if err is not None:
                    print(f"    [SKIP] {Path(fp_str).name}: {err}")
                    continue
                if mfcc is None or mfcc.size == 0:
                    continue
                counts = audio_vq.histogram(mfcc)
                audio_hists[stem] = {
                    f"a_{i:04d}": int(c) for i, c in enumerate(counts) if c > 0
                }
                if n_done % log_every == 0:
                    print(f"    [audio quant] {n_done}/{len(args)}")
        print(f"  audio quant terminado en {time.perf_counter() - t0:.1f}s")

    # Insertar a BD (sequential porque las inserts a Postgres son rapidas)
    print(f"  insertando {total_items} canciones a BD...")
    inserted = 0
    t0 = time.perf_counter()

    # Loop 1: lyrics (Spotify)
    for stem, lp in lyrics_items:
        try:
            lyrics_hist, lyrics_text = _aggregate_lyrics_hist(lp, text_codebook)
        except Exception as e:
            print(f"  [SKIP lyrics] {lp.name}: {e.__class__.__name__}: {e}")
            lyrics_hist, lyrics_text = {}, None
        meta = metadata.get(stem, {})
        try:
            save_song(
                title=meta.get("title") or stem,
                artist=meta.get("artist"),
                genre=meta.get("genre"),
                lyrics_path=str(lp),
                audio_path=None,
                lyrics_text=lyrics_text,
                lyrics_hist=lyrics_hist,
                audio_hist={},
                audio_emb=None,
                metadata={"source": "persist_only"},
            )
            inserted += 1
        except Exception as e:
            print(f"  [SKIP insert lyrics] {stem}: {e.__class__.__name__}: {e}")

    # Loop 2: audio (FMA)
    for stem, ap in audio_items:
        if audio_vq is None:
            break
        audio_hist = audio_hists.get(stem, {})
        audio_emb = hist_to_dense(audio_hist, audio_keys) if audio_keys else None
        meta = metadata.get(stem, {})
        try:
            save_song(
                title=meta.get("title") or stem,
                artist=meta.get("artist"),
                genre=meta.get("genre"),
                lyrics_path=None,
                audio_path=str(ap),
                lyrics_text=None,
                lyrics_hist={},
                audio_hist=audio_hist,
                audio_emb=audio_emb,
                metadata={"source": "persist_only"},
            )
            inserted += 1
        except Exception as e:
            print(f"  [SKIP insert audio] {stem}: {e.__class__.__name__}: {e}")

    print(f"  insertadas {inserted}/{total_items} en {time.perf_counter() - t0:.1f}s")

    # Registrar codebooks (si una corrida previa murio en [4/4], esto faltaba)
    if text_bag:
        save_codebook(
            app=app, modality="text",
            codebook_size=len(text_bag),
            bag_of_words=text_bag,
            index_dir=str(index_dir / "text"),
        )
    if centroids is not None:
        save_codebook(
            app=app, modality="audio",
            codebook_size=len(centroids),
            centroids_path=str(centroids_path),
            index_dir=str(index_dir / "audio"),
        )

    return {
        "items": total_items,
        "with_lyrics": len(lyrics_items),
        "with_audio": len(audio_items),
        "inserted": inserted,
        "codebook_text": len(text_bag) if text_bag else 0,
        "codebook_audio": len(centroids) if centroids is not None else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Persiste a BD reusando codebooks ya entrenados (rapido)"
    )
    parser.add_argument("--app", default="music",
                        help="Etiqueta para save_codebook (ej. music)")
    parser.add_argument("--lyrics-dir", type=Path, default=None)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--codebooks-dir", type=Path, default=Path("codebooks"))
    parser.add_argument("--metadata-csv", type=Path, action="append", default=None)
    parser.add_argument("--lyrics-ext", default=".txt")
    parser.add_argument("--audio-exts", default=".mp3,.wav,.flac,.m4a,.ogg")
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--reset", action="store_true",
                        help="TRUNCATE songs antes de insertar")
    args = parser.parse_args()
    if args.lyrics_dir is None and args.audio_dir is None:
        parser.error("Debes pasar --lyrics-dir y/o --audio-dir")

    result = run(
        app=args.app,
        lyrics_dir=args.lyrics_dir,
        audio_dir=args.audio_dir,
        index_dir=args.index_dir,
        metadata_csv=args.metadata_csv,
        lyrics_ext=args.lyrics_ext,
        audio_exts=tuple(args.audio_exts.split(",")),
        reset=args.reset,
        n_mfcc=args.n_mfcc,
        codebooks_dir=args.codebooks_dir,
    )
    print("\n" + json.dumps(result, indent=2))
