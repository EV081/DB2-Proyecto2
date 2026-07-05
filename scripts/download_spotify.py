from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

KAGGLE_DATASET = os.environ.get(
    "SPOTIFY_KAGGLE_DATASET",
    "zinasakr/40k-songs-with-audio-features-and-lyrics",
)


def _kaggle_creds_ok() -> bool:
    home = Path.home() / ".kaggle"
    return (
        (home / "kaggle.json").exists()
        or (home / "access_token").exists()
        or bool(os.environ.get("KAGGLE_API_TOKEN"))
        or (bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY")))
    )


def _safe_stem(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (raw or "").strip())
    return s.strip("_")


def _pick(row: dict, *candidates: str) -> str:
    for c in candidates:
        if c in row and row[c]:
            return str(row[c])
    return ""


def run(out_dir: Path, limit: int | None, language: str | None = "en") -> dict:
    if not _kaggle_creds_ok():
        sys.exit(
            "Faltan credenciales de Kaggle. Acepta cualquiera de:\n"
            "  - ~/.kaggle/access_token (token nuevo KGAT_...)\n"
            "  - ~/.kaggle/kaggle.json   (formato clásico)\n"
            "  - env KAGGLE_API_TOKEN, o KAGGLE_USERNAME + KAGGLE_KEY\n"
            "Crea/regenera token en https://www.kaggle.com/settings (sección API)."
        )
    try:
        import kagglehub
    except ImportError:
        sys.exit("Instala con: pip install kagglehub")

    print(f"Descargando '{KAGGLE_DATASET}'...")
    src_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"  cache HF kagglehub: {src_dir}")

    csv_files = list(src_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No se encontró CSV en {src_dir}")
    csv_path = max(csv_files, key=lambda p: p.stat().st_size)
    print(f"  CSV: {csv_path.name} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    lyrics_dir = out_dir / "lyrics"
    lyrics_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"
    written = 0
    n_lang_skipped = 0
    seen: set[str] = set()

    with csv_path.open(encoding="utf-8", errors="ignore") as f, \
         meta_path.open("w", encoding="utf-8", newline="") as mf:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(mf, fieldnames=["stem", "title", "artist", "genre"])
        writer.writeheader()
        for i, row in enumerate(reader):
            if limit and written >= limit:
                break
            lyrics = _pick(row, "lyrics", "text", "seq")
            if not lyrics or len(lyrics) < 20:
                continue
            if language:
                lang = (_pick(row, "language", "lang") or "").strip().lower()
                if lang and lang != language:
                    n_lang_skipped += 1
                    continue
            title = _pick(row, "song_name", "track_name", "title", "song", "name")
            artist = _pick(row, "track_artist", "artist", "artists", "artist_name")
            genre = _pick(row, "playlist_genre", "genre", "track_genre", "genres")
            stem = _safe_stem(f"{artist}_{title}") or f"song_{i:06d}"
            base, k = stem, 1
            while stem in seen:
                stem = f"{base}_{k}"
                k += 1
            seen.add(stem)
            (lyrics_dir / f"{stem}.txt").write_text(lyrics, encoding="utf-8")
            writer.writerow({"stem": stem, "title": title, "artist": artist, "genre": genre})
            written += 1

    if language:
        print(f"OK: {written} canciones ({language}), {n_lang_skipped} descartadas por idioma -> {lyrics_dir}, meta -> {meta_path}")
    else:
        print(f"OK: {written} canciones (todos los idiomas) -> {lyrics_dir}, meta -> {meta_path}")
    return {"lyrics_dir": str(lyrics_dir), "metadata_csv": str(meta_path), "n": written, "language": language}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Descarga Spotify lyrics (Kaggle)")
    p.add_argument("--out-dir", type=Path, default=Path("data/spotify"))
    p.add_argument("--limit", type=int, default=None,
                   help="Limitar nro de canciones (debug). Default: todas.")
    p.add_argument("--language", default="en",
                   help="Filtrar por idioma (ISO 639-1). Vacio o 'all' desactiva. Default: en.")
    args = p.parse_args()
    lang = args.language if args.language and args.language.lower() != "all" else None
    run(out_dir=args.out_dir, limit=args.limit, language=lang)
