from __future__ import annotations
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default: fma_large (~93 GB, 106k tracks). Se corta con --limit durante extraccion.
URL = os.environ.get("FMA_URL", "https://os.unil.cloud.switch.ch/fma/fma_large.zip")
SHA1 = os.environ.get("FMA_SHA1", "497109f4dd721066b5ce5e5f250ec604dc78939e")
DEFAULT_LIMIT = int(os.environ.get("FMA_LIMIT", "40000"))

META_URL = os.environ.get("FMA_METADATA_URL", "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip")
META_SHA1 = os.environ.get("FMA_METADATA_SHA1", "f0df49ffe5f2a6008d7dc83c6915b31835dfe733")


def _sha1(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def _curl(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["curl", "-L", "--fail", "-C", "-", "-o", str(dst), url]
    )


def _fetch_metadata(out_dir: Path, skip_sha1: bool) -> Path:
    meta_zip = out_dir / "fma_metadata.zip"
    tracks_csv = out_dir / "fma_metadata" / "tracks.csv"
    if not tracks_csv.exists():
        if not meta_zip.exists():
            print(f"Descargando {META_URL} -> {meta_zip} (~340MB)...")
            _curl(META_URL, meta_zip)
        if not skip_sha1:
            got = _sha1(meta_zip)
            if got != META_SHA1:
                print(f"  WARN metadata SHA1 mismatch: got {got}, continuo igual")
        print(f"Descomprimiendo metadata en {out_dir} ...")
        with zipfile.ZipFile(meta_zip) as zf:
            zf.extractall(out_dir)
    if not tracks_csv.exists():
        raise SystemExit(f"No encontre {tracks_csv} tras descomprimir")
    return tracks_csv


def _emit_flat_metadata_csv(tracks_csv: Path, out_csv: Path) -> int:
    import pandas as pd
    print(f"Generando {out_csv} desde {tracks_csv} ...")
    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1], low_memory=False)
    title = df[("track", "title")]
    artist = df[("artist", "name")]
    genre = df[("track", "genre_top")]
    rows = []
    for track_id, t, a, g in zip(df.index, title, artist, genre):
        stem = f"{int(track_id):06d}"
        rows.append({
            "stem": stem,
            "title": ("" if pd.isna(t) else str(t)).strip(),
            "artist": ("" if pd.isna(a) else str(a)).strip(),
            "genre": ("" if pd.isna(g) else str(g)).strip(),
        })
    out_df = pd.DataFrame(rows, columns=["stem", "title", "artist", "genre"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"  {len(out_df)} tracks -> {out_csv}")
    return len(out_df)


def _selective_extract(zip_path: Path, out_dir: Path, extract_root: Path, limit: int | None) -> int:

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        mp3s = sorted(n for n in zf.namelist() if n.endswith(".mp3"))
        if limit is not None and limit < len(mp3s):
            to_extract = mp3s[:limit]
            print(f"Extrayendo {len(to_extract)}/{len(mp3s)} mp3s (limit={limit})...")
        else:
            to_extract = mp3s
            print(f"Extrayendo {len(to_extract)} mp3s...")
        for i, name in enumerate(to_extract, 1):
            target = out_dir / name
            if not target.exists():
                zf.extract(name, out_dir)
            if i % 5000 == 0:
                print(f"  [{i}/{len(to_extract)}]")
    return len(to_extract)


def run(out_dir: Path, skip_sha1: bool = False, limit: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_name = URL.rsplit("/", 1)[-1]                     # p.ej. fma_large.zip
    corpus_name = zip_name.rsplit(".", 1)[0]              # p.ej. fma_large
    zip_path = out_dir / zip_name
    extract_root = out_dir / corpus_name
    flat_dir = out_dir / f"{corpus_name}_flat"

    if not zip_path.exists():
        print(f"Descargando {URL} -> {zip_path} ...")
        _curl(URL, zip_path)
    else:
        print(f"[skip] zip ya presente: {zip_path}")

    if not skip_sha1:
        size_gb = zip_path.stat().st_size / (1024 ** 3)
        print(f"Validando SHA1 sobre {size_gb:.1f} GB (puede tomar varios minutos)...")
        got = _sha1(zip_path)
        if got != SHA1:
            raise SystemExit(f"SHA1 mismatch: got {got}, expected {SHA1}")
        print("  OK")

    # Extraccion selectiva: solo `limit` mp3s si se especifica
    if not extract_root.exists() or not any(extract_root.rglob("*.mp3")):
        _selective_extract(zip_path, out_dir, extract_root, limit)
    else:
        n_ya = sum(1 for _ in extract_root.rglob("*.mp3"))
        print(f"[skip] ya descomprimido: {extract_root} ({n_ya} mp3s)")

    flat_dir.mkdir(exist_ok=True)
    n_linked = 0
    for mp3 in extract_root.rglob("*.mp3"):
        link = flat_dir / mp3.name
        if not link.exists():
            link.symlink_to(mp3.resolve())
            n_linked += 1
    n_total = sum(1 for _ in flat_dir.glob("*.mp3"))
    print(f"Aplanado: {n_linked} symlinks nuevos, {n_total} total en {flat_dir}")

    tracks_csv = _fetch_metadata(out_dir, skip_sha1)
    flat_meta = flat_dir / "metadata.csv"
    n_meta = _emit_flat_metadata_csv(tracks_csv, flat_meta)

    return {
        "audio_dir": str(flat_dir),
        "n_tracks": n_total,
        "metadata_csv": str(flat_meta),
        "n_metadata": n_meta,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Descarga FMA (large por defecto) -> data/")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--skip-sha1", action="store_true",
                   help="No verificar SHA1 (ahorra minutos en fma_large)")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help=f"Max mp3s a extraer. 0 = sin limite. Default: {DEFAULT_LIMIT}")
    args = p.parse_args()
    if shutil.which("curl") is None:
        sys.exit("Necesitas `curl` instalado.")
    limit = args.limit if args.limit > 0 else None
    run(out_dir=args.out_dir, skip_sha1=args.skip_sha1, limit=limit)
