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
    _product_params,
    ensure_emb_column,
    hist_to_dense,
    reset_table,
    save_codebook,
    save_products_batch,
)
from src.engine.image_pipeline import index_image_corpus
from src.engine.text_pipeline import index_text_corpus
from src.extraction.image_sift import extract_sift_features
from src.extraction.text_tfidf import extract_tfidf_features


def _pair_files(
    images_dir: Path, descs_dir: Path,
    image_exts: tuple[str, ...], desc_ext: str,
) -> list[tuple[str, Path, Path]]:
    images: dict[str, Path] = {}
    for ext in image_exts:
        for p in images_dir.glob(f"*{ext}"):
            images.setdefault(p.stem, p)
    descs = {p.stem: p for p in descs_dir.glob(f"*{desc_ext}")}
    common = sorted(set(images) & set(descs))
    return [(stem, images[stem], descs[stem]) for stem in common]


def _aggregate_desc_hist(
    desc_path: Path, codebook: set[str],
) -> tuple[dict[str, int], str]:
    chunks_tf = extract_tfidf_features(str(desc_path))
    agg: dict[str, int] = {}
    for tf in chunks_tf:
        for term, count in tf.items():
            if term in codebook:
                agg[term] = agg.get(term, 0) + count
    raw_text = desc_path.read_text(encoding="utf-8", errors="ignore")
    return agg, raw_text


def _load_metadata_csv(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = (row.get("stem") or "").strip()
            if not stem:
                continue
            out[stem] = {
                "name": (row.get("name") or stem).strip(),
                "category": (row.get("category") or "").strip() or None,
                "subcategory": (row.get("subcategory") or "").strip() or None,
            }
    return out


def _image_hist(
    image_path: Path, vq,
    cache_dir: Path | None = None,
) -> dict[str, int]:
    desc = None
    if cache_dir is not None:
        cached = cache_dir / f"{image_path.stem}.npy"
        if cached.exists():
            desc = np.load(cached, mmap_mode=None)
    if desc is None:
        try:
            desc = extract_sift_features(str(image_path))
        except Exception as e:
            print(f"  [SKIP image persist] {image_path.name}: {e.__class__.__name__}: {e}")
            return {}
    if desc.size == 0:
        return {}
    counts = vq.histogram(desc)
    return {f"v_{i:04d}": int(c) for i, c in enumerate(counts) if c > 0}


def run(
    images_dir: Path, descs_dir: Path,
    codebook_image: int, codebook_text: int,
    index_dir: Path,
    image_exts: tuple[str, ...], desc_ext: str,
    reset: bool,
    metadata_csv: Path | None = None,
    max_image_samples: int = 50_000,
    codebooks_dir: Path = Path("codebooks"),
    app_name: str = "fashion",
) -> dict:
    metadata = _load_metadata_csv(metadata_csv)
    pairs = _pair_files(images_dir, descs_dir, image_exts, desc_ext)
    if not pairs:
        raise SystemExit(
            f"No hay pares de archivos: revisa que {images_dir} y {descs_dir} "
            f"compartan nombres (ej. p_001.jpg + p_001.txt)."
        )
    print(f"Pares encontrados: {len(pairs)}")

    index_dir.mkdir(parents=True, exist_ok=True)
    codebooks_dir.mkdir(parents=True, exist_ok=True)
    text_index_dir = index_dir / "text"
    image_index_dir = index_dir / "image"

    image_paths = [str(ip) for _, ip, _ in pairs]
    desc_paths = [str(dp) for _, _, dp in pairs]

    t0 = time.perf_counter()
    print("[1/4] Construyendo codebook textual + indice SPIMI...")
    text_idx, text_codebook = index_text_corpus(
        file_paths=desc_paths,
        codebook_size=codebook_text,
        index_dir=text_index_dir,
    )
    text_idx.close()
    text_bag = sorted(text_codebook)
    print(f"      codebook texto: {len(text_bag)} palabras  "
          f"({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    print("[2/4] Construyendo codebook visual + indice SPIMI...")
    image_idx, image_vq = index_image_corpus(
        file_paths=image_paths,
        codebook_size=codebook_image,
        index_dir=image_index_dir,
        max_samples=max_image_samples,
    )
    image_idx.close()
    centroids = list(image_vq.centroids)
    image_keys = [f"v_{i:04d}" for i in range(len(centroids))]
    centroids_path = codebooks_dir / f"{app_name}_image_centroids.npy"
    np.save(centroids_path, np.array(centroids))
    print(f"      codebook imagen: {len(centroids)} centroides  "
          f"({time.perf_counter() - t0:.1f}s)")

    print("[3/4] Ajustando schema y limpiando tabla products...")
    if reset:
        reset_table("products")
    ensure_emb_column("products", "image_emb", len(centroids))

    print(f"[4/4] Persistiendo {len(pairs)} productos en BD...")
    image_cache = image_index_dir / "_features"
    if not image_cache.exists():
        image_cache = None
    inserted = 0
    batch: list[dict] = []
    batch_size = 500
    t0 = time.perf_counter()
    log_every = max(500, len(pairs) // 40)
    for i, (stem, ip, dp) in enumerate(pairs, 1):
        desc_hist, desc_text = _aggregate_desc_hist(dp, text_codebook)
        image_hist = _image_hist(ip, image_vq, cache_dir=image_cache)
        image_emb = hist_to_dense(image_hist, image_keys)
        meta = metadata.get(stem, {})
        batch.append(_product_params(
            name=meta.get("name") or stem,
            category=meta.get("category"),
            subcategory=meta.get("subcategory"),
            image_path=str(ip),
            description=desc_text,
            desc_hist=desc_hist,
            image_hist=image_hist,
            image_emb=image_emb,
            metadata={"source": "etl_fashion"},
        ))
        if len(batch) >= batch_size:
            inserted += save_products_batch(batch)
            batch = []
        if i % log_every == 0:
            print(f"  [persist] {i}/{len(pairs)}  ({time.perf_counter() - t0:.1f}s)")
    if batch:
        inserted += save_products_batch(batch)

    print(f"      insertados {inserted}/{len(pairs)} en {time.perf_counter() - t0:.1f}s")

    bag_path = codebooks_dir / f"{app_name}_text_bag.json"
    bag_path.write_text(json.dumps(text_bag, ensure_ascii=False), encoding="utf-8")
    save_codebook(
        app=app_name, modality="text",
        codebook_size=len(text_bag),
        bag_of_words=text_bag,
        index_dir=str(text_index_dir),
    )
    save_codebook(
        app=app_name, modality="image",
        codebook_size=len(centroids),
        centroids_path=str(centroids_path),
        index_dir=str(image_index_dir),
    )

    return {
        "pairs": len(pairs),
        "inserted": inserted,
        "codebook_text": len(text_bag),
        "codebook_image": len(centroids),
        "text_index_dir": str(text_index_dir),
        "image_index_dir": str(image_index_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Fashion (App 4)")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--descriptions-dir", required=True, type=Path)
    parser.add_argument("--codebook-image", type=int, default=1024)
    parser.add_argument("--codebook-text", type=int, default=1000)
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/fashion"))
    parser.add_argument("--codebooks-dir", type=Path, default=Path("codebooks"))
    parser.add_argument("--app-name", default="fashion")
    parser.add_argument(
        "--image-exts", default=".jpg,.jpeg,.png,.webp",
        help="Coma-separated lista de extensiones de imagen aceptadas",
    )
    parser.add_argument("--desc-ext", default=".txt")
    parser.add_argument("--max-image-samples", type=int, default=150_000,
                        help="Subsample de descriptores SIFT para KMeans (0 = sin limite)")
    parser.add_argument("--reset", action="store_true",
                        help="TRUNCATE products antes de insertar")
    parser.add_argument("--metadata-csv", type=Path, default=None,
                        help="CSV opcional con stem,name,category,subcategory")
    args = parser.parse_args()

    result = run(
        images_dir=args.images_dir,
        descs_dir=args.descriptions_dir,
        codebook_image=args.codebook_image,
        codebook_text=args.codebook_text,
        index_dir=args.index_dir,
        image_exts=tuple(args.image_exts.split(",")),
        desc_ext=args.desc_ext,
        reset=args.reset,
        metadata_csv=args.metadata_csv,
        max_image_samples=args.max_image_samples,
        codebooks_dir=args.codebooks_dir,
        app_name=args.app_name,
    )
    print("\n" + json.dumps(result, indent=2))