from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

KAGGLE_DATASET = os.environ.get(
    "FASHION_KAGGLE_DATASET",
    "paramaggarwal/fashion-product-images-dataset",
)


def _kaggle_creds_ok() -> bool:
    home = Path.home() / ".kaggle"
    return (
        (home / "kaggle.json").exists()
        or (home / "access_token").exists()
        or bool(os.environ.get("KAGGLE_API_TOKEN"))
        or (bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY")))
    )


def _compose_desc(row: dict) -> str:
    parts = [
        row.get("productDisplayName") or row.get("name") or "",
        row.get("gender", ""),
        row.get("masterCategory", ""),
        row.get("subCategory", ""),
        row.get("articleType", ""),
        row.get("baseColour", ""),
        row.get("usage", ""),
        row.get("season", ""),
    ]
    return ". ".join(p.strip() for p in parts if p and p.strip())


def run(out_dir: Path, limit: int | None) -> dict:
    if not _kaggle_creds_ok():
        sys.exit(
            "Faltan credenciales de Kaggle. Acepta cualquiera de:\n"
            "  - ~/.kaggle/access_token (token nuevo KGAT_...)\n"
            "  - ~/.kaggle/kaggle.json   (formato clásico)\n"
            "  - env KAGGLE_API_TOKEN, o KAGGLE_USERNAME + KAGGLE_KEY"
        )
    try:
        import kagglehub
    except ImportError:
        sys.exit("Instala con: pip install kagglehub")

    print(f"Descargando '{KAGGLE_DATASET}'...")
    src_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"  cache kagglehub: {src_dir}")

    src_images = next(src_dir.rglob("images"), None)
    styles = next(src_dir.rglob("styles.csv"), None)
    if src_images is None or styles is None:
        sys.exit(f"No se encontraron images/ o styles.csv en {src_dir}")
    print(f"  images: {src_images}")
    print(f"  styles: {styles}")

    images_out = out_dir / "images"
    descs_out = out_dir / "descs"
    images_out.mkdir(parents=True, exist_ok=True)
    descs_out.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"

    n_linked = n_descs = 0
    with styles.open(encoding="utf-8", errors="ignore") as f, \
         meta_path.open("w", encoding="utf-8", newline="") as mf:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(mf, fieldnames=["stem", "name", "category", "subcategory"])
        writer.writeheader()
        for row in reader:
            if limit and n_descs >= limit:
                break
            pid = (row.get("id") or "").strip()
            if not pid:
                continue
            img_src = src_images / f"{pid}.jpg"
            if not img_src.exists():
                continue
            img_link = images_out / f"{pid}.jpg"
            if not img_link.exists():
                img_link.symlink_to(img_src.resolve())
                n_linked += 1
            desc = _compose_desc(row)
            (descs_out / f"{pid}.txt").write_text(desc, encoding="utf-8")
            writer.writerow({
                "stem": pid,
                "name": row.get("productDisplayName") or pid,
                "category": row.get("masterCategory") or "",
                "subcategory": row.get("subCategory") or "",
            })
            n_descs += 1

    print(f"OK: {n_descs} productos  ({n_linked} symlinks nuevos)")
    print(f"  images -> {images_out}")
    print(f"  descs  -> {descs_out}")
    print(f"  meta   -> {meta_path}")
    return {
        "images_dir": str(images_out),
        "descs_dir": str(descs_out),
        "metadata_csv": str(meta_path),
        "n": n_descs,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Descarga Fashion images (Kaggle)")
    p.add_argument("--out-dir", type=Path, default=Path("data/fashion"))
    p.add_argument("--limit", type=int, default=None,
                   help="Limitar nro de productos (debug). Default: todos.")
    args = p.parse_args()
    run(out_dir=args.out_dir, limit=args.limit)