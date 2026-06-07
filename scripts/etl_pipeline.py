import os
import sys
from pathlib import Path

KAGGLE_BASE = Path("datasets")

DATASET_STRUCTURE = {
    "audio": KAGGLE_BASE / "audio" / "songs",
    "images": KAGGLE_BASE / "images" / "products",
    "text": KAGGLE_BASE / "text" / "lyrics",
}

def find_files(directory: Path, extensions: tuple) -> list[Path]:
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*") if p.suffix in extensions]

def scan_datasets() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}

    for modality, path in DATASET_STRUCTURE.items():
        print(f"[ETL] Escaneando {modality}: {path}")
        if modality == "audio":
            files = find_files(path, (".wav", ".mp3", ".flac"))
        elif modality == "images":
            files = find_files(path, (".jpg", ".jpeg", ".png", ".webp"))
        else:
            files = find_files(path, (".txt", ".json", ".csv"))
        found[modality] = files
        print(f"[ETL]   -> {len(files)} archivos encontrados")

    return found

def print_summary(data: dict[str, list[Path]]) -> None:
    print("\n" + "=" * 50)
    print("RESUMEN ETL (MOCK)")
    print("=" * 50)
    total = 0
    for modality, files in data.items():
        print(f"\n  [{modality.upper()}] ({len(files)} archivos):")
        for f in files[:5]:
            print(f"    {f}")
        if len(files) > 5:
            print(f"    ... y {len(files) - 5} más")
        total += len(files)
    print(f"\n  TOTAL: {total} archivos encontrados")
    print("=" * 50)

if __name__ == "__main__":
    if not KAGGLE_BASE.exists():
        print(f"[ETL] Directorio '{KAGGLE_BASE}' no encontrado. Usando datos mock.")
        print("[ETL] Lugar esperado: descargar datasets de Kaggle en ./datasets/")
        print("[ETL] Ejemplo: kaggle datasets download -d <dataset> -p datasets/\n")

    data = scan_datasets()
    print_summary(data)

    print("\n[ETL] Pipeline completado (modo mock — sin procesamiento real).")
