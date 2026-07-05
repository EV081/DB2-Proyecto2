from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from src.db.database import get_session


# Mapeo seccion_json -> (tabla_db, columna_filtro)
_SECTION_TO_COUNT = {
    "music_lyrics":   ("songs",    "lyrics_text IS NOT NULL"),
    "music_audio":    ("songs",    "audio_path IS NOT NULL"),
    "fashion_desc":   ("products", "description IS NOT NULL"),
    "fashion_image":  ("products", "image_path IS NOT NULL"),
}

# Etiquetas humanas en las tablas del INFORME
_ENGINE_LABEL = {
    "spimi":    "SPIMI",
    "gin":      "GIN",
    "gist":     "GiST",
    "pgvector": "pgvector",
}


def _corpus_size(section: str) -> int:
    table, where = _SECTION_TO_COUNT[section]
    with get_session() as session:
        row = session.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {where}")).first()
    return int(row[0] or 0) if row else 0


def _closest_size_row(size_actual: int, sizes_en_tabla: list[int]) -> int:
    return min(sizes_en_tabla, key=lambda n: abs(n - size_actual))


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _replace_row(md_text: str, n_docs: int, engine_label: str, metrics: dict, recall: str) -> str:
    n_str = f"{n_docs:,}".replace(",", " ")
    # Intenta match con separador de miles con espacio Y sin separador
    for n_variant in (n_str, str(n_docs)):
        pattern = re.compile(
            r"^(\|\s*" + re.escape(n_variant) + r"\s*\|\s*" + re.escape(engine_label) +
            r"\s*)\|\s*\[INSERTAR\]\s*\|\s*\[INSERTAR\]\s*\|\s*\[INSERTAR\]\s*\|\s*\[INSERTAR\]\s*\|\s*\[INSERTAR\]\s*\|",
            re.MULTILINE,
        )
        replacement = (
            r"\1"
            + f"| {_fmt(metrics.get('avg_ms'))} "
            + f"| {_fmt(metrics.get('p50_ms'))} "
            + f"| {_fmt(metrics.get('p95_ms'))} "
            + f"| {_fmt(metrics.get('throughput_qps'))} "
            + f"| {recall} |"
        )
        new_text, n_subs = pattern.subn(replacement, md_text)
        if n_subs > 0:
            return new_text
    return md_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True,
                        help="JSON producido por bench_full.py")
    parser.add_argument("--recall", default=None,
                        help="JSON producido por compute_recall.py (opcional)")
    parser.add_argument("--informe", default="docs/INFORME.md")
    args = parser.parse_args()

    bench = json.loads(Path(args.bench).read_text(encoding="utf-8"))
    recall = (json.loads(Path(args.recall).read_text(encoding="utf-8"))
              if args.recall else {})

    informe_path = Path(args.informe)
    md = informe_path.read_text(encoding="utf-8")

    # Detecta los N que ya estan escritos en el INFORME por seccion
    # (miramos las filas | XXX | SPIMI | [INSERTAR] ...)
    sizes_by_section = {}
    for section in _SECTION_TO_COUNT:
        candidates = re.findall(
            r"\|\s*([\d ]+?)\s*\|\s*(?:SPIMI|GIN|GiST|pgvector)\s*\|\s*\[INSERTAR\]",
            md,
        )
        # NB: candidates es global, no por seccion; lo usamos solo para elegir el N cercano
        cleaned = []
        for c in candidates:
            n = int(c.replace(" ", ""))
            if n not in cleaned:
                cleaned.append(n)
        sizes_by_section[section] = cleaned

    total_reemplazos = 0
    for section, engines_metrics in bench.items():
        if section not in _SECTION_TO_COUNT:
            continue
        n_actual = _corpus_size(section)
        if n_actual == 0:
            print(f"[skip] {section}: DB vacia")
            continue

        candidatos = sizes_by_section.get(section, [])
        if not candidatos:
            print(f"[warn] {section}: no encontre filas [INSERTAR] en el INFORME")
            continue

        n_fila = _closest_size_row(n_actual, candidatos)
        section_recall = recall.get(section, {})
        print(f"[section] {section}: DB tiene {n_actual}, rellenando fila N={n_fila}")

        for engine, metrics in engines_metrics.items():
            engine_label = _ENGINE_LABEL.get(engine, engine)
            r = section_recall.get(engine, {}).get("recall_at_k") if section_recall else None
            recall_str = f"{r:.3f}" if isinstance(r, (int, float)) else "-"
            new_md = _replace_row(md, n_fila, engine_label, metrics, recall_str)
            if new_md != md:
                total_reemplazos += 1
                md = new_md
                print(f"  [ok]   {engine_label}")
            else:
                print(f"  [miss] {engine_label}  (no encontre patron)")

    informe_path.write_text(md, encoding="utf-8")
    print(f"\n{total_reemplazos} filas rellenadas en {informe_path}")


if __name__ == "__main__":
    main()