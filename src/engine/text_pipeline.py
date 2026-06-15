from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from src.engine.inverted_index import (
    InvertedIndex,
    build_meta,
    merge_blocks,
    spimi_invert,
)
from src.extraction.text_tfidf import (
    compute_tf,
    extract_tfidf_features,
    remove_stopwords,
    stem,
    tokenize,
)
from src.ml.text_topk import TokKWords


def build_text_codebook(file_paths: Iterable[str | Path], codebook_size: int) -> set[str]:
    tk = TokKWords(top_k=codebook_size)
    for fp in file_paths:
        chunks_tf = extract_tfidf_features(str(fp))
        for tf in chunks_tf:
            tk.apply_document_tf(tf)
    return set(tk.close())


def _doc_stream(
    file_paths: Iterable[str | Path],
    codebook: set[str],
) -> Iterator[tuple[str, dict[str, int]]]:
    for fp in file_paths:
        stem_name = Path(fp).stem
        chunks_tf = extract_tfidf_features(str(fp))
        for i, tf in enumerate(chunks_tf):
            filtered = {k: v for k, v in tf.items() if k in codebook}
            if filtered:
                yield f"{stem_name}#chunk{i:03d}", filtered


def index_text_corpus(
    file_paths: Iterable[str | Path],
    codebook_size: int,
    index_dir: str | Path,
    block_size_postings: int = 500_000,
) -> tuple[InvertedIndex, set[str]]:
    file_list = [Path(p) for p in file_paths]
    codebook = build_text_codebook(file_list, codebook_size)

    out = Path(index_dir)
    blocks = spimi_invert(
        _doc_stream(file_list, codebook),
        block_size_postings=block_size_postings,
        out_dir=out / "blocks",
    )
    if not blocks:
        raise ValueError("Ningun chunk produjo terminos del codebook; corpus vacio")
    postings_path, vocab_path = merge_blocks(blocks, out / "final")
    meta_path = build_meta(postings_path, vocab_path, out / "final")
    idx = InvertedIndex(postings_path, vocab_path, meta_path).open()
    return idx, codebook


def prepare_query(query_text: str, codebook: set[str] | None = None) -> dict[str, int]:
    tokens = tokenize(query_text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    tf = compute_tf(tokens)
    if codebook is not None:
        tf = {k: v for k, v in tf.items() if k in codebook}
    return tf


__all__ = [
    "build_text_codebook",
    "index_text_corpus",
    "prepare_query",
]