from __future__ import annotations

import tempfile
from pathlib import Path

from src.engine.text_pipeline import (
    build_text_codebook,
    index_text_corpus,
    prepare_query,
)


# Corpus sintetico: 3 archivos, 2 parrafos cada uno = 6 chunks. Cada archivo
# tiene un tema distinto para que el top-1 de cada query sea predecible.
DOCS = {
    "love.txt": (
        "I love love this song so much it warms my heart.\n\n"
        "My heart beats fast with love and pure joy tonight."
    ),
    "party.txt": (
        "Dance all night in the club with friends and music.\n\n"
        "The music is loud and the dance never stops until morning."
    ),
    "sad.txt": (
        "I cry alone in the rain on this cold lonely night.\n\n"
        "The rain falls hard tonight and I cry again with sadness."
    ),
}


def _make_corpus(tmp: Path) -> list[str]:
    paths = []
    for name, content in DOCS.items():
        p = tmp / name
        p.write_text(content)
        paths.append(str(p))
    return paths


def test_build_text_codebook_returns_top_k_stems() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_corpus(Path(tmp))
        bag = build_text_codebook(paths, codebook_size=10)
        assert len(bag) <= 10
        # Las raices mas frecuentes esperadas tras Porter stem
        assert "love" in bag
        assert "danc" in bag       # dance/dances/dancing -> danc
        assert "rain" in bag


def test_index_text_corpus_builds_indexable_corpus() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = _make_corpus(tmp)
        idx, codebook = index_text_corpus(
            file_paths=paths,
            codebook_size=30,
            index_dir=tmp / "index",
        )
        try:
            assert idx.n_docs == 6              # 3 archivos x 2 parrafos
            assert idx.vocab_size() > 0
            assert idx.vocab_size() <= len(codebook)
        finally:
            idx.close()


def test_query_returns_expected_top_doc() -> None:
    expectations = [
        ("love song heart", "love"),
        ("dance club night", "party"),
        ("rain crying lonely", "sad"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = _make_corpus(tmp)
        idx, codebook = index_text_corpus(
            file_paths=paths,
            codebook_size=30,
            index_dir=tmp / "index",
        )
        try:
            for query_text, expected_file_prefix in expectations:
                q_tf = prepare_query(query_text, codebook=codebook)
                results = idx.search_topk(q_tf, k=3)
                assert results, f"sin resultados para {query_text!r}"
                top_doc, _ = results[0]
                assert top_doc.startswith(expected_file_prefix), (
                    f"query {query_text!r}: esperaba {expected_file_prefix!r}, "
                    f"top fue {top_doc!r}"
                )
        finally:
            idx.close()


def test_prepare_query_filters_to_codebook() -> None:
    codebook = {"love", "heart"}
    tf = prepare_query("I love this random unrelated thing my heart", codebook=codebook)
    assert set(tf) <= codebook
    assert "love" in tf
    assert "heart" in tf


def test_prepare_query_without_codebook_keeps_all_stems() -> None:
    tf = prepare_query("I love dancing in the rain")
    assert "love" in tf
    assert "danc" in tf
    assert "rain" in tf


def test_query_unknown_terms_returns_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = _make_corpus(tmp)
        idx, codebook = index_text_corpus(
            file_paths=paths,
            codebook_size=30,
            index_dir=tmp / "index",
        )
        try:
            q_tf = prepare_query("xyzfoo bargibberish", codebook=codebook)
            assert q_tf == {}
            assert idx.search_topk(q_tf, k=5) == []
        finally:
            idx.close()


def test_query_io_seeks_match_valid_query_terms() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = _make_corpus(tmp)
        idx, codebook = index_text_corpus(
            file_paths=paths,
            codebook_size=30,
            index_dir=tmp / "index",
        )
        try:
            q_tf = prepare_query("love heart dance", codebook=codebook)
            idx.reset_io_counters()
            idx.search_topk(q_tf, k=5)
            valid_terms = sum(1 for t in q_tf if idx.has_term(t))
            assert idx.io_seeks == valid_terms
        finally:
            idx.close()


def _run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} tests pasaron.")


if __name__ == "__main__":
    _run_all()
