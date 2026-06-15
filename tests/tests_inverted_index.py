from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.engine.inverted_index import (
    InvertedIndex,
    PostingList,
    iter_block,
    merge_blocks,
    spimi_invert,
)
from src.engine.mock_data import TEXT_COLLECTION


def test_posting_list_starts_empty() -> None:
    pl = PostingList()
    assert len(pl) == 0
    assert pl.to_list() == []
    assert pl.capacity() == 4


def test_posting_list_append_and_iterate() -> None:
    pl = PostingList()
    pl.append("doc1", 3)
    pl.append("doc2", 1)
    pl.append("doc5", 7)
    assert len(pl) == 3
    assert list(pl) == [("doc1", 3), ("doc2", 1), ("doc5", 7)]


def test_posting_list_grows_with_doubling() -> None:
    pl = PostingList()
    assert pl.capacity() == 4
    for i in range(4):
        pl.append(f"doc{i}", 1)
    assert pl.capacity() == 4   # llena pero no creció todavía
    pl.append("doc4", 1)
    assert pl.capacity() == 8   # doubling
    for i in range(5, 8):
        pl.append(f"doc{i}", 1)
    assert pl.capacity() == 8
    pl.append("doc8", 1)
    assert pl.capacity() == 16


def test_posting_list_preserves_insertion_order() -> None:
    # Si los docs se procesan en orden creciente, la posting list queda ordenada
    # por doc_id sin esfuerzo extra (invariante crítico para multi-way merge).
    pl = PostingList()
    for i in range(20):
        pl.append(f"doc_{i:03d}", i + 1)
    ids = [p[0] for p in pl]
    assert ids == sorted(ids)


def test_posting_list_getitem() -> None:
    pl = PostingList()
    pl.append("a", 1)
    pl.append("b", 2)
    assert pl[0] == ("a", 1)
    assert pl[1] == ("b", 2)
    try:
        _ = pl[2]
    except IndexError:
        pass
    else:
        raise AssertionError("se esperaba IndexError")


def test_posting_list_sort_by_doc_id() -> None:
    pl = PostingList()
    for doc_id, tf in [("doc_c", 1), ("doc_a", 2), ("doc_b", 3)]:
        pl.append(doc_id, tf)
    pl.sort_by_doc_id()
    assert pl.to_list() == [("doc_a", 2), ("doc_b", 3), ("doc_c", 1)]


def test_posting_list_doubling_doesnt_lose_data() -> None:
    # Stress: 1000 appends -> contenido íntegro y orden preservado.
    pl = PostingList()
    expected = [(f"d{i:04d}", i) for i in range(1000)]
    for doc_id, tf in expected:
        pl.append(doc_id, tf)
    assert len(pl) == 1000
    assert pl.to_list() == expected
    # Capacity es potencia de 2 >= 1000
    assert pl.capacity() >= 1000
    assert (pl.capacity() & (pl.capacity() - 1)) == 0


def test_spimi_single_block_when_collection_small() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        blocks = spimi_invert(
            doc_stream=iter(TEXT_COLLECTION.items()),
            block_size_postings=10_000,   # nada lo dispara, todo en un solo bloque
            out_dir=tmp,
        )
        assert len(blocks) == 1
        terms_block = [t for t, _ in iter_block(blocks[0])]
        assert terms_block == sorted(terms_block)


def test_spimi_multiple_blocks_when_threshold_low() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        blocks = spimi_invert(
            doc_stream=iter(TEXT_COLLECTION.items()),
            block_size_postings=5,        # umbral pequeño -> varios bloques
            out_dir=tmp,
        )
        assert len(blocks) >= 2
        for path in blocks:
            terms = [t for t, _ in iter_block(path)]
            assert terms == sorted(terms)


def test_spimi_preserves_all_postings() -> None:
    expected_total = sum(
        sum(1 for tf in tf_doc.values() if tf > 0)
        for tf_doc in TEXT_COLLECTION.values()
    )
    with tempfile.TemporaryDirectory() as tmp:
        blocks = spimi_invert(
            doc_stream=iter(TEXT_COLLECTION.items()),
            block_size_postings=3,
            out_dir=tmp,
        )
        total = 0
        for path in blocks:
            for _, postings in iter_block(path):
                total += len(postings)
        assert total == expected_total


def test_spimi_skips_zero_tf() -> None:
    docs = [
        ("d1", {"a": 5, "b": 0, "c": 2}),
        ("d2", {"a": 0, "c": 1}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        blocks = spimi_invert(iter(docs), block_size_postings=100, out_dir=tmp)
        all_terms = []
        for path in blocks:
            for term, _ in iter_block(path):
                all_terms.append(term)
        assert "b" not in all_terms      # tf=0 nunca se indexa
        assert set(all_terms) == {"a", "c"}


def test_spimi_block_file_format_is_jsonl_per_term() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        blocks = spimi_invert(
            iter([("d1", {"x": 1, "y": 2})]),
            block_size_postings=100,
            out_dir=tmp,
        )
        path = blocks[0]
        with path.open() as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert all("t" in obj and "p" in obj for obj in lines)
        assert [obj["t"] for obj in lines] == ["x", "y"]


def test_merge_blocks_unifies_terms_alphabetically() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        blocks = spimi_invert(
            iter(TEXT_COLLECTION.items()),
            block_size_postings=4,   # forzar varios bloques
            out_dir=Path(tmp) / "blocks",
        )
        assert len(blocks) >= 2
        postings_path, vocab_path = merge_blocks(blocks, Path(tmp) / "final")

        with postings_path.open() as f:
            terms = [json.loads(l)["t"] for l in f if l.strip()]
        assert terms == sorted(terms)
        assert len(terms) == len(set(terms))   # cada termino aparece una sola vez


def test_merge_blocks_preserves_all_postings() -> None:
    expected: dict[str, set] = {}
    for doc_id, tf_doc in TEXT_COLLECTION.items():
        for term, tf in tf_doc.items():
            if tf > 0:
                expected.setdefault(term, set()).add((doc_id, tf))

    with tempfile.TemporaryDirectory() as tmp:
        blocks = spimi_invert(
            iter(TEXT_COLLECTION.items()),
            block_size_postings=3,
            out_dir=Path(tmp) / "blocks",
        )
        postings_path, _ = merge_blocks(blocks, Path(tmp) / "final")

        with postings_path.open() as f:
            actual: dict[str, set] = {}
            for line in f:
                obj = json.loads(line)
                actual[obj["t"]] = {(p[0], p[1]) for p in obj["p"]}

    assert actual == expected


def test_merge_blocks_vocab_offsets_locate_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        blocks = spimi_invert(
            iter(TEXT_COLLECTION.items()),
            block_size_postings=4,
            out_dir=Path(tmp) / "blocks",
        )
        postings_path, vocab_path = merge_blocks(blocks, Path(tmp) / "final")
        vocab = json.loads(vocab_path.read_text())

        with postings_path.open("rb") as f:
            for term, meta in vocab.items():
                f.seek(meta["offset"])
                raw = f.read(meta["length"])
                obj = json.loads(raw)
                assert obj["t"] == term
                assert len(obj["p"]) == meta["df"]


def test_merge_blocks_postings_sorted_by_doc_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        blocks = spimi_invert(
            iter(TEXT_COLLECTION.items()),
            block_size_postings=3,
            out_dir=Path(tmp) / "blocks",
        )
        postings_path, _ = merge_blocks(blocks, Path(tmp) / "final")
        with postings_path.open() as f:
            for line in f:
                obj = json.loads(line)
                doc_ids = [p[0] for p in obj["p"]]
                assert doc_ids == sorted(doc_ids)


def _build_text_index(tmp: str) -> tuple[Path, Path]:
    blocks = spimi_invert(
        iter(TEXT_COLLECTION.items()),
        block_size_postings=4,
        out_dir=Path(tmp) / "blocks",
    )
    return merge_blocks(blocks, Path(tmp) / "final")


def test_inverted_index_loads_vocab() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        postings_path, vocab_path = _build_text_index(tmp)
        idx = InvertedIndex(postings_path, vocab_path,
                            n_docs=len(TEXT_COLLECTION))
        assert idx.vocab_size() > 0
        assert idx.n_docs == len(TEXT_COLLECTION)


def test_inverted_index_get_postings_known_term() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        postings_path, vocab_path = _build_text_index(tmp)
        with InvertedIndex(postings_path, vocab_path) as idx:
            postings = idx.get_postings("love")
            expected_docs = {
                doc_id for doc_id, tf in TEXT_COLLECTION.items()
                if tf.get("love", 0) > 0
            }
            assert {p[0] for p in postings} == expected_docs


def test_inverted_index_unknown_term_returns_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        postings_path, vocab_path = _build_text_index(tmp)
        with InvertedIndex(postings_path, vocab_path) as idx:
            assert idx.get_postings("noexiste") == []
            assert idx.df("noexiste") == 0
            assert not idx.has_term("noexiste")


def test_inverted_index_io_counters_increment() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        postings_path, vocab_path = _build_text_index(tmp)
        with InvertedIndex(postings_path, vocab_path) as idx:
            assert idx.io_seeks == 0
            assert idx.io_read_bytes == 0
            idx.get_postings("love")
            assert idx.io_seeks == 1
            assert idx.io_read_bytes > 0
            idx.get_postings("dance")
            assert idx.io_seeks == 2
            # Termino inexistente: no debe incrementar (no hay I/O)
            idx.get_postings("ghost")
            assert idx.io_seeks == 2


def test_inverted_index_reset_io_counters() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        postings_path, vocab_path = _build_text_index(tmp)
        with InvertedIndex(postings_path, vocab_path) as idx:
            idx.get_postings("love")
            idx.reset_io_counters()
            assert idx.io_seeks == 0
            assert idx.io_read_bytes == 0


def test_inverted_index_df_matches_collection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        postings_path, vocab_path = _build_text_index(tmp)
        with InvertedIndex(postings_path, vocab_path) as idx:
            for term in ["love", "dance", "cry"]:
                expected_df = sum(
                    1 for tf in TEXT_COLLECTION.values()
                    if tf.get(term, 0) > 0
                )
                assert idx.df(term) == expected_df


def _run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} tests pasaron.")


if __name__ == "__main__":
    _run_all()