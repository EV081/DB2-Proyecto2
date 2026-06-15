from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.engine.inverted_index import PostingList, iter_block, spimi_invert
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


def _run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} tests pasaron.")


if __name__ == "__main__":
    _run_all()