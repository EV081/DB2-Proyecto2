from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

Posting = tuple[str, int] #(doc_id, tf)
TermFreqs = dict[str, int]
DocStream = Iterable[tuple[str, TermFreqs]]


class PostingList:
    __slots__ = ("_data", "_size")

    _INITIAL_CAPACITY = 4

    def __init__(self) -> None:
        self._data: list[Posting | None] = [None] * self._INITIAL_CAPACITY
        self._size = 0

    def append(self, doc_id: str, tf: int) -> None:
        if self._size == len(self._data):
            self._grow()
        self._data[self._size] = (doc_id, tf)
        self._size += 1

    def _grow(self) -> None:
        new_cap = len(self._data) * 2
        new_data: list[Posting | None] = [None] * new_cap
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Posting]:
        for i in range(self._size):
            yield self._data[i]

    def __getitem__(self, i: int) -> Posting:
        if i < 0 or i >= self._size:
            raise IndexError(f"posting index {i} out of range [0, {self._size})")
        return self._data[i]

    def capacity(self) -> int:
        return len(self._data)

    def to_list(self) -> list[Posting]:
        return [self._data[i] for i in range(self._size)] 

    def sort_by_doc_id(self) -> None:
        sorted_view = sorted(
            (self._data[i] for i in range(self._size)),
            key=lambda p: p[0],
        )
        for i, p in enumerate(sorted_view):
            self._data[i] = p


def _write_block(dictionary: dict[str, PostingList], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for term in sorted(dictionary):
            line = {"t": term, "p": dictionary[term].to_list()}
            f.write(json.dumps(line, ensure_ascii=False))
            f.write("\n")


def spimi_invert(
    doc_stream: DocStream,
    block_size_postings: int,
    out_dir: str | Path,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    block_paths: list[Path] = []
    dictionary: dict[str, PostingList] = {}
    posting_count = 0
    block_idx = 0

    for doc_id, tf_doc in doc_stream:
        for term, tf in tf_doc.items():
            if tf <= 0:
                continue
            pl = dictionary.get(term)
            if pl is None:
                pl = PostingList()
                dictionary[term] = pl
            pl.append(doc_id, tf)
            posting_count += 1

        if posting_count >= block_size_postings:
            path = out / f"block_{block_idx:04d}.jsonl"
            _write_block(dictionary, path)
            block_paths.append(path)
            dictionary = {}
            posting_count = 0
            block_idx += 1

    if dictionary:
        path = out / f"block_{block_idx:04d}.jsonl"
        _write_block(dictionary, path)
        block_paths.append(path)

    return block_paths


def iter_block(path: Path) -> Iterator[tuple[str, list[Posting]]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            postings: list[Posting] = [(p[0], p[1]) for p in obj["p"]]
            yield obj["t"], postings


__all__ = [
    "Posting", "PostingList", "TermFreqs", "DocStream",
    "spimi_invert", "iter_block",
]
