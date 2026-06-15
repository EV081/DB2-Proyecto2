from __future__ import annotations

import heapq
import json
from math import log10, sqrt
from pathlib import Path
from typing import Iterable, Iterator

from src.engine.similarity import log_tf

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


def merge_blocks(
    block_paths: list[Path],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    postings_path = out / "final.postings"
    vocab_path = out / "vocab.json"

    iters = [iter_block(p) for p in block_paths]
    merged = heapq.merge(*iters, key=lambda x: x[0])

    vocab: dict[str, dict] = {}
    current_term: str | None = None
    current_postings: list[Posting] = []

    with postings_path.open("w", encoding="utf-8") as f:
        for term, postings in merged:
            if term != current_term:
                if current_term is not None:
                    _write_term(f, current_term, current_postings, vocab)
                current_term = term
                current_postings = list(postings)
            else:
                current_postings.extend(postings)
        if current_term is not None:
            _write_term(f, current_term, current_postings, vocab)

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    return postings_path, vocab_path


def _write_term(
    f,
    term: str,
    postings: list[Posting],
    vocab: dict[str, dict],
) -> None:
    postings.sort(key=lambda p: p[0])
    offset = f.tell()
    line = json.dumps({"t": term, "p": postings}, ensure_ascii=False)
    f.write(line)
    f.write("\n")
    length = f.tell() - offset
    vocab[term] = {"offset": offset, "length": length, "df": len(postings)}


def build_meta(
    postings_path: str | Path,
    vocab_path: str | Path,
    out_dir: str | Path,
) -> Path:
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))

    doc_ids: set[str] = set()
    with Path(postings_path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for doc_id, _ in obj["p"]:
                doc_ids.add(doc_id)
    n_docs = len(doc_ids)

    norms_sq: dict[str, float] = {}
    with Path(postings_path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            term = obj["t"]
            df_t = vocab[term]["df"]
            idf_t = log10(n_docs / df_t) if df_t > 0 else 0.0
            for doc_id, tf_d in obj["p"]:
                w = log_tf(tf_d) * idf_t
                norms_sq[doc_id] = norms_sq.get(doc_id, 0.0) + w * w

    doc_norms = {doc_id: sqrt(s) for doc_id, s in norms_sq.items()}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta_path.write_text(
        json.dumps({"n_docs": n_docs, "doc_norms": doc_norms}, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta_path


class InvertedIndex:
    def __init__(
        self,
        postings_path: str | Path,
        vocab_path: str | Path,
        meta_path: str | Path | None = None,
        n_docs: int = 0,
    ) -> None:
        self._postings_path = Path(postings_path)
        self._vocab: dict[str, dict] = json.loads(
            Path(vocab_path).read_text(encoding="utf-8")
        )
        if meta_path is not None:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            self.n_docs: int = meta["n_docs"]
            self._doc_norms: dict[str, float] = meta["doc_norms"]
        else:
            self.n_docs = n_docs
            self._doc_norms = {}
        self._fh = None
        self.io_seeks = 0
        self.io_read_bytes = 0

    def open(self) -> "InvertedIndex":
        if self._fh is None:
            self._fh = self._postings_path.open("rb")
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "InvertedIndex":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def vocab_size(self) -> int:
        return len(self._vocab)

    def df(self, term: str) -> int:
        meta = self._vocab.get(term)
        return meta["df"] if meta else 0

    def has_term(self, term: str) -> bool:
        return term in self._vocab

    def get_postings(self, term: str) -> list[Posting]:
        meta = self._vocab.get(term)
        if meta is None:
            return []
        if self._fh is None:
            self.open()
        self._fh.seek(meta["offset"])
        self.io_seeks += 1
        raw = self._fh.read(meta["length"])
        self.io_read_bytes += meta["length"]
        obj = json.loads(raw)
        return [(p[0], p[1]) for p in obj["p"]]

    def reset_io_counters(self) -> None:
        self.io_seeks = 0
        self.io_read_bytes = 0

    def search_topk(
        self,
        query_tf: TermFreqs,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        if self.n_docs <= 0:
            raise ValueError("n_docs no esta seteado; usar meta_path o n_docs=N")
        if not self._doc_norms:
            raise ValueError("doc_norms no cargado; usar meta_path o build_meta()")

        q_weights: dict[str, float] = {}
        for term, tf in query_tf.items():
            if tf <= 0:
                continue
            df_t = self.df(term)
            if df_t == 0:
                continue
            idf_t = log10(self.n_docs / df_t)
            q_weights[term] = log_tf(tf) * idf_t

        if not q_weights:
            return []

        q_norm = sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0.0:
            return []

        scores: dict[str, float] = {}
        for term, w_q in q_weights.items():
            df_t = self.df(term)
            idf_t = log10(self.n_docs / df_t)
            for doc_id, tf_d in self.get_postings(term):
                w_d = log_tf(tf_d) * idf_t
                scores[doc_id] = scores.get(doc_id, 0.0) + w_q * w_d

        for doc_id in scores:
            d_norm = self._doc_norms.get(doc_id, 0.0)
            if d_norm > 0:
                scores[doc_id] /= (q_norm * d_norm)
            else:
                scores[doc_id] = 0.0

        return heapq.nlargest(k, scores.items(), key=lambda x: x[1])


__all__ = [
    "Posting", "PostingList", "TermFreqs", "DocStream",
    "spimi_invert", "iter_block", "merge_blocks", "build_meta",
    "InvertedIndex",
]
