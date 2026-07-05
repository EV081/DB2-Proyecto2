from __future__ import annotations

from src.ml.text_topk import TopKWords


_DOCS = [
    ["hola", "mundo", "hola", "test"],
    ["mundo", "foo", "bar", "test", "test"],
    ["hola", "foo", "test"],
]

def test_top_k_returns_most_frequent_tokens() -> None:
    tk = TopKWords(top_k=3)
    for doc in _DOCS:
        tk.apply_document(doc)
    bag = tk.close()
    # test y hola son inequivocos; el 3ro es empate mundo vs foo,
    # Counter rompe empates por orden de insercion -> mundo (doc1) antes que foo (doc2)
    assert bag == ["test", "hola", "mundo"]


def test_apply_document_tf_equivalent_to_apply_document() -> None:
    tk_tokens = TopKWords(top_k=3)
    tk_tf = TopKWords(top_k=3)
    for doc in _DOCS:
        tk_tokens.apply_document(doc)
        tf: dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        tk_tf.apply_document_tf(tf)
    assert tk_tokens.close() == tk_tf.close()


def test_close_returns_all_when_fewer_unique_than_k() -> None:
    tk = TopKWords(top_k=100)
    tk.apply_document(["a", "b", "a", "c"])
    bag = tk.close()
    assert sorted(bag) == ["a", "b", "c"]
    assert len(bag) == 3  # menos de top_k=100


def test_reset_clears_state() -> None:
    tk = TopKWords(top_k=2)
    tk.apply_document(["x", "y", "z"])
    tk.reset()
    assert tk.bag_of_words == []
    # despues del reset se puede repoblar y cerrar limpio
    tk.apply_document(["new", "new", "word"])
    assert tk.close() == ["new", "word"]


if __name__ == "__main__":
    test_top_k_returns_most_frequent_tokens()
    test_apply_document_tf_equivalent_to_apply_document()
    test_close_returns_all_when_fewer_unique_than_k()
    test_reset_clears_state()
    print("all tests passed")
