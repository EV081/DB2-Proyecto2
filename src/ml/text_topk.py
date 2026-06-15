from collections import Counter


class TokKWords:
    """Selecciona las top_k palabras más frecuentes de un conjunto de documentos.

    Uso:
        tk = TokKWords(top_k=100)
        for tokens in lista_de_documentos:
            tk.apply_document(tokens)
        tk.close()
        # tk.bag_of_words contiene las 100 palabras más frecuentes
    """

    def __init__(self, top_k=100):
        """Configura top_k e inicializa contador."""
        self.top_k = top_k
        self._counter = Counter()
        self.bag_of_words = []

    def reset(self):
        """Limpia el contador y el bag_of_words."""
        self._counter.clear()
        self.bag_of_words = []

    def apply_document(self, tokens):
        """Acumula frecuencias de los tokens del documento."""
        self._counter.update(tokens)

    def apply_document_tf(self, tf):
        """Acumula frecuencias desde un dict {token: count} ya pre-contado."""
        self._counter.update(tf)

    def close(self):
        """Selecciona las top_k palabras y limpia el contador."""
        self.bag_of_words = [word for word, _ in self._counter.most_common(self.top_k)]
        self._counter.clear()
        return self.bag_of_words


if __name__ == "__main__":
    docs = [
        ["hola", "mundo", "hola", "test"],
        ["mundo", "foo", "bar", "test", "test"],
        ["hola", "foo", "test"],
    ]

    tk = TokKWords(top_k=3)
    for doc in docs:
        tk.apply_document(doc)
    top = tk.close()
    print("Bag of words:", top)

    tk.reset()
    print("Despues de reset:", tk.bag_of_words)
