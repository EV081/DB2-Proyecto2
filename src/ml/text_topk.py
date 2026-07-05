from collections import Counter


class TopKWords:

    def __init__(self, top_k=100):
        self.top_k = top_k
        self._counter = Counter()
        self.bag_of_words = []

    def reset(self):
        self._counter.clear()
        self.bag_of_words = []

    def apply_document(self, tokens):
        self._counter.update(tokens)

    def apply_document_tf(self, tf):
        self._counter.update(tf)

    def close(self):
        self.bag_of_words = [word for word, _ in self._counter.most_common(self.top_k)]
        self._counter.clear()
        return self.bag_of_words
