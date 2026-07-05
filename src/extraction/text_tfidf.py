import os
import re
import string
from typing import Dict, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from .split import split_text

_ENG_STOPWORDS = None
_STEMMER = PorterStemmer()


def _ensure_nltk_data():
    global _ENG_STOPWORDS
    if _ENG_STOPWORDS is not None:
        return
    try:
        _ENG_STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        _ENG_STOPWORDS = set(stopwords.words('english'))


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[' + re.escape(string.punctuation + '0123456789') + r']', ' ', text)
    return [t for t in text.split() if t]


def remove_stopwords(tokens: List[str]) -> List[str]:
    _ensure_nltk_data()
    return [t for t in tokens if t not in _ENG_STOPWORDS]


def stem(tokens: List[str]) -> List[str]:
    return [_STEMMER.stem(t) for t in tokens]


def compute_tf(tokens: List[str]) -> Dict[str, int]:
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def extract_tfidf_features(text_path: str) -> List[Dict[str, int]]:
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Texto no encontrado: {text_path}")

    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    chunks = split_text(text)
    results = []
    for chunk in chunks:
        tokens = tokenize(chunk)
        tokens = remove_stopwords(tokens)
        tokens = stem(tokens)
        results.append(compute_tf(tokens))
    return results


__all__ = [
    "tokenize",
    "remove_stopwords",
    "stem",
    "compute_tf",
    "extract_tfidf_features",
]
