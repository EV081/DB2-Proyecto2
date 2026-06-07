import os
import numpy as np

VALID_EXTENSIONS = ('.txt',)

def extract_tfidf_features(text_path: str, n_terms: int = 100, seed: int = 42) -> np.ndarray:
    ext = os.path.splitext(text_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Formato no soportado: '{ext}'. Use: {VALID_EXTENSIONS}")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Texto no encontrado: {text_path}")

    rng = np.random.default_rng(seed)
    return rng.random((1, n_terms)).astype(np.float32)
