import os
import numpy as np

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def extract_sift_features(image_path: str, seed: int = 42) -> np.ndarray:
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Formato no soportado: '{ext}'. Use: {VALID_EXTENSIONS}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    rng = np.random.default_rng(seed)
    n_keypoints = rng.integers(50, 200)
    return rng.random((n_keypoints, 128)).astype(np.float32)
