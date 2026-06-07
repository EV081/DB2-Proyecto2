import os
import numpy as np

VALID_EXTENSIONS = ('.wav', '.mp3', '.flac')

def extract_mfcc_features(audio_path: str, n_mfcc: int = 13, seed: int = 42) -> np.ndarray:
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Formato no soportado: '{ext}'. Use: {VALID_EXTENSIONS}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    rng = np.random.default_rng(seed)
    n_frames = rng.integers(100, 500)
    return rng.random((n_frames, n_mfcc)).astype(np.float32)
