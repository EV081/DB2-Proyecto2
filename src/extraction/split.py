import os
from typing import List


# Texto — dividir en párrafos
def split_text(text: str, min_paragraph_chars: int = 20) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    return [p for p in paragraphs if len(p) >= min_paragraph_chars] or paragraphs[:1]


# Imagen
def split_image(image_path: str, patch_size: int = 32, stride: int = 16, max_dim: int = 400):
    import cv2
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    # Redimensionamos la imagen si es demasiado grande para evitar memoria excesiva
    h0, w0 = img.shape[:2]
    if max_dim and max(h0, w0) > max_dim:
        scale = max_dim / max(h0, w0)
        img = cv2.resize(
            img, (int(w0 * scale), int(h0 * scale)),
            interpolation=cv2.INTER_AREA,
        )

    h, w = img.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y + patch_size, x:x + patch_size]
            patches.append(patch)

    if not patches:
        patches.append(img)

    return patches


# Audio — dividir en ventanas
def split_audio(audio_path: str, window_ms: int = 100, hop_ms: int = 100,
                target_sr: int = 16000):
    import librosa
    import numpy as np
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    window_len = int(sr * window_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    if len(y) < window_len:
        y = np.pad(y, (0, window_len - len(y)))

    frames = librosa.util.frame(y, frame_length=window_len, hop_length=hop_len)
    return frames.T


__all__ = [
    "split_text",
    "split_image",
    "split_audio",
]
