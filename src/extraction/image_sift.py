import os
import cv2
import numpy as np
from .split import split_image

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')


def extract_sift_features(
    image_path: str,
    patch_size: int = 32,
    stride: int = 16,
) -> np.ndarray:
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Formato no soportado: '{ext}'. Use: {VALID_EXTENSIONS}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    patches = split_image(image_path, patch_size=patch_size, stride=stride)
    sift = cv2.SIFT_create()
    all_descriptors = []

    for patch in patches:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        _, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)

    if not all_descriptors:
        return np.empty((0, 128), dtype=np.float32)

    return np.vstack(all_descriptors).astype(np.float32)


__all__ = [
    "extract_sift_features",
]
