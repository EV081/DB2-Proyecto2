from __future__ import annotations

import os
import tempfile

import numpy as np

from src.extraction.audio_mfcc import extract_mfcc_features
from src.extraction.image_sift import extract_sift_features
from src.extraction.split import split_audio, split_image, split_text
from src.extraction.text_tfidf import extract_tfidf_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# Helpers
def _txt_path(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write(content)
    f.close()
    return f.name


def _cleanup(path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)


# Split — Texto
def test_split_text_paragraphs() -> None:
    text = (
        "Primer párrafo con suficiente longitud para superar el mínimo de caracteres."
        "\n\n"
        "Segundo párrafo también con longitud suficiente para superar el mínimo."
        "\n\n"
        "Tercer párrafo igualmente largo para poder pasar el filtro de caracteres."
    )
    chunks = split_text(text)
    assert len(chunks) == 3, f"Esperaba 3 chunks, obtuve {len(chunks)}"


def test_split_text_single_paragraph() -> None:
    text = "Texto único sin saltos de línea."
    chunks = split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_filters_empty() -> None:
    text = (
        "Párrafo uno con suficiente longitud para el mínimo de caracteres."
        "\n\n\n\n"
        "Párrafo dos también con longitud suficiente para el mínimo."
    )
    chunks = split_text(text)
    assert len(chunks) == 2


def test_split_text_fallback_small() -> None:
    text = "Corto.\n\nMínimo."
    chunks = split_text(text)
    assert len(chunks) == 1

# Split — Imagen
def test_split_image_returns_patches() -> None:
    img_path = os.path.join(DATA_DIR, "test.png")
    patches = split_image(img_path, patch_size=32, stride=16)
    assert len(patches) > 0
    for p in patches:
        assert p.shape[0] == 32
        assert p.shape[1] == 32


def test_split_image_invalid_path() -> None:
    try:
        split_image("/no/existe.jpg")
        assert False, "Debió lanzar ValueError"
    except (ValueError, FileNotFoundError):
        pass


# Split — Audio
def test_split_audio_returns_frames() -> None:
    audio_path = os.path.join(DATA_DIR, "test.mp3")
    frames = split_audio(audio_path, window_ms=100, hop_ms=50)
    assert frames.ndim == 2
    assert frames.shape[0] > 0
    assert frames.shape[1] > 0


def test_split_audio_invalid_path() -> None:
    try:
        split_audio("/no/existe.mp3")
        assert False, "Debió lanzar excepción"
    except (ValueError, FileNotFoundError):
        pass


# Extracción — Texto TF-IDF
def test_extract_tfidf_empty_text() -> None:
    path = _txt_path("")
    try:
        result = extract_tfidf_features(path)
        assert isinstance(result, list)
        assert len(result) == 0 or (len(result) == 1 and result[0] == {})
    finally:
        _cleanup(path)


def test_extract_tfidf_known_input() -> None:
    path = _txt_path("Love is beautiful. Love is powerful.")
    try:
        result = extract_tfidf_features(path)
        assert len(result) == 1
        tf = result[0]
        assert "love" in tf
        assert tf["love"] == 2
        assert "beauti" in tf
        assert "power" in tf
        for stopword in ("is", "the", "and"):
            assert stopword not in tf, f"Stopword '{stopword}' no fue filtrada"
    finally:
        _cleanup(path)


def test_extract_tfidf_chunked() -> None:
    path = _txt_path("First paragraph about music and songs.\n\nSecond paragraph about fashion and style.")
    try:
        result = extract_tfidf_features(path)
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)
    finally:
        _cleanup(path)


def test_extract_tfidf_removes_digits() -> None:
    path = _txt_path("Test123 with numbers456.")
    try:
        result = extract_tfidf_features(path)
        tf = result[0]
        assert "test123" not in tf
        assert "test" in tf
    finally:
        _cleanup(path)


# Extracción — Imagen SIFT
def test_extract_sift_returns_descriptors() -> None:
    img_path = os.path.join(DATA_DIR, "test.png")
    desc = extract_sift_features(img_path)
    assert isinstance(desc, np.ndarray)
    assert desc.shape[1] == 128
    assert desc.dtype == np.float32


def test_extract_sift_invalid_format() -> None:
    try:
        extract_sift_features("/ruta/imagen.bmp")
        assert False, "Debió lanzar ValueError"
    except ValueError:
        pass


def test_extract_sift_not_found() -> None:
    try:
        extract_sift_features("/no/existe.jpg")
        assert False, "Debió lanzar FileNotFoundError"
    except FileNotFoundError:
        pass


# Extracción — Audio MFCC
def test_extract_mfcc_returns_matrix() -> None:
    audio_path = os.path.join(DATA_DIR, "test.mp3")
    mfcc = extract_mfcc_features(audio_path, n_mfcc=13)
    assert isinstance(mfcc, np.ndarray)
    assert mfcc.ndim == 2
    assert mfcc.shape[1] == 13
    assert mfcc.dtype == np.float32
    assert mfcc.shape[0] > 0


def test_extract_mfcc_invalid_format() -> None:
    try:
        extract_mfcc_features("/ruta/audio.xyz")
        assert False, "Debió lanzar ValueError"
    except ValueError:
        pass


def test_extract_mfcc_not_found() -> None:
    try:
        extract_mfcc_features("/no/existe.wav")
        assert False, "Debió lanzar FileNotFoundError"
    except FileNotFoundError:
        pass


# Pipeline integrado — Split + Extracción
def test_pipeline_text_split_extract() -> None:
    path = _txt_path("Primer chunk de prueba.\n\nSegundo chunk de prueba con más palabras para alcanzar el mínimo.")
    try:
        result = extract_tfidf_features(path)
        assert len(result) >= 1
        assert all(isinstance(r, dict) for r in result)
    finally:
        _cleanup(path)


def test_pipeline_image_split_extract() -> None:
    img_path = os.path.join(DATA_DIR, "test.png")
    desc = extract_sift_features(img_path)
    assert isinstance(desc, np.ndarray)
    assert desc.shape[1] == 128


def test_pipeline_audio_split_extract() -> None:
    audio_path = os.path.join(DATA_DIR, "test.mp3")
    mfcc = extract_mfcc_features(audio_path, n_mfcc=13)
    assert mfcc.shape[1] == 13
    assert mfcc.shape[0] > 0


# Runner
def _run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} pasaron, {failed} fallaron de {len(tests)} tests")
    assert failed == 0, f"{failed} test(s) fallaron"


if __name__ == "__main__":
    _run_all()
