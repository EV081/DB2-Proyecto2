from __future__ import annotations

Histogram = dict[str, float]


# ---------------------------------------------------------------------------
# Modalidad: AUDIO  (histogramas de MFCC -> codewords acústicos)
# ---------------------------------------------------------------------------
AUDIO_HISTOGRAMS: dict[str, Histogram] = {
    "song_001_rock":    {"a0": 12, "a1": 5,  "a2": 8,  "a4": 3,  "a7": 1},
    "song_002_rock":    {"a0": 10, "a1": 6,  "a2": 7,  "a4": 4,  "a8": 2},
    "song_003_pop":     {"a1": 4,  "a3": 9,  "a5": 11, "a6": 6,  "a9": 2},
    "song_004_pop":     {"a1": 3,  "a3": 10, "a5": 12, "a6": 5,  "a9": 3},
    "song_005_jazz":    {"a2": 2,  "a4": 1,  "a7": 14, "a8": 9,  "a9": 7},
    "song_006_jazz":    {"a2": 3,  "a4": 2,  "a7": 13, "a8": 10, "a9": 6},
    "song_007_silence": {"a0": 1,  "a9": 1},
}


# ---------------------------------------------------------------------------
# Modalidad: IMAGEN (histogramas de SIFT -> visual words)
# ---------------------------------------------------------------------------
IMAGE_HISTOGRAMS: dict[str, Histogram] = {
    "img_001_shoe":   {"v0": 9,  "v1": 4, "v2": 7,  "v5": 2, "v9": 1},
    "img_002_shoe":   {"v0": 8,  "v1": 5, "v2": 6,  "v5": 3, "v9": 2},
    "img_003_shirt":  {"v1": 3,  "v3": 8, "v4": 10, "v6": 5, "v8": 2},
    "img_004_shirt":  {"v1": 2,  "v3": 9, "v4": 11, "v6": 4, "v8": 3},
    "img_005_watch":  {"v2": 1,  "v5": 2, "v7": 12, "v8": 8, "v9": 6},
    "img_006_watch":  {"v2": 2,  "v5": 1, "v7": 13, "v8": 7, "v9": 5},
}


# ---------------------------------------------------------------------------
# Modalidad: TEXTO  (TF-IDF top-k -> palabras de un codebook lingüístico)
# Frecuencias quemadas a mano; el ranking debería poner juntas las parejas.
# ---------------------------------------------------------------------------
TEXT_HISTOGRAMS: dict[str, Histogram] = {
    "doc_001_love":   {"love": 8, "heart": 6, "you":  5, "night": 2, "dance": 1},
    "doc_002_love":   {"love": 7, "heart": 7, "you":  4, "night": 3, "dance": 2},
    "doc_003_party":  {"dance": 9, "night": 8, "club": 6, "music": 5, "love":  1},
    "doc_004_party":  {"dance": 10,"night": 7, "club": 7, "music": 4, "love":  2},
    "doc_005_sad":    {"cry":  9, "alone": 7, "rain": 6, "night": 4, "heart": 3},
    "doc_006_sad":    {"cry":  8, "alone": 8, "rain": 5, "night": 5, "heart": 2},
}


# ---------------------------------------------------------------------------
# Consultas (queries) hardcoded para probar el ranking
# ---------------------------------------------------------------------------
AUDIO_QUERIES: dict[str, Histogram] = {
    "q_rocky":  {"a0": 9, "a1": 5, "a2": 7, "a4": 3},
    "q_poppy":  {"a3": 8, "a5": 10, "a6": 5, "a9": 2},
    "q_jazzy":  {"a7": 12, "a8": 8, "a9": 6},
}

IMAGE_QUERIES: dict[str, Histogram] = {
    "q_shoe":   {"v0": 8, "v1": 4, "v2": 6, "v5": 2},
    "q_watch":  {"v7": 11, "v8": 7, "v9": 5},
}

TEXT_QUERIES: dict[str, Histogram] = {
    "q_love":   {"love": 7, "heart": 6, "you": 4},
    "q_party":  {"dance": 9, "night": 7, "club": 6},
    "q_sad":    {"cry":   8, "alone": 7, "rain": 5},
}


# ---------------------------------------------------------------------------
# Catálogo: cómo accede el resto del motor a la "colección indexable"
# ---------------------------------------------------------------------------
COLLECTIONS: dict[str, dict[str, Histogram]] = {
    "audio": AUDIO_HISTOGRAMS,
    "image": IMAGE_HISTOGRAMS,
    "text":  TEXT_HISTOGRAMS,
}

QUERIES: dict[str, dict[str, Histogram]] = {
    "audio": AUDIO_QUERIES,
    "image": IMAGE_QUERIES,
    "text":  TEXT_QUERIES,
}
