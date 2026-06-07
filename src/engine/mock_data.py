"""Datos hardcoded para Hito 1 — modelo del curso (PPT 08, 10, 11).

El modelo IR del curso es:
  - Cada documento es una bolsa de términos -> dict[term, tf_raw].
  - df_t = # documentos donde aparece t (se calcula sobre la colección).
  - N    = # documentos en la colección.
  - El peso de un término es TF-IDF: w = (1 + log10(tf)) * log10(N/df).
  - La similitud es Coseno sobre vectores TF-IDF (normalizados L2).

Para multimedia (PPT 11) el "término" es un codeword del Bag-of-Visual-Words
o Bag-of-Audio-Words, pero el cálculo de DF / IDF / Cosine es idéntico al de
texto. Por eso aquí modelamos las tres modalidades con la misma estructura.

Tipos:
  TermFreqs   = dict[str, int]               # tf-raw de un documento
  Collection  = dict[doc_id, TermFreqs]      # la colección indexable
"""

from __future__ import annotations

TermFreqs = dict[str, int]
Collection = dict[str, TermFreqs]


# ===========================================================================
# Ejemplo del PPT 08 — Jane Austen (slide "similitud coseno entre 3 documentos")
# Usado en los tests para reproducir cos(SS,OP)~0.94, cos(SS,CB)~0.79.
# ---------------------------------------------------------------------------
# Nota del slide: "para simplificar este ejemplo, no hacemos ponderación idf"
# -> en el test contra este caso se usa solo log-TF + length normalization.
# ===========================================================================
AUSTEN_COLLECTION: Collection = {
    "SS_sentido_sensibilidad": {"afecto": 115, "celosa": 10, "chisme": 2,  "borrascoso": 0},
    "OP_orgullo_prejuicio":    {"afecto": 58,  "celosa": 7,  "chisme": 0,  "borrascoso": 0},
    "CB_cumbres_borrascosas":  {"afecto": 20,  "celosa": 11, "chisme": 6,  "borrascoso": 38},
}


# ===========================================================================
# Modalidad TEXTO (TF-IDF + cosine como en PPT 08)
# Colección quemada de "letras de canciones" / textos cortos.
# ===========================================================================
TEXT_COLLECTION: Collection = {
    # cluster: love songs
    "doc_001_love":   {"love": 8, "heart": 6, "you":  5, "night": 2, "dance": 1},
    "doc_002_love":   {"love": 7, "heart": 7, "you":  4, "night": 3, "dance": 2},
    # cluster: party
    "doc_003_party":  {"dance": 9, "night": 8, "club": 6, "music": 5, "love":  1},
    "doc_004_party":  {"dance": 10,"night": 7, "club": 7, "music": 4, "love":  2},
    # cluster: sad
    "doc_005_sad":    {"cry":  9, "alone": 7, "rain": 6, "night": 4, "heart": 3},
    "doc_006_sad":    {"cry":  8, "alone": 8, "rain": 5, "night": 5, "heart": 2},
}

TEXT_QUERIES: Collection = {
    "q_love":   {"love": 7, "heart": 6, "you": 4},
    "q_party":  {"dance": 9, "night": 7, "club": 6},
    "q_sad":    {"cry":   8, "alone": 7, "rain": 5},
}


# ===========================================================================
# Modalidad IMAGEN — Bag of Visual Words (PPT 11)
# El "término" es la palabra visual v_i (centroide de SIFT cuantizado).
# ===========================================================================
IMAGE_COLLECTION: Collection = {
    "img_001_shoe":   {"v0": 9, "v1": 4, "v2": 7,  "v5": 2, "v9": 1},
    "img_002_shoe":   {"v0": 8, "v1": 5, "v2": 6,  "v5": 3, "v9": 2},
    "img_003_shirt":  {"v1": 3, "v3": 8, "v4": 10, "v6": 5, "v8": 2},
    "img_004_shirt":  {"v1": 2, "v3": 9, "v4": 11, "v6": 4, "v8": 3},
    "img_005_watch":  {"v2": 1, "v5": 2, "v7": 12, "v8": 8, "v9": 6},
    "img_006_watch":  {"v2": 2, "v5": 1, "v7": 13, "v8": 7, "v9": 5},
}

IMAGE_QUERIES: Collection = {
    "q_shoe":  {"v0": 8, "v1": 4, "v2": 6, "v5": 2},
    "q_watch": {"v7": 11, "v8": 7, "v9": 5},
}


# ===========================================================================
# Modalidad AUDIO — Bag of Audio Words (MFCC cuantizado, PPT 11)
# ===========================================================================
AUDIO_COLLECTION: Collection = {
    "song_001_rock":    {"a0": 12, "a1": 5,  "a2": 8,  "a4": 3,  "a7": 1},
    "song_002_rock":    {"a0": 10, "a1": 6,  "a2": 7,  "a4": 4,  "a8": 2},
    "song_003_pop":     {"a1": 4,  "a3": 9,  "a5": 11, "a6": 6,  "a9": 2},
    "song_004_pop":     {"a1": 3,  "a3": 10, "a5": 12, "a6": 5,  "a9": 3},
    "song_005_jazz":    {"a2": 2,  "a4": 1,  "a7": 14, "a8": 9,  "a9": 7},
    "song_006_jazz":    {"a2": 3,  "a4": 2,  "a7": 13, "a8": 10, "a9": 6},
    "song_007_silence": {"a0": 1,  "a9": 1},
}

AUDIO_QUERIES: Collection = {
    "q_rocky": {"a0": 9,  "a1": 5,  "a2": 7, "a4": 3},
    "q_poppy": {"a3": 8,  "a5": 10, "a6": 5, "a9": 2},
    "q_jazzy": {"a7": 12, "a8": 8,  "a9": 6},
}


# ===========================================================================
# Catálogo agrupado para iterar en demo / tests
# ===========================================================================
COLLECTIONS: dict[str, Collection] = {
    "text":  TEXT_COLLECTION,
    "image": IMAGE_COLLECTION,
    "audio": AUDIO_COLLECTION,
}

QUERIES: dict[str, Collection] = {
    "text":  TEXT_QUERIES,
    "image": IMAGE_QUERIES,
    "audio": AUDIO_QUERIES,
}
