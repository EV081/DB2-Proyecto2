# Módulo de Extracción — Documentación para el Equipo

## Responsable

Juan Carlos — `src/extraction/`

## Pipeline Unificado

```
Archivo crudo  →  Split  →  Extracción  →  Codebook (Paulo)  →  Índice Invertido (Elmer)  →  Búsqueda
                 [yo]       [yo]           [src/ml/]               [src/engine/]
```

Las 3 modalidades siguen exactamente el mismo flujo: primero se divide en chunks, luego se extraen características de cada chunk.

---

## Cómo importar

```python
from src.extraction import (
    extract_sift_features,
    extract_mfcc_features,
    extract_tfidf_features,
    split_text,
    split_image,
    split_audio,
)
```

---

## Testeo

```bash
source .venv/bin/activate && python3 tests/tests_extraction.py
```

---

## Texto

### Split

```python
from src.extraction import split_text

chunks = split_text(texto_str)
# -> List[str], un elemento por párrafo
```

Divide por `\n\n` (doble salto de línea). Filtra párrafos vacíos o menores a 50 caracteres. Si ningún párrafo cumple, retorna el texto completo como un solo chunk.

### Extracción

```python
from src.extraction import extract_tfidf_features

resultado = extract_tfidf_features("/ruta/archivo.txt")
# -> List[Dict[str, int]]
#     [{"love": 2, "feel": 1}, {"heartbreak": 1, "sad": 1}]
#     Un dict por párrafo, cada dict es {término_stemmed: frecuencia_bruta}
```

Pipeline interno:
```
Archivo .txt
  → split_text() → párrafos
  → tokenize() → minúsculas, elimina puntuación y dígitos
  → remove_stopwords() → filtra stopwords inglés (NLTK)
  → stem() → Porter Stemmer (reduce términos a raíz)
  → compute_tf() → cuenta frecuencias → List[Dict[str, int]]
```

### Lo que necesita Paulo (text_topk.py)

Entrada: todos los `Dict[str, int]` de todos los documentos → debe seleccionar las **k palabras más frecuentes** de toda la colección como codebook textual.

### Lo que necesita Elmer (inverted_index.py)

Cada dict de la lista es un **chunk**. Cada chunk se indexa por separado en el SPIMI. El formato `{término: peso}` es compatible con `similarity.py`.

---

## Imagen

### Split

```python
from src.extraction import split_image

patches = split_image("/ruta/imagen.jpg", patch_size=32, stride=16)
# -> List[ndarray], cada patch es una imagen recortada
```

Parámetros por defecto: `patch_size=32, stride=16` (50% de solapamiento). Adaptados al dataset Fashion (imágenes 80×60 px). Para imágenes más grandes se pueden cambiar.

### Extracción

```python
from src.extraction import extract_sift_features

descriptores = extract_sift_features("/ruta/imagen.jpg")
# -> ndarray (N, 128)  float32
#     N = número de keypoints SIFT encontrados en todos los patches
#     128 = dimensión del descriptor SIFT
```

Pipeline interno:
```
Imagen
  → split_image(32, 16) → patches superpuestos
  → cv2.SIFT_create() → detectAndCompute() por patch
  → concatenar todos los descriptores → ndarray (N, 128)
```

### Lo que necesita Paulo (kmeans_trainer.py + quantizer.py)

Entrada: todos los `ndarray (N, 128)` de todas las imágenes del dataset → debe entrenar K-Means para encontrar **k centroides** (visual words). Luego cuantizar cada imagen: cada descriptor se asigna al centroide más cercano y se construye un histograma de frecuencias por imagen.

---

## Audio

### Split

```python
from src.extraction import split_audio

frames = split_audio("/ruta/audio.mp3", window_ms=100, hop_ms=50)
# -> ndarray (frames, window_len)
#     frames = número de ventanas deslizantes
#     window_len = samples en 100ms
```

Ventanas de 100ms con hop de 50ms (solapamiento del 50%).

### Extracción

```python
from src.extraction import extract_mfcc_features

mfcc = extract_mfcc_features("/ruta/audio.mp3", n_mfcc=13)
# -> ndarray (frames, 13)  float32
#     frames = número de ventanas (≈ 599 por cada 30s de audio @ 22kHz)
#     13 = coeficientes MFCC
```

Pipeline interno:
```
Audio (.mp3/.wav/.flac)
  → split_audio(100, 50) → ventanas deslizantes
  → librosa.feature.mfcc() por ventana
  → concatenar → ndarray (frames, 13)
```

### Lo que necesita Paulo (kmeans_trainer.py + quantizer.py)

Entrada: todos los `ndarray (frames, 13)` de todas las canciones → debe entrenar K-Means para encontrar **k centroides** (acoustic words). Luego cuantizar cada canción: cada frame MFCC se asigna al centroide más cercano y se construye un histograma de frecuencias por canción.

---

## Formato de intercambio entre capas

| Modalidad | Mi output (Split + Extracción) | Lo que Paulo recibe | Lo que Elmer recibe |
|-----------|-------------------------------|---------------------|---------------------|
| **Texto** | `List[Dict[str, int]]` — un dict por párrafo | Todos los términos de todos los docs → seleccionar top-k | Cada dict es un chunk → SPIMI |
| **Imagen** | `ndarray (N, 128)` — SIFT descriptors por imagen | Todos los descriptores de todas las imágenes → K-Means → visual words | Histograma de frecuencias por imagen (Paulo lo genera) |
| **Audio** | `ndarray (frames, 13)` — MFCC por canción | Todos los MFCC de todas las canciones → K-Means → acoustic words | Histograma de frecuencias por canción (Paulo lo genera) |

---

## Dependencias agregadas

```txt
librosa          # Carga de audio y MFCC
soundfile        # Backend de I/O para librosa
opencv-python    # Lectura de imágenes y SIFT
nltk             # Stopwords y Porter Stemmer (texto inglés)
scikit-learn     # K-Means para Paulo
```

Además, `ffmpeg` debe estar instalado en el sistema para que librosa lea MP3.

---

## Datasets

- **App 4 (Recomendación Multimodal):** [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) — ~44K imágenes `.jpg` de 80×60 px + `styles.csv` con descripciones. SIFT se aplica por patches de 32×32 con stride 16.
- **App 2 (Búsqueda Musical):** [FMA dataset](https://github.com/mdeff/fma) (`fma_small.zip`, 8,000 tracks de 30s MP3, 7.2 GiB). El dataset de Spotify NO trae audios crudos.

---

## Consideraciones para el equipo

### Para Paulo

- Mis extractores producen **matrices de descriptores** (imagen/audio) y **dicts de términos** (texto). Tú necesitas iterar sobre todos los archivos del dataset, llamar a mi extractor por cada uno, y acumular los resultados para entrenar K-Means (imagen/audio) o seleccionar top-k (texto).
- `quantizer.py` ya tiene `nearest_centroid()` para asignar un vector a su centroide más cercano.

### Para Elmer

- Texto: cada `Dict[str, int]` en la lista es un chunk indexable. El SPIMI debe mapear cada término a los IDs de chunks donde aparece.
- Imagen/Audio: Paulo te dará un histograma (`Dict[str, int]` con frecuencias de visual/acoustic words) por cada imagen/canción. Eso es equivalente a un documento de texto para `similarity.py`.

### Para Josue

- Cuando conectes las rutas FastAPI, el flujo de una consulta es: recibir archivo → `extract_*_features()` → cuantizar con centroides de Paulo → comparar histogramas con Elmer → devolver top-K.
- Los modelos en `db/models.py` tienen `histogram JSONB` y `embedding vector(8)` — el histograma es lo que produce la cuantización.

### Para Joseph

- El ETL debe: leer cada archivo del dataset → llamar al extractor correspondiente → pasar a Paulo para codebook → una vez entrenado, cuantizar → guardar en BD.
- `split_text()` retorna `List[Dict[str, int]]` — cada elemento es un chunk separado. Para imagen/audio, el extractor retorna la matriz directamente, y la cuantización la hace Paulo.

---

## Archivos del módulo

```
src/extraction/
├── __init__.py        # Exporta todas las funciones públicas
├── split.py           # split_text(), split_image(), split_audio()
├── text_tfidf.py      # extract_tfidf_features()
├── image_sift.py      # extract_sift_features()
└── audio_mfcc.py      # extract_mfcc_features()
```
