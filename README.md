# Informe Técnico — Sistema Multimodal de Recuperación y Búsqueda

**Proyecto 2 · Base de Datos 2 · Ciclo 2026-1 · UTEC**

Este informe documenta el sistema que construimos para el Proyecto 2. Cubre lo pedido en el enunciado: cómo está armado, qué datos consume, cómo se implementó cada pieza, qué salió del benchmark, qué trade-offs vimos, cómo instalarlo y qué hace exactamente el sistema. Los números empíricos se producen con `scripts/bench_full.py` y `scripts/compute_recall.py`; los comentarios sobre por qué elegimos ciertos valores están en las secciones 3 y 5.

---

## 1. Descripción del sistema y arquitectura

### 1.1 Objetivo

El sistema indexa contenido de tres modalidades (texto, audio, imagen) sobre la misma tabla y compara motores de búsqueda distintos bajo el mismo corpus. El enunciado obliga a asignar motores por modalidad:

- **Texto** (letras de canciones, descripciones de productos): SPIMI propio, GIN y GiST sobre `tsvector`.
- **Audio** e **imagen**: SPIMI propio y `pgvector` con HNSW sobre los histogramas cuantizados.

SPIMI se implementa desde cero (`src/engine/inverted_index.py`). GIN, GiST y `pgvector` son índices nativos de PostgreSQL. Todos consumen el mismo pipeline de features y devuelven el mismo formato de respuesta, así que la variable que se mide es realmente el motor.

### 1.2 Vista global

![arquitectura-global](img/arq-glob.png)

> Nota: los indices en el flujo del etl, son los archivos locales, los de Zona de consulta son los mismos pero cargados a ram

### 1.3 Decisiones que vale la pena mencionar

**Todo termina siendo un histograma discreto.** El texto se cuantiza a los top-K stems más frecuentes; el audio, a codewords MFCC via K-Means; la imagen, a codewords SIFT via K-Means. Esto permite reutilizar el mismo índice invertido para las tres modalidades y comparar contra Postgres sin duplicar el pipeline.

**Cada documento vive en una sola fila.** La misma tabla `songs` (o `products`) guarda: el histograma como JSONB (para auditoría), el vector denso como `vector(k)` para `pgvector`, y una columna `tsvector` generada por Postgres para GIN/GiST. SPIMI usa el mismo histograma pero lo materializa como archivos binarios en `indexes/`.

**El router del backend evita SQL desde el cliente.** El cliente elige `(motor, modalidad)`, no una query SQL. Todas las rutas devuelven el mismo `dict` (`id`, `score`, `engine`, `latency_ms`, campos de display), lo cual hace directa la comparación en el frontend.

**El codebook se entrena una sola vez por modalidad.** SPIMI y `pgvector` ven exactamente los mismos histogramas — cualquier diferencia de resultados viene del motor, no del feature.

**Simplificación honesta:** música mezcla dos datasets (Spotify para letras, FMA para audio) que usan nombres de archivo distintos, así que en la práctica ninguna canción termina indexada en las dos modalidades a la vez. El pipeline lo trata como dos flujos independientes en lugar de fingir que hay `bimodalidad`.

---

## 2. Dataset utilizado y características

### 2.1 Fuentes y volúmenes

| Aplicación | Modalidades | Dataset | Volumen | Autenticación |
|---|---|---|---|---|
| **App · Búsqueda Musical Inteligente** | Texto (letras) + Audio (waveform) | Spotify Songs Lyrics (Kaggle `imuhammad/audio-features-and-lyrics-of-spotify-songs`) + FMA-large (Free Music Archive, `fma_large.zip` ~93 GB, recortado a 40k con `FMA_LIMIT=40000`) | ~40 000 canciones con letras EN + 40 000 pistas MP3 | Kaggle token + HTTPS directo |
| **App · Recomendación Multimodal (Fashion)** | Texto (descripciones) + Imagen (JPG) | Fashion Product Images (Kaggle `paramaggarwal/fashion-product-images-dataset`) | ~44 000 productos con imagen + descripción estructurada | Kaggle token |

### 2.2 Preprocesamiento y filtros

**Letras (Spotify).** Se descartan las que no están en inglés usando el campo `language` que trae el CSV de Kaggle. La razón es sencilla: el pipeline de texto usa las stopwords y el Porter stemmer de NLTK, que están entrenados en inglés. Meterle español o portugués deja stopwords sin filtrar y stems basura, lo cual arruina IDF.

**Metadatos de FMA.** El download baja también `fma_metadata.zip`, lee `tracks.csv` con pandas (índice multi-header) y escribe `data/fma_large_flat/metadata.csv` en formato `stem,title,artist,genre`. El ETL de música lo consume junto con la ruta del audio para que los resultados en la API traigan título y artista reales en lugar del número de track.

**Fashion.** El descriptor de cada producto se compone en `download_fashion.py` uniendo `productDisplayName + gender + masterCategory + subCategory + articleType + baseColour + usage + season` separados por punto. No incluye las etiquetas ("`Gender:`", "`Color:`"), porque terminan en todas las filas y IDF las anula — indexarlas era desperdiciar espacio del top-1000.

**Pareo por `stem`.** En Fashion el mismo número (por ej. `1164`) es la imagen (`1164.jpg`) y la descripción (`1164.txt`), así que cada producto se indexa en las dos modalidades. En música las dos fuentes usan esquemas de nombre distintos (`000602.mp3` en FMA vs. `Nirvana_Come_As_You_Are.txt` en Spotify), por lo cual cada canción es texto-solo o audio-solo. Documentado en el código como `_collect_lyrics()` y `_collect_audio()` independientes.

### 2.3 Características para retrieval

| Modalidad | Descriptor | Dimensión por unidad | Unidades por documento (típico) |
|---|---|---|---|
| Texto | Stem (Porter) | 1 token | 30–500 tokens |
| Audio | MFCC (`librosa`, `n_mfcc=13`, ventana 200 ms, hop 100 ms, sr=16 kHz) [6] | 13 | ~300 frames (30 s de pista) |
| Imagen | SIFT (`cv2.SIFT_create`) [4] | 128 | 200–2 000 keypoints |

---

## 3. Detalles de implementación por módulo

### Indexación general

![general-flow](img/general-flow.png)

### Indexación de texto

![text-flow](img/text-flow.png)

### Indexación de audio

![audio-flow](img/audio-flow.png)

### Indexación de imagenes

![imagen-flow](img/imagen-flow.png)


### 3.1 Extracción por modalidad (`src/extraction/`)

**`text_tfidf.py`.** Para cada archivo de texto:

```
leer archivo → text
chunks = split_text(text)                 # separa por párrafos (>= 20 chars)
para chunk en chunks:
    tokens = lowercase(chunk) → strip(puntuación, dígitos) → split(whitespace)
    tokens = [t for t in tokens if t not in NLTK_ENGLISH_STOPWORDS]
    stems  = [PorterStemmer.stem(t) for t in tokens]
    yield Counter(stems)
```

Salen `dict[stem, tf]` por chunk. La fase de codebook luego suma esos counters globalmente.

**`audio_mfcc.py`.** Un solo llamado a `librosa`:

```
y, sr = load_audio(archivo, target_sr=16000, mono=True)
n_fft      = int(sr * 200 / 1000)     # ventana 200 ms → 3200 samples @ 16 kHz
hop_length = int(sr * 100 / 1000)     # hop 100 ms     → 1600 samples @ 16 kHz
mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, center=False)
return mfcc.T   # shape (T, 13), con T ≈ 300 en pistas de 30 s
```

**`image_sift.py`.**

```
img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
detector = cv2.SIFT_create()
_, desc = detector.detectAndCompute(img, None)
return desc     # shape (N_keypoints, 128), típicamente 200 – 2000 keypoints
```

**`feature_cache.py`.** Cada extracción se persiste como `<stem>.npy`. La extracción es cara (MFCC ~200 ms, SIFT ~50 ms por archivo × decenas de miles). El caché evita repetir el trabajo cuando se cambia solo el codebook.

**Muestreo estratificado para K-Means (`aggregate_for_kmeans`).** No cargamos todos los descriptores a RAM (24M vectores para audio, decenas de millones para imagen). En vez de eso:

```
sizes = [ np.load(f, mmap_mode='r').shape[0] for f in feature_files ]
total = sum(sizes)
picks = random_sample(range(total), max_samples, seed=42)   # sin reemplazo
big_matrix = np.empty((max_samples, dim), dtype=float32)
cursor = 0
para cada (f, n) en zip(feature_files, sizes):
    ids_locales = [(p - offset) for p in picks if offset <= p < offset+n]
    if ids_locales:
        big_matrix[cursor:cursor+len(ids_locales)] = np.load(f, mmap_mode='r')[ids_locales]
        cursor += len(ids_locales)
```

Muestreo uniforme sobre descriptores (no sobre archivos): un archivo con 2000 keypoints aporta 4× más chances que uno con 500, lo cual respeta la distribución real de textura. Seed fija para reproducibilidad.

### 3.2 Codebooks y cuantización (`src/ml/`)

La cuantización de descriptores locales (SIFT/MFCC) a codewords vía K-Means es la piedra angular del enfoque **Bag-of-Visual-Words** / **Bag-of-Audio-Words**, introducido por Sivic y Zisserman [5] para retrieval visual.


**Codebook textual (`text_topk.py`).** `TopKWords` es un `Counter` global sobre los stems del corpus:

```
tk = TopKWords(top_k=1000)
para cada (stem_dict) devuelto por text_tfidf sobre cada archivo:
    tk.apply_document_tf(stem_dict)
bag_of_words = tk.close()   # los 1000 stems más frecuentes
```

Elegimos k=1000 por Zipf: en corpus del tamaño de Spotify lyrics (30–40k canciones tras filtro EN) los primeros ~1000 stems cubren >90 % de las ocurrencias. Ir más allá agrega términos con `df` muy chico que IDF ya está pesando fuerte pero cuya contribución al ranking es marginal.

**K-Means propio (`clustering_trainer.KClustering`).** No se usa `sklearn`. El bucle principal es el de un K-Means clásico:

```
inicializar centroides ← np.random.randn(k, dim)      # gaussianos
para iter en 1..max_iter:
    asignaciones = argmin_j ‖X - centroide_j‖²        # por batch de 100 000 filas
    para j en 0..k:
        miembros = X[asignaciones == j]
        si len(miembros) > 0:
            centroide_j ← mean(miembros)              # promedio
        else:
            (dejar centroide anterior — no re-inicializar)
    si max_movimiento < tol:
        break
```

Clusters vacíos se dejan como estaban en la iteración previa. Re-inicializarlos aleatoriamente introduce oscilación entre iteraciones y no mejora el codebook final para corpus del tamaño observado.

| Modalidad | k | `max_samples` | Vectores por centroide |
|---|---:|---:|---:|
| Audio (MFCC 13-dim) | 500 | 50 000 | 100 |
| Imagen (SIFT 128-dim) | 1 024 | 150 000 | 147 |

Regla operativa: `max_samples ≥ 100·k`. Con menos evidencia por centroide K-Means se vuelve inestable con clusters vacíos o con centroides que representan ruido.

**Cuantización (`quantizer.VectorQuantizer`).**

```
para cada descriptor local x del documento:
    j = argmin_j ‖x - centroide_j‖²
    histograma[j] += 1
serializar histograma como { "a_0037": 12, "a_0102": 4, ... }   # texto → JSONB
```

El histograma resultante es la representación del documento en el "vocabulario". Los prefijos `a_`, `v_`, `t_` (audio, visual, texto) evitan colisiones cuando SPIMI indexa las tres modalidades en la misma estructura.

### 3.3 Motor propio SPIMI (`src/engine/inverted_index.py`)

**SPIMI** (Single-Pass In-Memory Indexing, Manning et al. [3]) resuelve indexar un corpus que no cabe entero en RAM. Se procesa en tres fases.

**Fase 1 — inversión por bloques (`spimi_invert`).**

```
posting_lists = {}       # dict[term, PostingList]
posting_count = 0
block_id      = 0

para cada (doc_id, term_freqs) del stream de documentos:
    para cada (term, tf) en term_freqs:
        si tf <= 0: continuar
        si term no está en posting_lists:
            posting_lists[term] = PostingList()
        posting_lists[term].append((doc_id, tf))
        posting_count += 1

    si posting_count >= block_size_postings:
        ordenar keys de posting_lists alfabéticamente
        escribir a "block_<NNNN>.jsonl"   # una linea JSON por termino
        posting_lists = {}
        posting_count = 0
        block_id     += 1

flush del último bloque parcial
```

`PostingList` es una `list` de Python con `__slots__ = ("_data", "_size")` que crece duplicando capacidad (`_grow` desde `_INITIAL_CAPACITY = 4`) — array contiguo pero puro Python, no numpy. Se decidió así porque el hot path es `append`, no cómputo vectorizado.

**Fase 2 — merge externo (`merge_blocks`).** Los `K` bloques ya vienen ordenados individualmente. El K-way merge se resuelve con `heapq.merge` sobre iteradores (`iter_block`) que emiten `(term, postings)` línea a línea:

```
iters  = [iter_block(p) for p in block_paths]
merged = heapq.merge(*iters, key=lambda x: x[0])   # K-way merge por term

current_term, current_postings = None, []
para (term, postings) en merged:
    si term != current_term:
        si current_term is not None:
            _write_term(f, current_term, current_postings, vocab)
        current_term, current_postings = term, list(postings)
    sino:
        current_postings.extend(postings)   # mismo term de otro bloque
si current_term is not None:
    _write_term(f, current_term, current_postings, vocab)

# _write_term(f, term, postings, vocab):
#   offset = f.tell()
#   f.write(json.dumps({"t": term, "p": postings}) + "\n")
#   length = f.tell() - offset
#   vocab[term] = {"offset": offset, "length": length, "df": len(postings)}
```

Salidas del merge:

- `final.postings` — **texto JSONL**, una línea por término: `{"t": term, "p": [[doc_id, tf], ...]}`. Se lee por seek + `read(length)` bytes exactos → 1 seek + 1 read por término.
- `vocab.json` — `dict[term → {"offset": int, "length": int, "df": int}]`. `length` es clave para leer el término completo en una sola llamada sin escanear hasta el `\n`.
- `meta.json` — `{"n_docs": int, "doc_norms": dict[doc_id, float]}`. Las normas por documento se pre-computan en `build_meta` para no recalcular `‖d‖` en cada query.

**Fase 3 — consulta (`InvertedIndex.search_topk`).**

```
q_weights, idf_map = {}, {}
para (term, tf) en query_tf.items():
    si tf <= 0 o term no está en vocab: continuar
    df_t  = vocab[term]["df"]
    idf_t = log10(n_docs / df_t)
    idf_map[term]   = idf_t
    q_weights[term] = log_tf(tf) · idf_t             # log_tf(x) = 1 + log10(x)

q_norm = sqrt(Σ w²  for w in q_weights)
scores = {}
para (term, w_q) en q_weights.items():
    meta = vocab[term]
    fseek(final.postings, meta["offset"])            # 1 seek
    raw = fread(meta["length"])                      # 1 read exacto
    postings = json.loads(raw)["p"]                  # [[doc_id, tf], ...]
    idf_t = idf_map[term]
    para (doc_id, tf_d) en postings:
        w_d = log_tf(tf_d) · idf_t
        scores[doc_id] = scores.get(doc_id, 0.0) + w_q · w_d

para doc_id en scores:
    d_norm = doc_norms.get(doc_id, 0.0)
    scores[doc_id] = scores[doc_id] / (q_norm · d_norm)  if d_norm > 0 else 0.0
return heapq.nlargest(k, scores.items(), key=lambda x: x[1])
```

Ventaja concreta: solo se leen las posting lists de los términos activos de la query (típicamente 2–8 stems para una query de texto, 50–200 codewords para audio/imagen). No se recorre el vocabulario completo.

Se instrumentan dos contadores por consulta (`io_seeks`, `io_read_bytes`) que quedan expuestos en el resultado para el benchmark.

### 3.4 Índices nativos de PostgreSQL

Los campos `lyrics_text` y `description` tienen su columna `tsvector` **generada** por el motor (Postgres FTS [9]). Con esto no hace falta trigger de sincronización:

```sql
lyrics_tsv tsvector GENERATED ALWAYS AS
    (to_tsvector('english', coalesce(lyrics_text, ''))) STORED,
```

**GIN — Generalized Inverted Index (para texto).** Internamente es un B-tree sobre el lexema donde cada hoja apunta a una posting list de `ctid` (`(block_id, offset)` de la fila). Escritura relativamente cara: cada `INSERT` debe actualizar tantas posting lists como lexemas únicos tenga el documento; Postgres mitiga con la _pending list_ (`gin_pending_list_limit`). Lectura óptima para `@@ to_tsquery(...)`: intersección/unión directa de posting lists.

Índices creados: `idx_songs_lyrics_tsv_gin`, `idx_products_desc_tsv_gin`.

**GiST — Generalized Search Tree (para texto).** Árbol balanceado donde cada nodo interno guarda una firma comprimida (signature bloom) del conjunto de lexemas de su subárbol. Escritura barata (`log N` nodos por update); lectura con falsos positivos que se filtran leyendo la fila real. En corpus grandes GIN suele ganarle por eso, pero GiST vale la pena en escenarios de escritura muy frecuente. Aquí lo incluimos para comparar.

Índices creados: `idx_songs_lyrics_tsv_gist`, `idx_products_desc_tsv_gist`.

**pgvector HNSW (para audio/imagen).** Los histogramas cuantizados se guardan como `vector(k)` y se indexan con un grafo Hierarchical Navigable Small World (Malkov & Yashunin [7]) provisto por la extensión `pgvector` [8]. El search es aproximado (ANN) con calidad ajustable vía `ef_search`. Métrica: coseno.

```sql
CREATE INDEX idx_songs_audio_emb_hnsw
    ON songs USING hnsw (audio_emb vector_cosine_ops);
CREATE INDEX idx_products_image_emb_hnsw
    ON products USING hnsw (image_emb vector_cosine_ops);
```

**Por qué pgvector NO aparece en las modalidades de texto.** El enunciado lo asigna explícitamente a imagen y audio ("GIN/GiST para texto, pgvector para imágenes y audio"). Además HNSW asume vectores densos y semánticos (embeddings CNN/BERT), no vectores TF-IDF dispersos donde >90% de los componentes son cero. Aplicarlo a texto TF-IDF funcionaría a nivel de código pero no es el uso para el cual HNSW fue diseñado.

### 3.5 Ejecución de consultas (`src/engine/similarity.py`, `src/db/native_search.py`)

**Consulta por texto (SPIMI).** Ya cubierta en 3.3 fase 3: `search_topk` toma la query, arma `tf_qt`, para cada término abre la posting list y acumula scores.

**Consulta por texto (GIN/GiST).** SQL directo, delegando el matching y el ranking a Postgres. En el cliente pre-armamos la ts_query con **semántica OR** — cada token de la query se separa con `|`:

```python
# native_search._to_or_tsquery
# "black leather jacket" → "black | leather | jacket"
words = re.findall(r"[a-z0-9]+", query.lower())
ts_query = " | ".join(words) if words else "unlikelywordxyz"
```

```sql
SELECT id, title, artist, genre, lyrics_text,
       ts_rank(lyrics_tsv, to_tsquery('english', :ts_query)) AS score
FROM songs
WHERE lyrics_tsv @@ to_tsquery('english', :ts_query)
ORDER BY score DESC, id ASC
LIMIT :limit;
```

Se usa `to_tsquery` (no `plainto_tsquery`) para poder inyectar los `|` de forma explícita — `plainto_tsquery` fuerza AND y descartaría queries multi-palabra que no matchean todos los términos a la vez. Con OR, la comparación contra SPIMI (que también hace unión) es más justa. El planner de Postgres elige entre el índice GIN y el GiST según el nombre del índice existente y el costo estimado; nosotros forzamos uno u otro creando solamente el índice que corresponde y midiendo por separado.

**Consulta por audio o imagen (SPIMI).** Se re-usa el mismo flow que texto — el histograma cuantizado del archivo query es equivalente a un `tf_qt`:

```
mfcc  = extract_mfcc(audio_query)
hist  = { f"a_{j:04d}": count for j,count in quantizer.histogram(mfcc) }
resultados = InvertedIndex.search_topk(hist, k)
```

**Consulta por audio o imagen (pgvector).** El histograma se convierte a vector denso ordenado por keys del codebook y se compara vía distancia coseno HNSW:

```sql
SELECT id, title, artist, genre,
       1.0 - (audio_emb <=> CAST(:query_vec AS vector)) AS score
FROM songs
WHERE audio_emb IS NOT NULL
ORDER BY audio_emb <=> CAST(:query_vec AS vector)
LIMIT :k;
```

**KNN plano (baseline conceptual, no implementado en producción).**

```
para cada d en corpus:
    scores[d.id] = cosine(q_vec, d_vec)
return top-K de scores          # costo O(N · dim)
```

Cita comparativa: con `N=44 000` productos y `dim=1024`, la búsqueda plana calcula ~45M multiplicaciones por query. Sirve como sanity check pero no como sistema.

**Costo por motor.**

- **SPIMI** — costo ≈ `Σ_{t ∈ q} df(t)`, dominado por posting lists de los términos activos. En texto son 2–8 stems; en audio/imagen son 50–200 codewords, cada uno con `df` alto porque los codewords son pocos y frecuentes → posting lists largas.
- **GIN** — costo cercano a constante amortizada para lookup + costo de intersección. Muy rápido en full-text.
- **GiST** — `O(log N)` navegando el árbol + costo de verificación de falsos positivos.
- **pgvector HNSW** — `O(log N)` esperado en navegación del grafo, con recall parametrizable.

### 3.6 Router unificado FastAPI (`src/api/search_service.py`)

El backend no expone SQL al cliente. El dispatcher tiene cuatro funciones públicas, una por combinación válida de app y modalidad:

```python
search_music_lyrics(query_text, engine, k)     # engine ∈ {spimi, gin, gist}
search_music_audio(audio_path, engine, k)      # engine ∈ {spimi, pgvector}
search_fashion_desc(query_text, engine, k)     # engine ∈ {spimi, gin, gist}
search_fashion_image(image_path, engine, k)    # engine ∈ {spimi, pgvector}
```

Cada función sigue el mismo patrón:

```
t0 = perf_counter()
si engine == "spimi":
    idx = get_spimi(app, modality)         # cacheado en memoria
    tf  = convertir_query_a_histograma(query)
    ranking = idx.search_topk(tf, k)
    resultados = enrich_from_db(ranking, columnas=[title, artist, genre, ...])
sino:  # gin, gist, pgvector
    resultados = native_search[engine](query, k)   # SQL nativo
adjuntar_media_urls(resultados)                    # audio_url o image_url por id
log_search(app, modality, engine, query, latencia, len(resultados))
return { app, modality, engine, query, k, latency_ms, results }
```

`_spimi_enrich` es una consulta SQL única (`WHERE stem = ANY(:stems)`) que trae los campos de display para los `k` documentos rankeados. Evita `k+1` queries y mantiene el response idéntico al que devuelven las rutas nativas.

### 3.7 Frontend (`frontend/`)

Next.js 15 (App Router) + pnpm. Componentes: `SearchApp`, `EngineSelector`, `TextSearchForm`, `FileSearchForm`, `ResultCard`, `CompareEnginesPanel`, `LatencyBadge`. Cliente API en `frontend/lib/api.ts`. Estilos con Tailwind. La UI expone las cuatro rutas del backend y permite comparar dos motores lado a lado sobre la misma query.

### 3.8 Análisis de dimensionalidad

Los embeddings tienen dim 1000 (texto), 500 (audio) y 1024 (imagen). Todos son muy dispersos: típicamente 50–200 componentes no-cero, el resto es cero.

En alta dimensión hay dos problemas conocidos que afectan retrieval:

1. **Concentración de distancias.** Para vecinos aleatorios en alta dim, `dist_max / dist_min → 1`. La distancia deja de discriminar.
2. **Volumen exponencial.** Cubrir el espacio con precisión `ε` requiere `(1/ε)^d` particiones. Indexar SIFT sin cuantizar (~500 keypoints × 128 dim por imagen) es directamente inviable.

Cómo lo enfrentamos en concreto:

- **Cuantización a codewords.** Es el paso clave. Una imagen pasa de ~500 × 128 = 64 000 valores continuos a un histograma de 1024 conteos enteros. Se pierde precisión geométrica fina pero se gana estructura discreta que el índice invertido puede manejar.
- **Similitud coseno con normalización L2.** Elimina el sesgo por magnitud. Una canción larga tiene más frames MFCC pero eso no debería inflar su score.
- **TF-IDF con IDF real.** Un término que aparece en todos los documentos tiene `df = N`, entonces `idf = log(1) = 0` y su peso es cero. En Fashion esto anula automáticamente etiquetas ubicuas como `gender:` o `color:` que aparecen en todas las descripciones. Fue una de las razones por las que sacamos las etiquetas del descriptor (ver 2.2): no discriminan y ocupan espacio en el top-1000.
- **Poda del vocabulario textual al top-1000.** Recorta la cola larga de Zipf antes de que llegue al índice.
- **Subsampling estratificado para el K-Means** (sección 3.1).
- **HNSW para pgvector.** Búsqueda aproximada `O(log N)` con recall parametrizable, en lugar de KNN exacto `O(N)`.

---

## 4. Resultados experimentales

### 4.1 Marco experimental

- **Cargas:** 10 000 / 20 000 / 30 000 / 40 000 documentos (submuestreo estratificado del corpus real).
- **Fuente de verdad (ground truth).** La relevancia de un resultado se decide por la **etiqueta de clase** almacenada en Postgres — no por el top-K de ningún motor. Un resultado devuelto es TP cuando su etiqueta coincide con la de la query:
  - `music/lyrics`: columna `songs.genre` (default; alternativa `artist`)
  - `music/audio`: columna `songs.genre` (misma columna que lyrics)
  - `fashion/description`: columna `products.subcategory` (default; alternativa `category`)
  - `fashion/image`: columna `products.subcategory` (misma columna que descripción)

  Estas etiquetas vienen del dataset original (FMA `tracks.csv`, Spotify Kaggle, Fashion Product Images) y se cargan a Postgres durante el ETL. `compute_recall.py` las lee con un único `SELECT` al inicio y las cachea en un dict `{id → label}`. **Todos los motores (SPIMI, GIN, GiST, pgvector) se evalúan contra la misma tabla de labels** — pgvector no tiene ventaja por ser "de la casa".
- **Métricas:**
  - Latencia: avg / p50 / p95 en ms por consulta.
  - Throughput: consultas por segundo (`1000 / avg_ms`).
  - Recall@10: `TP / (relevantes_totales − 1)` — sufre tope estructural cuando la clase tiene cientos de items y `k = 10`.
  - Precision@10: `TP / |resultados devueltos|`.
  - Recall_norm@10: `TP / min(k, relevantes_totales − 1)` — evita que un motor con AND estricto (que devuelve pocos hits) parezca "perfecto" solo por retornar poco.
  - Memoria: RSS pico durante indexación (`/proc/self/status:VmHWM`).
  - Almacenamiento: bytes on-disk del índice (`pg_relation_size`, `_spimi_dir_bytes`).
- **Instrumentación:** `scripts/bench_full.py`, `scripts/compute_recall.py`.
- **Consultas:** 100 queries por combinación `(motor, modalidad, N)`, mismas para todos los motores. Cada query es un item real del dataset: para texto, los primeros ~6 tokens de las lyrics o los primeros 200 chars de la descripción; para audio, el `.mp3` completo; para imagen, la foto del producto. La query se excluye del top-K para no contarla como acierto trivial.

### 4.2 Resultados por app y modalidad

#### App · Música / Lyrics (texto)

Motores comparados: SPIMI, GIN, GiST.

| N docs | Motor | avg ms | p50 ms | p95 ms | QPS | Recall@10 | Precision@10 | Recall_norm@10 | F1@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 000 | SPIMI | 20.17 | 18.95 | 24.77 | 49.58 | 0.003 | 0.033 | 0.032 | 0.005 |
| 10 000 | GIN | 3.18 | 2.18 | 7.08 | 314.50 | 0.003 | 0.065 | 0.039 | 0.006 |
| 10 000 | GiST | 3.05 | 2.17 | 6.73 | 327.57 | 0.003 | 0.065 | 0.039 | 0.006 |
| 20 000 | SPIMI | 41.51 | 38.42 | 54.66 | 24.09 | 0.001 | 0.021 | 0.020 | 0.002 |
| 20 000 | GIN | 4.80 | 2.93 | 11.87 | 208.55 | 0.001 | 0.014 | 0.011 | 0.001 |
| 20 000 | GiST | 4.36 | 2.58 | 9.93 | 229.35 | 0.001 | 0.014 | 0.011 | 0.001 |
| 30 000 | SPIMI | 62.37 | 58.40 | 82.84 | 16.03 | 0.000 | 0.005 | 0.005 | 0.000 |
| 30 000 | GIN | 6.70 | 4.33 | 17.43 | 149.19 | 0.001 | 0.032 | 0.020 | 0.002 |
| 30 000 | GiST | 6.38 | 4.00 | 17.39 | 156.81 | 0.001 | 0.032 | 0.020 | 0.002 |
| 40 000 | SPIMI | 82.73 | 75.81 | 120.07 | 12.09 | 0.000 | 0.014 | 0.014 | 0.000 |
| 40 000 | GIN | 8.54 | 5.72 | 25.46 | 117.05 | 0.001 | 0.033 | 0.013 | 0.002 |
| 40 000 | GiST | 7.69 | 4.87 | 24.70 | 130.05 | 0.001 | 0.033 | 0.013 | 0.002 |

![Latencia — music/lyrics](benchmark/graphs/music_lyrics_latency.png)

![Throughput — music/lyrics](benchmark/graphs/music_lyrics_throughput.png)

**Escalabilidad music/lyrics:**

![avg ms vs N](benchmark/graphs/scale_music_lyrics_avg_ms.png)

![p95 ms vs N](benchmark/graphs/scale_music_lyrics_p95_ms.png)

![throughput QPS vs N](benchmark/graphs/scale_music_lyrics_throughput_qps.png)

![RSS pico MB vs N](benchmark/graphs/scale_music_lyrics_rss_peak_mb.png)

![Tamaño índice MB vs N](benchmark/graphs/scale_music_lyrics_index_size_mb.png)

#### App · Música / Audio

| N docs | Motor | avg ms | p50 ms | p95 ms | QPS | Recall@10 | Precision@10 | Recall_norm@10 | F1@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 000 | SPIMI | 138.03 | 129.25 | 192.37 | 7.24 | 0.006 | 0.380 | 0.342 | 0.011 |
| 10 000 | pgvector | 78.58 | 82.87 | 87.90 | 12.73 | 0.005 | 0.343 | 0.309 | 0.009 |
| 20 000 | SPIMI | 188.29 | 179.14 | 274.16 | 5.31 | 0.003 | 0.314 | 0.283 | 0.005 |
| 20 000 | pgvector | 72.97 | 70.44 | 107.61 | 13.71 | 0.002 | 0.293 | 0.264 | 0.004 |
| 30 000 | SPIMI | 246.95 | 232.37 | 385.97 | 4.05 | 0.003 | 0.297 | 0.267 | 0.005 |
| 30 000 | pgvector | 67.23 | 68.36 | 78.92 | 14.87 | 0.002 | 0.253 | 0.228 | 0.004 |
| 40 000 | SPIMI | 320.44 | 311.11 | 508.82 | 3.12 | 0.002 | 0.285 | 0.257 | 0.003 |
| 40 000 | pgvector | 83.16 | 74.96 | 117.24 | 12.02 | 0.001 | 0.249 | 0.224 | 0.002 |

![Latencia — music/audio](benchmark/graphs/music_audio_latency.png)

**Escalabilidad music/audio:**

![avg ms vs N](benchmark/graphs/scale_music_audio_avg_ms.png)

![p95 ms vs N](benchmark/graphs/scale_music_audio_p95_ms.png)

![throughput QPS vs N](benchmark/graphs/scale_music_audio_throughput_qps.png)

![RSS pico MB vs N](benchmark/graphs/scale_music_audio_rss_peak_mb.png)

![Tamaño índice MB vs N](benchmark/graphs/scale_music_audio_index_size_mb.png)

#### App · Fashion / Description (texto)

Motores comparados: SPIMI, GIN, GiST.

| N docs | Motor | avg ms | p50 ms | p95 ms | QPS | Recall@10 | Precision@10 | Recall_norm@10 | F1@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 000 | SPIMI | 19.59 | 18.21 | 23.42 | 51.05 | 0.019 | 0.950 | 0.855 | 0.036 |
| 10 000 | GIN | 1.85 | 1.64 | 3.61 | 540.89 | 0.016 | 1.000 | 0.290 | 0.032 |
| 10 000 | GiST | 1.90 | 1.69 | 3.75 | 526.72 | 0.016 | 1.000 | 0.290 | 0.032 |
| 20 000 | SPIMI | 37.13 | 34.48 | 46.77 | 26.93 | 0.018 | 0.974 | 0.879 | 0.034 |
| 20 000 | GIN | 2.19 | 1.83 | 4.18 | 456.59 | 0.006 | 1.000 | 0.463 | 0.012 |
| 20 000 | GiST | 2.25 | 1.86 | 4.06 | 443.92 | 0.006 | 1.000 | 0.463 | 0.012 |
| 30 000 | SPIMI | 58.34 | 52.70 | 123.76 | 17.14 | 0.005 | 0.972 | 0.878 | 0.011 |
| 30 000 | GIN | 2.86 | 2.20 | 6.40 | 350.23 | 0.003 | 1.000 | 0.476 | 0.005 |
| 30 000 | GiST | 2.95 | 2.28 | 6.75 | 338.44 | 0.003 | 1.000 | 0.476 | 0.005 |
| 40 000 | SPIMI | 76.21 | 69.14 | 145.33 | 13.12 | 0.009 | 0.988 | 0.892 | 0.017 |
| 40 000 | GIN | 3.18 | 2.44 | 6.98 | 314.33 | 0.003 | 1.000 | 0.514 | 0.006 |
| 40 000 | GiST | 3.02 | 2.32 | 6.42 | 330.90 | 0.003 | 1.000 | 0.514 | 0.006 |

![Latencia — fashion/description](benchmark/graphs/fashion_desc_latency.png)

**Escalabilidad fashion/description:**

![avg ms vs N](benchmark/graphs/scale_fashion_desc_avg_ms.png)

![p95 ms vs N](benchmark/graphs/scale_fashion_desc_p95_ms.png)

![throughput QPS vs N](benchmark/graphs/scale_fashion_desc_throughput_qps.png)

![RSS pico MB vs N](benchmark/graphs/scale_fashion_desc_rss_peak_mb.png)

![Tamaño índice MB vs N](benchmark/graphs/scale_fashion_desc_index_size_mb.png)

#### App · Fashion / Image

| N docs | Motor | avg ms | p50 ms | p95 ms | QPS | Recall@10 | Precision@10 | Recall_norm@10 | F1@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 000 | SPIMI | 597.99 | 601.86 | 866.26 | 1.67 | 0.006 | 0.540 | 0.486 | 0.011 |
| 10 000 | pgvector | 178.81 | 178.34 | 236.28 | 5.59 | 0.005 | 0.560 | 0.505 | 0.010 |
| 20 000 | SPIMI | 1043.97 | 973.28 | 1800.81 | 0.96 | 0.002 | 0.428 | 0.385 | 0.004 |
| 20 000 | pgvector | 162.03 | 156.15 | 197.95 | 6.17 | 0.005 | 0.451 | 0.406 | 0.009 |
| 30 000 | SPIMI | 1594.97 | 1434.70 | 2773.07 | 0.63 | 0.001 | 0.488 | 0.439 | 0.003 |
| 30 000 | pgvector | 161.39 | 157.01 | 197.38 | 6.20 | 0.001 | 0.515 | 0.465 | 0.003 |
| 40 000 | SPIMI | 2168.92 | 2107.04 | 3781.60 | 0.46 | 0.002 | 0.396 | 0.356 | 0.004 |
| 40 000 | pgvector | 167.22 | 162.22 | 212.76 | 5.98 | 0.002 | 0.473 | 0.426 | 0.005 |

![Latencia — fashion/image](benchmark/graphs/fashion_image_latency.png)

**Escalabilidad fashion/image:**

![avg ms vs N](benchmark/graphs/scale_fashion_image_avg_ms.png)

![p95 ms vs N](benchmark/graphs/scale_fashion_image_p95_ms.png)

![throughput QPS vs N](benchmark/graphs/scale_fashion_image_throughput_qps.png)

![RSS pico MB vs N](benchmark/graphs/scale_fashion_image_rss_peak_mb.png)

![Tamaño índice MB vs N](benchmark/graphs/scale_fashion_image_index_size_mb.png)

### 4.4 Costo de indexación, memoria y almacenamiento

Valores medidos sobre corpus N=40 000 (fashion N=40 000, music_audio N=40 000, music_lyrics N=40 000). RSS reportado corresponde al pico observado durante la fase de query en el proceso benchmark (no exclusivamente al build).

| Motor | Modalidad | RSS pico proc. (MB) | Índice on-disk (MB) |
|---|---|---:|---:|
| SPIMI | texto (lyrics) | 396.10 | 132.53 |
| SPIMI | texto (desc) | 505.16 | 9.59 |
| SPIMI | audio | 496.96 | 27.56 |
| SPIMI | imagen | 641.84 | 96.40 |
| GIN | texto (lyrics) | 396.10 | 16.50 |
| GIN | texto (desc) | 505.16 | 3.00 |
| GiST | texto (lyrics) | 396.10 | 12.84 |
| GiST | texto (desc) | 505.16 | 2.95 |
| pgvector | audio | 496.97 | 102.94 |
| pgvector | image | 646.42 | 307.34 |

**Observaciones:**

- **SPIMI en texto de letras es ~8× más pesado en disco que GIN** (132 MB vs 16.5 MB) por almacenar postings crudas como JSON. GIN comprime con posting trees.
- **SPIMI en audio/imagen es 3-4× más liviano que pgvector** porque histogramas BoVW sparse (~500 dimensiones no-cero) ocupan menos que embeddings densos (500/1024 dimensiones float).
- **pgvector con HNSW paga overhead de grafo** — en fashion image, 307 MB para 40k vectores × 1024 dim vs. SPIMI con 96 MB para el mismo corpus.

### 4.5 Impacto de dimensionalidad

El tamaño del codebook (número de "términos únicos" en el vocabulario) afecta directamente la latencia de SPIMI, porque cada query trae ~50-200 posting lists de disco y las escanea completamente para computar TF-IDF. En texto natural la distribución de frecuencia es Zipf (pocos stems frecuentes + cola larga), pero en modalidades no-textuales (BoVW audio/imagen) todos los codewords tienden a aparecer en todos los documentos, inflando cada posting list.

| Modalidad | Dim (codebook k) | Descriptores por doc | Latencia SPIMI @ N=40k | Motor nativo @ N=40k | Ratio SPIMI / nativo |
|---|---:|---:|---:|---:|---:|
| Texto (lyrics) | 1 000 | ~50-200 stems únicos | 82.7 ms | GIN 8.5 ms | 10× |
| Texto (desc) | 1 000 | ~10-50 stems únicos | 76.2 ms | GIN 3.2 ms | 24× |
| Audio (MFCC) | 500 | ~300 frames | 320.4 ms | pgvector 83.2 ms | 4× |
| Imagen (SIFT) | 1 024 | ~500-2000 keypoints | 2 168.9 ms | pgvector 167.2 ms | 13× |

**Observaciones:**

- **Codebook chico + descriptores densos = SPIMI muere.** En imagen, cada documento aporta cientos de descriptores SIFT que se cuantizan a los 1024 codewords → cada codeword aparece en ~40-60% de los documentos. Las posting lists son enormes y el scoring TAAT recorre toda la lista.
- **En texto natural funciona porque el vocabulario es sparse por documento.** Cada canción usa ~50-200 stems únicos de un vocabulario de 1000; las posting lists son de ~500-2000 documentos, mucho más manejables.
- **La dimensionalidad del embedding denso NO afecta a HNSW proporcionalmente.** Pasar de dim=500 (audio) a dim=1024 (imagen) solo duplica la latencia de pgvector (83 → 167 ms), mientras SPIMI se degrada 7× (320 → 2168 ms). HNSW navega el grafo logarítmicamente en la cardinalidad del corpus, no en la dimensión de los vectores.
- **Trade-off implícito con el codebook.** Aumentar k a >10 000 en imagen haría las posting lists más sparse y aceleraría SPIMI, pero rompe la premisa "codebook pequeño" del enunciado y encarece K-Means.

### 4.6 Lectura de los resultados

Los índices nativos de Postgres (GIN / GiST / pgvector) le pasan por encima al SPIMI propio en velocidad, siempre. En algunas modalidades por un factor de 4x, en otras por 13x, en las peores por 25x. No es sorpresa: son motores en C con años de tuning contra un motor en Python de un cuatrimestre. Lo interesante es *cuánto* pierde, en qué escala, y dónde SPIMI se defiende.

**music/lyrics (texto).** A N=40k, SPIMI queda ~10x más lento que GIN/GiST (82 ms vs 8 ms). Además solo devuelve resultados en 78/100 queries mientras GIN/GiST responden 93/100 — hay un hueco de cobertura, no solo de latencia. La brecha en velocidad es real pero manejable.

**music/audio (MFCC codebook).** pgvector va ~4x más rápido (83 ms vs 320 ms de SPIMI), pero acá pasa algo curioso: **SPIMI le gana levemente en precisión** (precision@10 0.285 vs 0.249). El motor propio es más lento pero un poquito más certero clasificando por género — porque el ranking exacto por coseno sobre el histograma no pierde información, mientras HNSW aproxima.

**fashion/desc (texto de descripciones).** Aquí SPIMI queda peor parado en velocidad relativa: **~24x más lento** (76 ms vs 3 ms). Pero fíjate el recall: SPIMI evalúa las 100 queries con precisión 0.987, GIN/GiST solo evalúan 44/100 (aunque con precisión 1.0 en las que sí responden). GIN/GiST es "cuando responde, acierta, pero responde menos veces". SPIMI cubre todo con casi la misma precisión.

**fashion/image (SIFT + BoVW).** El caso más doloroso: **2 segundos por query vs 167 ms**. 13x más lento y a un nivel que ya no es usable para UI interactiva. Encima pgvector tiene mejor precision (0.47 vs 0.40). No hay debate: la búsqueda por imagen se la lleva pgvector limpio.

**Escalabilidad — cómo crece la latencia al 4x los datos (10k → 40k):**

| modalidad | motor | 10k → 40k | factor |
|---|---|---|---:|
| lyrics | SPIMI | 20 → 82 ms | **4.1×** |
| lyrics | GIN | 3.2 → 8.5 ms | 2.7× |
| audio | SPIMI | 138 → 320 ms | **2.3×** |
| audio | pgvector | 78 → 83 ms | 1.06× |
| desc | SPIMI | 19 → 76 ms | **4.0×** |
| desc | GIN | 1.85 → 3.2 ms | 1.7× |
| image | SPIMI | 598 → 2168 ms | **3.6×** |
| image | pgvector | 178 → 167 ms | 0.94× |

Dos lecturas de esta tabla: (1) **SPIMI escala casi lineal con N** en texto (4x datos → 4x latencia), que es lo esperado de un scan sobre posting lists sin skip lists. (2) **pgvector es prácticamente plano** — 167 ms con 10k o con 40k. Su costo está dominado por overhead del HNSW, no por N; con más datos la brecha se agranda todavía más.

**Disco y RAM (N=40k).** No todo es velocidad — SPIMI gana en almacenamiento en modalidades no-textuales:

- lyrics: SPIMI 132 MB vs GIN 16 MB → **8× más grande SPIMI** (postings como JSON crudo).
- desc: SPIMI 9.6 MB vs GIN 3 MB → 3× más.
- audio: SPIMI 27 MB vs pgvector 103 MB → **SPIMI gana en disco**.
- image: SPIMI 96 MB vs pgvector 307 MB → **SPIMI gana en disco de nuevo**.

RSS pico es parecido entre motores (400-650 MB) porque comparten el proceso benchmark. El `rss_delta` de SPIMI en audio (100 MB) muestra que carga estructura pesada al abrir el índice; los de Postgres son ~0 porque el trabajo pesado está en el server, no en el cliente.

**Sobre los números de recall.** Los `recall_at_k` están todos por el suelo (0.001 a 0.02). **No es un bug**: la métrica usa como ground truth "todos los items del mismo género/subcategoría", y devolver 10 no puede cubrir cientos de items de una clase. Por eso la métrica útil es `precision@k` (y `recall_norm@k` que normaliza por K). Ahí sí se ve: fashion_desc con SPIMI está en 0.95-0.99 (casi perfecto), audio anda en 0.28-0.38, imagen en 0.4-0.5, y lyrics es un desastre en todos los motores (0.01-0.06) — la etiqueta "genre" simplemente no correlaciona bien con similitud coseno sobre TF-IDF de letras.

**En una frase.** SPIMI cumple: funciona, es correcto, escala predeciblemente, y en corpus 10k-20k se banca sin problemas. Pero Postgres es más rápido en todos los escenarios y la brecha se hace insalvable en fashion/image (13x) donde SPIMI ya no es interactivo. La conclusión del proyecto es la esperada: **implementar tu propio motor sirve para entender el costo real de IR desde cero, no para reemplazar a Postgres en producción**.

---

## 5. Análisis de trade-offs y conclusiones

### 5.1 Trade-offs por motor

**Latencia y throughput en texto.**
Los motores nativos de Postgres (GIN, GiST) dominan claramente en texto sobre todos los tamaños de corpus. Para `fashion/desc` con N=40k: GIN alcanza **314 QPS** (avg 3.18 ms) y GiST **331 QPS** (avg 3.02 ms), mientras SPIMI logra apenas **13 QPS** (avg 76 ms) — una diferencia de ~24×. En `music/lyrics` la brecha es similar: GIN a 117 QPS vs SPIMI a 12 QPS. La razón es doble: GIN/GiST integran búsqueda + fetch en un solo SQL, mientras SPIMI paga un round-trip adicional a Postgres para enriquecer los resultados con metadatos, y su almacenamiento JSON de postings es más costoso de deserializar que el formato binario comprimido de GIN. GiST empata o gana marginalmente a GIN por su firma más compacta que reduce cache misses, aunque en corpus más grandes la ventaja se difumina.

**Latencia en audio/imagen.**
pgvector con HNSW gana consistentemente en audio e imagen para corpus medianos–grandes. Para `fashion/image` a N=40k: pgvector 167 ms vs SPIMI 2168 ms — **13× más lento SPIMI**. En audio la brecha es menor (SPIMI 320 ms vs pgvector 83 ms, ~4×) porque los histogramas BoVW son más sparse. La razón fundamental es que SPIMI en modalidades no-textuales sufre de posting lists muy largas: con solo 500 (audio) o 1024 (imagen) codewords y 40k documentos, cada codeword aparece en cientos de documentos, forzando lectura y scoring sobre listas enormes. HNSW en cambio hace navegación logarítmica del grafo. SPIMI solo sería competitivo si aumentáramos el codebook a K > 10k para hacer las posting lists más sparse — pero eso rompería la asunción "codebook chico" del proyecto.

**Precisión.**
Con ground truth categórica (`genre` para música, `subcategory` para fashion), las métricas absolutas son bajas por el techo estructural: recall clásico nunca supera 2% porque cada clase tiene cientos de miembros y K=10. Se reportan tres métricas complementarias:

- **Recall@10 (clásico):** TP / (relevantes en corpus − 1). Sufre el techo estructural.
- **Precision@10:** TP / (resultados devueltos). Puede inflarse si el motor devuelve pocos resultados.
- **Recall_norm@10:** TP / min(K, relevantes − 1). Normaliza por el máximo alcanzable en top-K. Cuando |clase| ≥ K, se comporta como precision; cuando |clase| < K, se comporta como recall clásico. **Métrica más informativa para comparar motores.**

Los resultados por modalidad a N=40k:

- **Fashion/desc:** SPIMI recall_norm = **0.892** (vs GIN/GiST 0.514). Aquí precision engaña: GIN muestra 1.000 solo porque devuelve muy pocos resultados por query (queries FTS con AND estricto → 1-2 hits) y ese poco es correcto. Recall_norm normaliza por K=10, exponiendo que SPIMI cubre casi el 90% del top-10 alcanzable mientras GIN apenas la mitad.
- **Music/audio:** SPIMI recall_norm = **0.257** vs pgvector 0.224. SPIMI supera marginalmente al HNSW gracias al ranking exacto por coseno sobre BoVW.
- **Fashion/image:** pgvector 0.426 vs SPIMI 0.356. pgvector gana porque el embedding denso captura información espacial que SIFT+BoVW pierde.
- **Music/lyrics:** todos los motores en recall_norm ≤ 0.04. El vocabulario de canciones cruza géneros masivamente ("love", "heart", "night" son universales), por lo que la clasificación por género vía similaridad textual es inherentemente débil. La comparación real en esta modalidad es por latencia.

**Nota metodológica.** Precision alta con GIN/GiST en text search debe interpretarse con cuidado: cuando el motor devuelve 1-2 resultados y los dos son correctos, precision = 1.0, pero el motor NO está siendo mejor — solo está siendo más selectivo. Recall_norm evita ese sesgo dividiendo siempre por K = 10 (o menos si la clase es tiny).

**Escalabilidad.**

- **SPIMI.** El costo escala con el largo de las posting lists de los términos de la query. En texto crece lentamente (`df` de un stem crece ~logarítmicamente con `N`). En audio/imagen crece rápido porque los codewords son pocos (500 / 1024) y muy frecuentes.
- **GIN.** Costo casi constante amortizado por lookup; la escritura penaliza pero el read es óptimo.
- **pgvector HNSW.** `O(log N)` esperado en la navegación del grafo.

**Memoria e IO.**

- **SPIMI.** Streaming desde disco. Bajo consumo de RAM en query; el bottleneck es la cantidad de seeks. Bueno cuando el índice no cabe en RAM; malo con discos lentos.
- **pgvector HNSW.** El grafo se sube a `shared_buffers` de Postgres. Alta RAM al principio, casi cero seeks luego. Bueno para latencia; caro cuando `N` es grande y el índice pesa.
- **GIN/GiST.** Postgres gestiona buffers + WAL. Está entre los dos.

### 5.2 Límite del enfoque BoVW/BoAW

SIFT + Bag-of-Visual-Words captura textura y gradientes locales, no "objeto". Es fuerte para retrieval de la misma imagen o de una instancia casi idéntica; es débil para retrieval por categoría (encontrarme "otra camiseta" a partir de una camiseta). Cuando probamos consultas visuales en Fashion, las primeras respuestas contienen productos con textura similar a la query (jerseys deportivos con logos grandes, patrones de estampado), pero no necesariamente el mismo tipo de prenda.

Los sistemas modernos usan features aprendidos por CNN o transformers (ResNet, CLIP). Subir `k` en BoVW no cierra la brecha semántica: el descriptor SIFT no ve "camiseta", ve gradientes. Cambiar el descriptor sí — pero eso sale del alcance del enunciado, que exige un motor propio, no un motor SOTA.

### 5.3 Elecciones de tamaño de codebook

- **Texto — `k = 1000`** (Zipf [3]). Los primeros 1000 stems Porter cubren >90 % de las ocurrencias del corpus.
- **Audio — `k = 500`** (Nanni et al. [1]). Empezamos en 200, subimos a 500 porque `recall@10` sobre FMA seguía ambiguo. Nanni reporta ganancia hasta `k = 1024`.
- **Imagen — `k = 1024`** (Yang et al. [2]). Rango típico en BoVW; con `max_samples = 150 000` da ~147 vectores por centroide, suficiente para evitar clusters vacíos con nuestro K-Means propio.

### 5.4 Conclusiones experimentales

Del sweep sobre 4 tamaños (10k, 20k, 30k, 40k) emerge un patrón claro por modalidad:

**Texto (lyrics + descripciones).** Los motores nativos de Postgres (GIN, GiST) son la elección práctica en producción. GIN sostiene 117-540 QPS según la sección con avg 3-9 ms, mientras SPIMI se degrada de 51 QPS a N=10k hasta 12 QPS a N=40k. GiST empata a GIN en latencia con firma más compacta, aunque su recall efectivo depende de la re-verificación posterior sobre `tsvector`. **Recomendación:** para búsqueda textual sobre corpus > 10k documentos, GIN es la opción por default.

**Audio.** pgvector con HNSW gana consistentemente en latencia (67-83 ms vs 138-320 ms de SPIMI) y throughput (12-15 QPS vs 3-7 QPS). SPIMI recupera algo en precision@10 sobre pgvector (0.285 vs 0.249 a N=40k) porque el ranking exacto por coseno sobre BoVW no pierde información. **Recomendación:** pgvector para producción; SPIMI solo si se requiere ranking determinístico auditable.

**Imagen.** pgvector domina con margen amplio: 167 ms vs 2168 ms de SPIMI a N=40k (~13×). SPIMI en imagen sufre severamente por posting lists largas — cada uno de los 1024 codewords aparece en cientos de imágenes, forzando lecturas masivas. **Recomendación:** pgvector claramente superior en imagen; SPIMI no es viable en producción a esta escala.

**Régimen por tamaño de corpus:**

- **Corpus pequeño (< 5k):** SPIMI es competitivo y su ranking exacto tiene valor si se necesita reproducibilidad total.
- **Corpus mediano (5k-30k):** motores nativos ganan por 3-10× en latencia; SPIMI queda como referencia académica.
- **Corpus grande (> 30k):** solo motores nativos (GIN para texto, pgvector para vectores) son viables. SPIMI se degrada superlinealmente por overhead de seeks + parsing JSON.

**Sobre las métricas de recall/precision.** El ground truth categórico (`genre`/`subcategory`) impone un techo estructural: recall clásico nunca supera 2% con K=10 sobre clases de cientos de miembros. La **precision@10** es equivalente al **recall normalizado por el máximo alcanzable** cuando el tamaño de clase supera K, y es la métrica más informativa para comparar motores en este benchmark. F1@10 replica el patrón de recall clásico y no aporta señal adicional.

### 5.5 Comparación con sistemas del mercado

| Sistema | Qué hace | Modalidades | En qué es mejor que este proyecto | En qué es peor (o irrelevante para el proyecto) |
|---|---|---|---|---|
| Elasticsearch + dense_vector | Inverted index + kNN sobre embeddings | Texto + embedding externo | Producción, sharding, réplicas, monitoreo | No expone el motor de indexado propio; embeddings los tenés que producir vos |
| Milvus / Weaviate / Pinecone | ANN vectorial puro | Cualquiera con embedding | Escala a millones de vectores | No hay FTS nativo; es una caja negra pedagógicamente |
| Solr / Lucene | Índice invertido clásico | Texto | BM25 muy pulido, faceting, highlighting | Cuantización de audio/imagen es manual (fuera del scope) |
| Google Vertex AI Matching Engine | ANN + CLIP-like | Multimodal semántico | Retrieval semántico real | Cerrado, caro, requiere CNN entrenada |
| **Este proyecto** | SPIMI propio + GIN/GiST + pgvector | Texto + Audio + Imagen | Puede ver dentro de cada motor y comparar bajo el mismo corpus | K-Means propio es lento vs FAISS; SIFT + BoVW no capta "objeto" |

---

## 6. Instrucciones de instalación y uso

Guía resumida — la versión completa está en el `README.md`.

### 6.1 Requisitos

- Python 3.10+
- Docker + Docker Compose
- pnpm (para el frontend)
- Token de Kaggle (`KAGGLE_API_TOKEN` formato `KGAT_…`) — necesario para Spotify y Fashion.

### 6.2 Setup del backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # editar POSTGRES_PASSWORD, KAGGLE_API_TOKEN, DATABASE_URL

docker compose up -d    # PostgreSQL 16 + pgvector; init.sql se aplica solo
docker compose ps       # confirmar "healthy"
```

### 6.3 Setup con un solo comando

```bash
bash scripts/setup_all.sh
```

Descarga (idempotente), levanta Postgres, corre los ETLs y arranca FastAPI en <http://127.0.0.1:8000> (docs en `/docs`). Flags útiles:

```bash
bash scripts/setup_all.sh --no-serve         # ETL sin levantar API
bash scripts/setup_all.sh --force-index      # re-indexar sin re-descargar
bash scripts/setup_all.sh --only fma         # un solo dataset
bash scripts/setup_all.sh --port 8080
```

### 6.4 Setup del frontend

```bash
cd frontend
pnpm install
pnpm dev            # http://localhost:3000
```

### 6.5 Benchmarks y reproducibilidad

```bash
# Reset limpio (borra datos y volumen Postgres)
docker compose down -v
rm -rf data indexes codebooks

# Setup completo
bash scripts/setup_all.sh

# Benchmarks
python3 scripts/bench_full.py --queries 100 --k 10 --out-json benchmark_results.json
python3 scripts/compute_recall.py --k 10
```

---

## 7. Ejemplos de uso

Se implementan **dos** de las Ideas Sugeridas del enunciado, ambas con GUI y con endpoints REST.

### 7.1 App · Búsqueda Musical Inteligente

**GUI.**

1. Elegir la modalidad: **Letra** (input de texto) o **Audio** (subir `.mp3`).
2. Elegir el motor: `spimi` / `gin` / `gist` para letras; `spimi` / `pgvector` para audio.
3. Marcar (opcional) **"Comparar los 2 motores a la vez"** para ver dos rankings lado a lado.
4. Cada resultado trae: título, artista, género, letra completa, score, latencia y botón de reproducción del audio.

**REST — búsqueda por letra:**

```bash
curl "http://127.0.0.1:8000/api/music/search/lyrics?q=heartbreak+in+the+rain&engine=spimi&k=10"
```

**REST — búsqueda por audio:**

```bash
curl -X POST "http://127.0.0.1:8000/api/music/search/audio?engine=pgvector&k=10" \
  -F "file=@sample.mp3"
```

### 7.2 App · Recomendación Multimodal (Fashion)

**GUI.**

1. Elegir la modalidad: **Descripción** (texto) o **Imagen** (`.jpg`/`.png`).
2. Elegir el motor con el mismo criterio de música.
3. Cada resultado trae: nombre, categoría, subcategoría, descripción completa, score y miniatura del producto.

**REST — búsqueda por descripción:**

```bash
curl "http://127.0.0.1:8000/api/fashion/search/description?q=black+leather+jacket&engine=gin&k=10"
```

**REST — búsqueda por imagen:**

```bash
curl -X POST "http://127.0.0.1:8000/api/fashion/search/image?engine=spimi&k=10" \
  -F "file=@query.jpg"
```

### 7.3 Evidencias visuales

Las capturas del frontend (home, búsqueda por letras/audio/descripción/imagen y OpenAPI docs) están en el **[Manual de Usuario](docs/MANUAL_USUARIO.md)**, junto con el paso a paso para cada flujo.

---

## Anexos

### A. Estructura de código relevante

```
src/
├── extraction/      # SIFT, MFCC, TF-IDF (extractores por modalidad)
├── ml/              # KClustering (K-Means), VectorQuantizer, TopKWords
├── engine/          # SPIMI (inverted_index.py), similarity.py, *_pipeline.py
├── db/              # SQLModel schemas, native_search.py, storage.py
└── api/             # FastAPI: routes_music, routes_fashion, search_service (dispatcher)

scripts/
├── download_*.py    # Descargas idempotentes (FMA, Spotify, Fashion)
├── etl_music.py     # Codebook + SPIMI + persist para música
├── etl_fashion.py   # Codebook + SPIMI + persist para fashion
├── bench_full.py    # Benchmarks latencia/throughput + helpers RSS/pg_relation_size
├── compute_recall.py
└── setup_all.sh     # Orquesta descarga + Postgres + ETL + backend

frontend/
├── app/             # Next.js App Router (rutas de UI)
├── components/      # SearchApp, EngineSelector, ResultCard, CompareEnginesPanel…
└── lib/api.ts       # Cliente REST tipado

docker/
└── init.sql         # Schema + extensiones + índices GIN/GiST/HNSW
```

### B. Fórmulas de referencia

- **log-TF:** `log_tf(t, d) = 1 + log10(tf(t,d))` si `tf > 0`, si no `0`.
- **IDF:** `idf(t) = log10(N / df(t))`.
- **TF-IDF:** `w(t,d) = log_tf(t,d) · idf(t)`.
- **Coseno:** `cos(q,d) = Σ_t w(t,q)·w(t,d) / (‖q‖·‖d‖)`.
- **Recall@K:** `|resultados_top_K ∩ relevantes| / |relevantes|`.

### C. Planificación y seguimiento

- Hitos y avances en **GitHub Projects** del repositorio (kanban por miembro y por hito).
- Presentación oral: demo viva de App 2 (música) y App 4 (fashion) con comparación de motores en el frontend.

### D. Bibliografía

[1] L. Nanni, Y. Costa, S. Brahnam. *Set of texture descriptors for music genre classification*. arXiv:1312.5457, 2013. <https://arxiv.org/abs/1312.5457>

[2] J. Yang, Y.-G. Jiang, A. G. Hauptmann, C.-W. Ngo. *Evaluating Bag-of-Visual-Words Representations in Scene Classification*. En *Proc. ACM Int. Workshop on Multimedia Information Retrieval (MIR '07)*, 2007. <https://www.researchgate.net/publication/225103338_On_Vocabulary_Size_in_Bag-of-Visual-Words_Representation>

[3] C. D. Manning, P. Raghavan, H. Schütze. *Introduction to Information Retrieval*, Cambridge University Press, 2008 — capítulos 4 (construcción del índice, SPIMI) y 6 (scoring TF-IDF y ponderación por longitud del documento). <https://nlp.stanford.edu/IR-book/>

[4] D. G. Lowe. *Distinctive Image Features from Scale-Invariant Keypoints*. *International Journal of Computer Vision*, 60(2):91–110, 2004 — descriptor SIFT. <https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf>

[5] J. Sivic, A. Zisserman. *Video Google: A Text Retrieval Approach to Object Matching in Videos*. En *Proc. IEEE ICCV*, 2003 — origen del Bag-of-Visual-Words para retrieval. <https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/sivic03.pdf>

[6] B. Logan. *Mel Frequency Cepstral Coefficients for Music Modeling*. En *Proc. ISMIR*, 2000 — justificación de MFCC como descriptor acústico. <https://ismir2000.ismir.net/papers/logan_paper.pdf>

[7] Yu. Malkov, D. Yashunin. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. *IEEE TPAMI*, 42(4):824–836, 2020 — algoritmo HNSW usado por `pgvector`. <https://arxiv.org/abs/1603.09320>

[8] pgvector (A. Kane). Documentación oficial de la extensión de PostgreSQL. <https://github.com/pgvector/pgvector>

[9] PostgreSQL Global Development Group. *PostgreSQL 16 Documentation — Full Text Search* (índices GIN/GiST sobre `tsvector`). <https://www.postgresql.org/docs/16/textsearch-indexes.html>
