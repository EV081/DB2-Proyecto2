### Estrategia de GitHub Projects: Hitos y Desarrollo Paralelo (Actualizada)

#### Hito 1: Infraestructura y Mocks (Semana 1)

**Objetivo:** Todos levantan su entorno y programan entradas/salidas con datos inventados. Nadie depende de nadie.

| Miembro | Rol y Rama en Git (`feature/...`) | Tareas (Issues en GitHub) |
| --- | --- | --- |
| **Juan Carlos** | `feature/signal-extraction` | Escribir funciones que lean una imagen/audio local y retornen una matriz de ceros o números aleatorios (simulando SIFT/MFCC). |
| **Paulo** | `feature/ml-codebook` | Crear un script que reciba una matriz aleatoria y le aplique K-Means usando `scikit-learn`. |
| **Elmer** | `feature/custom-index` | Crear diccionarios quemados en Python (hardcoded) y programar la matemática de la Distancia Coseno o Jaccard sobre ellos. |
| **Josue** | `feature/db-api-docker` | Levantar el `docker-compose.yml` con PostgreSQL y `pgvector`. Crear rutas FastAPI que retornen JSONs estáticos de prueba. |
| **Joseph** | `feature/etl-ui` | Crear el esqueleto del script ETL que lea los paths de los datasets en Kaggle e imprima en consola (sin procesar aún). |

#### Hito 2: Lógica Core y Algoritmos (Semanas 2-3)

**Objetivo:** Reemplazar los datos inventados con la matemática real. Aquí es donde se concentra el volumen más pesado de *commits*.

| Miembro | Tareas de Programación Real (Issues) | Independencia |
| --- | --- | --- |
| **Juan Carlos** | Implementar `librosa` para ventanas de audio y TF-IDF para letras. | Prueba sus scripts con un solo audio `.wav` y un `.txt` local. |
| **Paulo** | Entrenar K-Means con lotes de matrices generadas y programar la función de cuantización. | Usa datasets de matrices sintéticas para validar que K-Means converge. |
| **Elmer** | Programar el Índice Invertido (SPIMI) y la lógica de ranking para devolver el Top-K. | Usa un archivo `.json` masivo generado con datos basura para medir sus tiempos de búsqueda y optimizar la recuperación en memoria. |
| **Josue** | Definir los modelos con SQLModel y programar las consultas GIN/GiST nativas en la base de datos. | Hace inserciones directas a la BD usando scripts sueltos en Python conectándose al contenedor. |
| **Joseph** | Automatizar el *Pipeline Maestro* (que junte la lectura masiva) y crear los scripts de la Fase 4 para disparar miles de consultas simuladas. | Programa wrappers de temporizadores (latencia) y medidores de RAM. |

#### Hito 3: Integración, Evaluación y Cierre (Semanas 4-5)

**Objetivo:** Consolidar el sistema multimodal, desplegar la interfaz de usuario y documentar los resultados experimentales.

* **Frente de Integración Core Backend (Juan Carlos y Elmer):**
* Consolidar las ramas `feature/signal-extraction` y `feature/custom-index` hacia la rama principal (`develop`/`main`).
* Resolver conflictos de Git y asegurar la compatibilidad de las estructuras de datos (matrices, vectores e histogramas) entre la extracción de datos y el índice invertido en memoria.
* Garantizar la fluidez del pipeline de búsqueda offline y online antes de la conexión externa.

* **Frente de UI, Benchmarking e Informe (Josue, Paulo y Joseph):**
* **Conexión con el Frontend:** Conectar las rutas de FastAPI con la interfaz visual en Streamlit o Gradio para permitir búsquedas interactivas (subir un audio/imagen/texto y mostrar el Top-K de resultados).
* **Pruebas de Rendimiento (Benchmarking):** Ejecutar los scripts de evaluación masiva (`run_experiments.py`) para medir la latencia de búsqueda, consumo de memoria RAM y la métrica de *Recall*.
* **Informe Técnico Final:** Redactar las secciones de arquitectura, metodología y analizar los gráficos de *trade-offs* (motor propio vs. `pgvector`) para el entregable final.

Resumen detallado de la distribución de roles. Este formato les servirá como una excelente guía o "contrato de equipo" para que todos tengan claro su alcance desde el primer día:

### 1. Juan Carlos: Ingeniero de Procesamiento de Señales (Capa de Extracción)

* **Reto de código:** Matemáticas y manipulación de matrices.
* **Responsabilidad:** Programar los algoritmos que toman un archivo crudo (foto, canción, documento) y lo convierten en números.
* **Tareas de programación:**
* Código para dividir los archivos en *chunks* (ventanas de tiempo para audio, parches superpuestos para imágenes, separación de párrafos para texto).
* Código para implementar los extractores matemáticos: TF-IDF (texto), MFCC usando Librosa (audio) y SIFT/Inception usando OpenCV/PyTorch (imagen).

* **Cómo trabaja en paralelo:** Desde el día 1 puede programar probando con archivos sueltos en su computadora, sin necesidad de que exista la base de datos o la API.

### 2. Paulo: Ingeniero de Machine Learning (Capa de Codebook y Cuantización)

* **Reto de código:** Modelos de agrupamiento estadístico.
* **Responsabilidad:** Tomar los millones de vectores que extrae Juan Carlos y programar la lógica para crear los "diccionarios" y convertirlos en histogramas.
* **Tareas de programación:**
* Código que implementa y entrena el algoritmo K-Means a gran escala para encontrar los centroides (las "palabras" acústicas y visuales).
* Código del seleccionador lingüístico (Top-k palabras para el texto).
* Código del "Cuantizador": una función que reciba un vector nuevo y calcule a qué centroide del diccionario pertenece para armar el histograma final de frecuencias.

* **Cómo trabaja en paralelo:** Genera matrices de números aleatorios (datos falsos) en Python que simulen los vectores de extracción y programa el K-Means sobre eso.

Especificación para Paulo:
Dile a Paulo que tiene razón a medias. Es verdad que el documento no usa la palabra literal "entrenar" como si fueran a crear una red neuronal compleja o una IA generativa, pero el concepto técnico y estadístico sí está explícitamente en el proyecto.

Cuando me refiero a "entrenar", hablo exactamente de lo que el documento llama **"Agrupación"** o **"Clustering"** para construir el Codebook. Dile que su trabajo es aplicar **Aprendizaje No Supervisado (Unsupervised Machine Learning)**.

Aquí te desgloso cada punto para que se lo pases y le quede clarísimo de dónde sale cada tarea de su rol:

**1. El K-Means a gran escala (El "Entrenamiento")**
El documento especifica claramente que el Módulo Codebook requiere el uso de "K-Means/K-Medoids para imágenes y audio". En la teoría de Machine Learning, cuando ejecutas el algoritmo K-Means para que encuentre patrones en miles de vectores y defina los centroides óptimos, a ese proceso computacional se le conoce formalmente como "ajustar" (fit) o "entrenar" el modelo .

* Para imágenes, el documento pide usar "K-Means sobre todos los SIFT de todas las imágenes" para que esos centroides representen las "palabras visuales".
* Para audio, pide usar "K-Means sobre todos los MFCC de todos los audios".

**2. El Seleccionador Lingüístico (Para el texto)**
Para la modalidad de texto, el documento indica que no se usa K-Means, sino que se aplican "técnicas lingüísticas". La tarea de Paulo aquí es escribir el código que tome todas las palabras extraídas por Juan Carlos y seleccione "las k palabras más frecuentes en toda la colección". Este top de palabras conformará el diccionario textual.

**3. El "Cuantizador" (La Cuantización)**
Esto también está explícito en la sección de consideraciones técnicas. El documento exige aplicar "Cuantización (imagen/audio): reduce descriptores altos a k codewords (compresión lossy)". Una vez que Paulo encuentre los centroides con el K-Means, su código debe recibir cada vector nuevo de las canciones o imágenes, calcular matemáticamente a qué centroide se parece más (cuantizarlo) y armar con eso el "histograma de codewords".

En resumen: Paulo no va a entrenar a ChatGPT, pero sí tiene que programar y ejecutar un modelo estadístico (K-Means) sobre miles de matrices de datos para encontrar los clústeres. ¡Eso es 100% trabajo de Machine Learning!

### 3. Elmer: Ingeniero de Algoritmos (Capa del Motor Customizado)

* **Reto de código:** Estructuras de datos complejas y optimización de memoria.
* **Responsabilidad:** Programar desde cero el motor de búsqueda manual que exige el proyecto, sin usar las ayudas de PostgreSQL.
* **Tareas de programación:**
* Código para construir el Índice Invertido (SPIMI) en memoria/disco (mapeando cada ID de palabra/centroide a los IDs de los *chunks*).
* Código del motor de similitud: implementar matemáticamente el cálculo de la Distancia Coseno o Similitud de Jaccard para comparar el histograma de una consulta contra los almacenados.
* Código de *ranking* para ordenar los miles de resultados de mayor a menor coincidencia de forma ultra-rápida.

* **Cómo trabaja en paralelo:** Crea diccionarios y listas de Python inventadas (histogramas falsos) y programa los algoritmos de búsqueda y ordenamiento directamente ejecutándolos desde la terminal, aislando completamente su lógica de la base de datos.

### 4. Josue: Arquitecto de Base de Datos y API (Capa de Postgres y FastAPI)

* **Reto de código:** Integración backend y bases de datos vectoriales.
* **Responsabilidad:** Construir la infraestructura que sostiene todo y programar la comparativa nativa de PostgreSQL.
* **Tareas de programación:**
* Código de los modelos ORM (SQLModel/SQLAlchemy) para crear las tablas en la base de datos.
* Código de las consultas nativas: implementar la inserción y búsqueda usando `pgvector` (índices HNSW) y GIN/GiST.
* Código de las rutas de FastAPI (los endpoints que recibirán los archivos de los usuarios y devolverán respuestas JSON).

* **Cómo trabaja en paralelo:** Levanta el entorno con Docker Compose y programa las rutas de FastAPI devolviendo respuestas simuladas (mocks) hasta que los demás terminen sus módulos.

### 5. Joseph: Ingeniero de Integración, UI y Benchmarking (Capa de Evaluación)

* **Reto de código:** Automatización, Testing Masivo y Frontend.
* **Responsabilidad:** Unir todas las piezas, programar el entorno para la Fase 4 (los experimentos) y armar la interfaz visual.
* **Tareas de programación:**
* Código del "Pipeline Maestro" (ETL): el script pesado que lee los datasets completos de Kaggle, llama a los módulos de los demás y llena la base de datos de forma automática (indexación offline).
* Código del Evaluador: scripts que disparen miles de consultas falsas por segundo a la API para medir y registrar programáticamente la latencia, consumo de RAM y calcular la métrica de Recall.
* Código de la interfaz de usuario en Streamlit o Gradio para la demostración en vivo.

* **Cómo trabaja en paralelo:** Puede ir diseñando la interfaz visual y programar los scripts de medición de tiempo y memoria envolviendo funciones vacías, listas para conectarse al código real en las semanas finales.

## Infraestructura

```text
proyecto2_multimodal/
├── datasets/                  # (Ignorado en .gitignore) Aquí van los audios, imágenes y textos crudos.
├── docker/                    # [Dueño: Josue]
│   └── init.sql               # Script para crear la DB, activar pgvector y crear tablas base.
├── notebooks/                 # [Para todos] Pruebas aisladas en Jupyter (ej. probar librosa, k-means, etc.).
├── scripts/                   # [Dueño: Joseph]
│   ├── etl_pipeline.py        # Pipeline Maestro: Lee Kaggle, llama a extracción/ml y guarda en BD.
│   └── run_experiments.py     # Fase 4: Dispara consultas masivas, mide latencia, RAM y recall.
├── src/                       # Código fuente principal
│   ├── api/                   # [Dueño: Josue]
│   │   ├── routes_music.py    # Endpoints de FastAPI para buscar canciones.
│   │   └── routes_store.py    # Endpoints de FastAPI para buscar productos.
│   ├── db/                    # [Dueño: Josue]
│   │   ├── database.py        # Conexión a PostgreSQL (SQLModel/SQLAlchemy).
│   │   ├── models.py          # Definición de tablas y esquemas vectoriales.
│   │   └── native_search.py   # Consultas GIN/GiST y pgvector.
│   ├── extraction/            # [Dueño: Juan Carlos] Capa 1
│   │   ├── audio_mfcc.py      # Lógica de librosa y ventanas deslizantes.
│   │   ├── image_sift.py      # Lógica de OpenCV/Inception para patches.
│   │   └── text_tfidf.py      # Lógica de tokenización, stopwords y TF-IDF.
│   ├── ml/                    # [Dueño: Paulo] Capa 2
│   │   ├── kmeans_trainer.py  # Entrenamiento masivo de centroides.
│   │   ├── text_topk.py       # Selección lingüística del codebook textual.
│   │   └── quantizer.py       # Asignación de vectores a centroides (creación de histogramas).
│   ├── engine/                # [Dueño: Elmer] Capa 3
│   │   ├── inverted_index.py  # Lógica del SPIMI (mapeo diccionario -> chunks).
│   │   └── similarity.py      # Matemáticas de Distancia Coseno/Jaccard y Top-K ranking.
│   └── main.py                # [Dueño: Josue] Punto de entrada que levanta FastAPI.
├── ui/                        # [Dueño: Joseph] 
│   └── app.py                 # Interfaz visual en Streamlit/Gradio para la demo final.
├── docker-compose.yml         # [Dueño: Josue] Levanta Postgres, pgvector y el backend.
├── requirements.txt           # Dependencias compartidas (fastapi, librosa, scikit-learn, etc.).
└── README.md                  # El Informe Técnico (arquitectura, resultados, trade-offs).

```
