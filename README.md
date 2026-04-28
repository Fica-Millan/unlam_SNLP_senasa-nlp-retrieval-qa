# Sistema de Recuperación y Comprensión de Normativa SENASA

Trabajo Final Integrador - Seminario de Procesamiento del Lenguaje Natural  
Universidad Nacional de La Matanza · 2026


<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green?logo=fastapi)
![NLP](https://img.shields.io/badge/NLP-Retrieval%20%26%20QA-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue)

![Status](https://img.shields.io/badge/Status-Terminado-success)
![Author](https://img.shields.io/badge/Autor-Yesica%20Fica%20Mill%C3%A1n-purple)
![Author](https://img.shields.io/badge/Autor-Franco%20Petraroia-brown)

</div>


## Descripción del proyecto

Sistema híbrido de recuperación y comprensión de normativa del **Servicio Nacional de Sanidad y Calidad Agroalimentaria (SENASA)**. Dado un corpus de resoluciones en PDF, el sistema permite consultar la normativa en lenguaje natural y obtener respuestas con trazabilidad hacia el documento fuente.

El pipeline cubre el ciclo completo de un proyecto NLP aplicado:

```
PDFs  →  extracción y limpieza  →  chunking estructural
      →  indexación (TF-IDF / Embeddings / Híbrido)
      →  recuperación (IR)
      →  respuesta en lenguaje natural (QA)
      →  API REST consultable
```

### Enfoques implementados

| Componente | Modelo clásico | Modelo profundo |
|---|---|---|
| Retrieval | TF-IDF + similitud coseno | Embeddings semánticos (SentenceTransformer) + FAISS |
| Recuperación combinada | — | Híbrido TF-IDF + Embeddings (interpolación lineal, α=0.8) |
| QA extractivo | Overlap léxico heurístico | BERT fine-tuneado en SQuAD español (`mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es`) |
| RAG | — | DistilBERT + FAISS (`mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es`) |

---

## Estructura del proyecto

```
TFI_comprension_normativa/
│
├── data/
│   ├── raw/pdfs/               # Resoluciones SENASA (corpus, 10 documentos)
│   └── eval_queries.json       # Gold standard de evaluación (15 queries verificadas)
│
├── docs/
│   ├── informe_final.pdf       # Informe académico completo del proyecto
│   └── presentacion.pdf        # Presentación utilizada en la defensa
│
├── src/
│   ├── pdf_loader.py           # Extracción y limpieza de texto desde PDF
│   ├── chunker.py              # División en chunks por secciones normativas
│   ├── utils.py                # Funciones compartidas (normalización, clasificación)
│   ├── retriever_tfidf.py      # Modelo clásico: TF-IDF + coseno
│   ├── retriever_embeddings.py # Modelo profundo: SentenceTransformer + FAISS
│   ├── retriever_hybrid.py     # Híbrido: interpolación lineal + re-ranking
│   ├── mock_embeddings.py      # Mock offline (LSA/SVD) para tests sin HuggingFace
│   ├── qa_engine.py            # QA extractivo y sintético (sin modelo neuronal)
│   ├── qa_transformer.py       # QA con BERT fine-tuneado en SQuAD español
│   ├── rag_distilbert.py       # Pipeline RAG completo con DistilBERT
│   ├── evaluation.py           # Métricas de retrieval (P@k, R@k, HR@k, MRR, F1)
│   ├── visualization.py        # Generación de gráficos comparativos
│   └── run_evaluation.py       # Script de evaluación completo y reproducible
│
├── api/
│   ├── main.py                 # Endpoints FastAPI
│   ├── models.py               # Schemas Pydantic (request/response)
│   └── pipeline.py             # Singleton: carga y mantiene modelos en memoria
│
├── outputs/
│   └── figures/                # Gráficos generados (ignorados en git)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación y setup

### 1. Clonar el repositorio y crear entorno virtual

```bash
git clone https://github.com/Fica-Millan/unlam_SNLP_senasa-nlp-retrieval-qa.git
cd TFI_comprension_normativa
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar modelos de HuggingFace (una sola vez)

El sistema funciona en modo offline con mocks (LSA/SVD para embeddings, overlap léxico para QA). Para usar los modelos neuronales reales:

```bash
# Embeddings semánticos
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# BERT QA español
python -c "from transformers import pipeline; pipeline('question-answering', model='mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es')"

# DistilBERT QA español (RAG)
python -c "from transformers import pipeline; pipeline('question-answering', model='mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es')"
```

---

## Cómo correr la evaluación

El script `run_evaluation.py` ejecuta la evaluación completa de los tres modelos de retrieval y los tres modos QA sobre el gold standard, y guarda los resultados en `outputs/`.

```bash
# Modo mock (default) — sin HuggingFace, totalmente reproducible
python src/run_evaluation.py

# Modo real — SentenceTransformer + BERT (requiere modelos descargados)
python src/run_evaluation.py --modo real
```

El modo `mock` usa `MockEmbeddingRetriever` (LSA/SVD) y `MockQATransformer` (overlap léxico) para garantizar reproducibilidad sin acceso a HuggingFace. Si el modo `real` falla al cargar algún modelo, cae automáticamente al mock con un aviso en consola.

**Salidas generadas:**

| Archivo | Contenido |
|---|---|
| `outputs/eval_retrieval.json` | Métricas de retrieval por modelo y tipo de query |
| `outputs/eval_qa.json` | Métricas QA (EM + F1 tokens) por modelo y modo |
| `outputs/eval_resumen.json` | Tabla consolidada para el informe |
| `outputs/eval_rag_distilbert.json` | Métricas del RAG DistilBERT (solo `--modo real`) |

> Los resultados de la sección **Resultados** corresponden al modo `--modo real` con los modelos neuronales completos, sobre el gold standard unificado de 30 queries.

---

## Cómo levantar la API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

La documentación interactiva queda disponible en `http://localhost:8000/docs`.

### Endpoints principales

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/estado` | Estado del pipeline: documentos, chunks y modelos cargados |
| `POST` | `/buscar` | Búsqueda con un modelo (`tfidf` / `embeddings` / `hybrid`) |
| `POST` | `/comparar` | Misma query en los 3 modelos, resultados lado a lado |
| `POST` | `/consultar` | QA completo: retrieval + respuesta en lenguaje natural |
| `GET` | `/documento/{id}` | Chunks de un documento específico |
| `GET` | `/documentos` | Lista todos los documentos indexados |

### Ejemplo de consulta

```bash
curl -X POST http://localhost:8000/consultar \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué es la brucelosis bovina?",
    "modelo": "hybrid",
    "modo_qa": "transformer",
    "top_k": 5
  }'
```

---

## Resultados

Evaluación sobre 30 queries del gold standard (9 léxicas, 13 semánticas, 8 mixtas), k=5.
Resultados obtenidos con `--modo real` (SentenceTransformer + BERT español).

### Retrieval

| Modelo | P@5 | R@5 | HR@5 | MRR | F1@5 | MRR-léxico | MRR-semántico | MRR-mixto |
|---|---|---|---|---|---|---|---|---|
| TF-IDF | 0.187 | 0.558 | 0.733 | 0.484 | 0.260 | 0.633 | 0.359 | 0.521 |
| Embeddings | 0.147 | 0.423 | 0.533 | 0.397 | 0.202 | 0.544 | 0.308 | 0.375 |
| **Híbrido** | 0.207 | 0.593 | **0.800** | **0.561** | **0.284** | 0.633 | **0.471** | **0.625** |

### QA - F1 de tokens (estilo SQuAD)

| Modelo | F1-extractivo | F1-sintético | F1-transformer |
|---|---|---|---|
| TF-IDF | 0.242 | 0.237 | 0.093 |
| Embeddings | 0.247 | 0.201 | 0.084 |
| **Híbrido** | **0.251** | 0.233 | 0.079 |

### RAG DistilBERT (contexto concatenado)

| Modelo | Retriever | F1 |
|---|---|---|
| RAG DistilBERT | Embeddings (SentenceTransformer + FAISS) | 0.190 |

El RAG DistilBERT concatena los top-5 chunks en un único contexto (truncado a 1800 chars) y aplica una sola inferencia de DistilBERT, a diferencia del modo transformer que evalúa cada chunk por separado. El F1 más bajo refleja la dificultad de extraer el span correcto de un contexto mixto y truncado. Se genera en `outputs/eval_rag_distilbert.json` solo con `--modo real`.

> **Nota sobre Exact Match (EM):** El EM es 0.0 en todos los casos, lo cual es esperable. Las respuestas son frases largas extraídas de texto normativo; el EM exige igualdad carácter a carácter con el gold, condición que prácticamente nunca se cumple en QA sobre texto libre. La métrica relevante es F1 de tokens.

> **Nota sobre F1-transformer:** El F1 del modo transformer es menor que el de los modos heurísticos porque BERT extrae spans cortos y exactos del texto, que comparten menos tokens con las respuestas gold (redactadas como frases completas). Esto no refleja peor calidad, sino una diferencia en el estilo de respuesta. Con un gold standard de spans exactos el resultado sería superior.

---

## Decisiones de diseño

**Chunking estructural sobre chunking fijo**  
Las resoluciones SENASA tienen estructura normativa explícita (VISTO / CONSIDERANDO / ARTÍCULO N). Dividir por estas secciones preserva la coherencia semántica de cada chunk mejor que ventanas de tamaño fijo. El CONSIDERANDO se subdivide además por cláusulas "Que" para granularidad fina.

**Regla estructural de artículo (bypass de retrieval)**  
Cuando la query menciona un artículo específico ("artículo 3", "art. 5"), todos los retrievers retornan directamente los chunks de ese artículo con score=1.0, sin pasar por el ranking vectorial. Esto garantiza precisión perfecta en queries estructurales, que son frecuentes en consultas normativas.

**Alpha=0.8 en el híbrido**  
El peso del componente semántico se optimizó mediante grid search sobre MRR. El valor 0.8 favorece embeddings sobre TF-IDF porque las consultas de usuarios tienden a ser paráfrasis, no términos exactos. Con embeddings reales (SentenceTransformer), este sesgo hacia semántica tiene más impacto que con el mock LSA/SVD.

**Mock offline para reproducibilidad**  
`MockEmbeddingRetriever` (LSA/SVD con sklearn) y `MockQATransformer` (overlap léxico) permiten correr tests y la evaluación completa sin descargar modelos. El contrato de interfaz es idéntico al de los modelos reales, por lo que el reemplazo es transparente.

**QA extractivo como baseline, Transformer como techo**  
El modo extractivo (overlap léxico) establece el baseline de QA sin ningún modelo neuronal. El modo sintético combina heurísticas sobre múltiples chunks. El modo transformer (BERT fine-tuneado en SQuAD español) establece el techo del sistema con la misma arquitectura de retrieval. La comparación de los tres modos permite aislar la contribución del componente neuronal.

**Lazy loading del QATransformer en la API**  
El modelo BERT se carga en el primer uso del endpoint `/consultar` con `modo_qa=transformer`, no al arrancar la API. Esto reduce el tiempo de startup de ~30s a <2s en entornos sin GPU, sin afectar la disponibilidad de los otros endpoints.

## Documentación

- Informe completo: `docs/informe_final.pdf`
- Presentación: `docs/presentacion.pdf`
