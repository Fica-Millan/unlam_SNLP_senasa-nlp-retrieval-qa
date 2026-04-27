"""
main.py
-------
API REST para el sistema de recuperación de normativa SENASA.

Endpoints:
  GET  /              → info de la API
  GET  /estado        → estado del pipeline (chunks, docs, modelos)
  POST /buscar        → búsqueda con un modelo
  POST /comparar      → búsqueda comparativa entre los 3 modelos
  POST /consultar      → consulta en lenguaje natural con respuesta QA
  GET  /documento/{id}→ chunks de un documento específico
  GET  /documentos     → lista de documentos indexados

Ejecutar:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api.models import (
    BusquedaRequest, BusquedaResponse,
    ComparacionRequest, ComparacionResponse,
    EstadoResponse, ChunkResultado, ResultadoModelo,
    ConsultaRequest, ConsultaResponse, FuenteQA,
)
from api.pipeline import get_pipeline
from utils import detectar_tipo_query
from qa_engine import responder as qa_responder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# 🟢 STARTUP
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa el pipeline al arrancar la API."""
    logger.info("Cargando pipeline NLP...")
    get_pipeline()
    logger.info("Pipeline listo. API disponible.")
    yield


# =========================================================
# 🟢 APP
# =========================================================

app = FastAPI(
    title="API de Normativa SENASA",
    description=(
        "Sistema de recuperación y comprensión de resoluciones del "
        "Servicio Nacional de Sanidad y Calidad Agroalimentaria (SENASA). "
        "Permite consultar normativa en lenguaje natural mediante modelos "
        "TF-IDF, Embeddings semánticos y un retriever Híbrido."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 🟢 HELPERS
# =========================================================

def chunk_a_resultado(r: dict) -> ChunkResultado:
    """Convierte un resultado del retriever al schema de respuesta."""
    chunk = r["chunk"]
    texto = chunk.get("texto", "")
    return ChunkResultado(
        chunk_id=chunk.get("chunk_id", ""),
        doc_id=chunk.get("doc_id", ""),
        score=round(r["score"], 4),
        seccion=chunk.get("seccion", ""),
        article_number=chunk.get("article_number"),
        es_definicion=chunk.get("es_definicion", False),
        es_obligacion=chunk.get("es_obligacion", False),
        texto=texto,
        texto_preview=texto[:300] + ("..." if len(texto) > 300 else ""),
    )


def aplicar_filtros(resultados: list, solo_articulos: bool, solo_definiciones: bool) -> list:
    """Aplica filtros opcionales sobre los resultados."""
    if solo_articulos:
        resultados = [r for r in resultados if "ARTÍCULO" in r["chunk"].get("seccion", "")]
    if solo_definiciones:
        resultados = [r for r in resultados if r["chunk"].get("es_definicion")]
    return resultados


# =========================================================
# 🟢 ENDPOINTS
# =========================================================

@app.get("/", tags=["Info"])
def raiz():
    return {
        "nombre": "API de Normativa SENASA",
        "version": "1.0.0",
        "endpoints": ["/estado", "/buscar", "/comparar", "/documento/{id}"],
        "documentacion": "/docs",
    }


@app.get("/estado", response_model=EstadoResponse, tags=["Info"])
def estado():
    """Retorna el estado del pipeline: documentos, chunks y modelos cargados."""
    p = get_pipeline()
    return EstadoResponse(
        status="ok",
        chunks_indexados=len(p.chunks),
        documentos_cargados=len(p.docs),
        modelos_disponibles=["tfidf", "embeddings", "hybrid"],
        pdfs=[doc["id"] for doc in p.docs],
    )


@app.post("/buscar", response_model=BusquedaResponse, tags=["Búsqueda"])
def buscar(req: BusquedaRequest):
    """
    Realiza una búsqueda sobre la normativa SENASA.

    - **query**: Pregunta o frase en lenguaje natural.
    - **modelo**: `tfidf` | `embeddings` | `hybrid` (default: hybrid).
    - **top_k**: Número de resultados (1-20, default: 5).
    - **solo_articulos**: Filtra solo chunks de artículos.
    - **solo_definiciones**: Filtra solo chunks con definiciones.
    """
    p = get_pipeline()

    try:
        retriever = p.get_retriever(req.modelo)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Buscar con margen para filtros
    fetch_k = req.top_k * 3 if (req.solo_articulos or req.solo_definiciones) else req.top_k
    resultados_raw = retriever.buscar(req.query, top_k=fetch_k)

    resultados_raw = aplicar_filtros(
        resultados_raw,
        req.solo_articulos,
        req.solo_definiciones,
    )[:req.top_k]

    tipo = detectar_tipo_query(req.query)

    return BusquedaResponse(
        query=req.query,
        modelo=req.modelo,
        tipo_query=tipo,
        total_resultados=len(resultados_raw),
        resultados=[chunk_a_resultado(r) for r in resultados_raw],
    )


@app.post("/comparar", response_model=ComparacionResponse, tags=["Búsqueda"])
def comparar(req: ComparacionRequest):
    """
    Ejecuta la misma query en los 3 modelos y retorna los resultados
    lado a lado para comparación.
    """
    p = get_pipeline()
    tipo = detectar_tipo_query(req.query)

    comparacion = []
    for nombre in ["tfidf", "embeddings", "hybrid"]:
        retriever = p.get_retriever(nombre)
        resultados_raw = retriever.buscar(req.query, top_k=req.top_k)
        comparacion.append(ResultadoModelo(
            modelo=nombre,
            resultados=[chunk_a_resultado(r) for r in resultados_raw],
        ))

    return ComparacionResponse(
        query=req.query,
        tipo_query=tipo,
        comparacion=comparacion,
    )


@app.post("/consultar", response_model=ConsultaResponse, tags=["QA"])
def consultar(req: ConsultaRequest):
    """
    Consulta en lenguaje natural sobre la normativa SENASA.

    Pipeline completo:
        1. Recupera chunks relevantes con el modelo elegido.
        2. Genera una respuesta con el QA engine seleccionado.

    Modos de respuesta:
    - **extractivo**: oración más relevante por overlap léxico (rápido).
    - **sintetico**: síntesis heurística de múltiples chunks (sin modelo neuronal).
    - **transformer**: span exacto extraído por BERT fine-tuneado en QA español
      (`mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es`).
      Incluye `score_qa` (confianza 0-1) en la respuesta.

    La respuesta incluye las fuentes exactas (doc_id + sección) para
    trazabilidad completa hacia la normativa original.
    """
    p = get_pipeline()

    try:
        retriever = p.get_retriever(req.modelo)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 1. Recuperar chunks
    resultados_raw = retriever.buscar(req.query, top_k=req.top_k)

    if not resultados_raw:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron fragmentos relevantes para la consulta.",
        )

    # 2. Generar respuesta con el modo QA elegido
    if req.modo_qa == "transformer":
        qa_t = p.get_qa_transformer()
        qa_resultado = qa_responder(
            resultados_raw,
            req.query,
            modo="transformer",
            top_n=req.top_n,
            qa_transformer_instance=qa_t,
        )
    else:
        qa_resultado = qa_responder(
            resultados_raw,
            req.query,
            modo=req.modo_qa,
            top_n=req.top_n,
        )

    # 3. Armar fuentes como objetos tipados
    fuentes_raw = qa_resultado.get("fuentes", [])
    if not fuentes_raw and "fuente" in qa_resultado:
        fuentes_raw = [{
            "doc_id":  qa_resultado.get("fuente", ""),
            "seccion": qa_resultado.get("seccion", ""),
            "score":   qa_resultado.get("score_retrieval",
                       qa_resultado.get("score", 0.0)),
        }]

    fuentes = [
        FuenteQA(
            doc_id=f.get("doc_id", ""),
            seccion=f.get("seccion", ""),
            score=f.get("score", 0.0),
        )
        for f in fuentes_raw
    ]

    return ConsultaResponse(
        query=req.query,
        modelo=req.modelo,
        modo_qa=req.modo_qa,
        tipo_query=qa_resultado.get("tipo_query", detectar_tipo_query(req.query)),
        respuesta=qa_resultado["respuesta"],
        score_qa=qa_resultado.get("score_qa"),
        fuentes=fuentes,
        n_chunks_usados=qa_resultado.get("n_chunks_usados", len(fuentes)),
        chunks_recuperados=[chunk_a_resultado(r) for r in resultados_raw],
    )


@app.get("/documento/{doc_id}", tags=["Documentos"])
def chunks_de_documento(
    doc_id: str,
    seccion: str = Query(default=None, description="Filtrar por sección (ej: ARTÍCULO 3)"),
):
    """
    Retorna todos los chunks de un documento específico.
    Opcionalmente filtra por sección.
    """
    p = get_pipeline()

    chunks = [c for c in p.chunks if c["doc_id"] == doc_id]

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"Documento '{doc_id}' no encontrado. "
                   f"Disponibles: {[d['id'] for d in p.docs]}",
        )

    if seccion:
        seccion_upper = seccion.upper()
        chunks = [c for c in chunks if seccion_upper in c.get("seccion", "").upper()]

    return {
        "doc_id": doc_id,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "seccion": c["seccion"],
                "article_number": c.get("article_number"),
                "es_definicion": c.get("es_definicion"),
                "es_obligacion": c.get("es_obligacion"),
                "texto_preview": c["texto"][:200] + "..." if len(c["texto"]) > 200 else c["texto"],
            }
            for c in chunks
        ],
    }


@app.get("/documentos", tags=["Documentos"])
def listar_documentos():
    """Lista todos los documentos indexados con su metadata."""
    p = get_pipeline()
    return {
        "total": len(p.docs),
        "documentos": [
            {
                "id": d["id"],
                "numero_resolucion": d.get("numero_resolucion"),
                "anio": d.get("anio"),
                "titulo": d.get("titulo", "")[:120],
                "chunks": sum(1 for c in p.chunks if c["doc_id"] == d["id"]),
            }
            for d in p.docs
        ],
    }
