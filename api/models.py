"""
models.py
---------
Schemas Pydantic para request/response de la API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# =========================================================
# 🟢 REQUEST SCHEMAS
# =========================================================

class BusquedaRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Consulta en lenguaje natural sobre normativa SENASA.",
        examples=["¿Qué establece el artículo 3?", "definición de producto veterinario"],
    )
    modelo: Literal["tfidf", "embeddings", "hybrid"] = Field(
        default="hybrid",
        description="Modelo de recuperación a utilizar.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de resultados a retornar.",
    )
    solo_articulos: Optional[bool] = Field(
        default=False,
        description="Si True, filtra solo chunks de tipo ARTÍCULO.",
    )
    solo_definiciones: Optional[bool] = Field(
        default=False,
        description="Si True, filtra solo chunks marcados como definición.",
    )


class ComparacionRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Consulta a comparar entre modelos.",
    )
    top_k: int = Field(default=3, ge=1, le=10)


# =========================================================
# 🟢 RESPONSE SCHEMAS
# =========================================================

class ChunkResultado(BaseModel):
    chunk_id: str
    doc_id: str
    score: float = Field(description="Score de relevancia (mayor = más relevante).")
    seccion: str
    article_number: Optional[int] = None
    es_definicion: bool
    es_obligacion: bool
    texto: str
    texto_preview: str = Field(description="Primeros 300 caracteres del texto.")


class BusquedaResponse(BaseModel):
    query: str
    modelo: str
    tipo_query: str = Field(description="Tipo detectado: lexica / semantica / mixta.")
    total_resultados: int
    resultados: List[ChunkResultado]


class ResultadoModelo(BaseModel):
    modelo: str
    resultados: List[ChunkResultado]


class ComparacionResponse(BaseModel):
    query: str
    tipo_query: str
    comparacion: List[ResultadoModelo]


class EstadoResponse(BaseModel):
    status: str
    chunks_indexados: int
    documentos_cargados: int
    modelos_disponibles: List[str]
    pdfs: List[str]


# =========================================================
# 🟢 SCHEMAS PARA ENDPOINT /consultar (QA)
# =========================================================

class ConsultaRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Pregunta en lenguaje natural sobre la normativa SENASA.",
        examples=["¿Qué es la brucelosis bovina?", "¿Cómo se registra un producto veterinario?"],
    )
    modelo: Literal["tfidf", "embeddings", "hybrid"] = Field(
        default="hybrid",
        description="Modelo de recuperación a usar como base.",
    )
    modo_qa: Literal["extractivo", "sintetico", "transformer"] = Field(
        default="extractivo",
        description=(
            "Modo de generación de respuesta. "
            "'extractivo': oración más relevante por overlap léxico. "
            "'sintetico': síntesis heurística de múltiples chunks. "
            "'transformer': span exacto extraído por BERT fine-tuneado en QA español."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de chunks a recuperar antes de generar la respuesta.",
    )
    top_n: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Chunks a usar para síntesis o QA Transformer (top_n ≤ top_k).",
    )


class FuenteQA(BaseModel):
    doc_id: str
    seccion: str
    score: float


class ConsultaResponse(BaseModel):
    query: str
    modelo: str
    modo_qa: str
    tipo_query: str = Field(description="Tipo detectado: definicion / articulo / procedimiento / general.")
    respuesta: str
    score_qa: Optional[float] = Field(
        default=None,
        description="Confianza del Transformer en la respuesta (0-1). Solo en modo transformer."
    )
    fuentes: List[FuenteQA]
    n_chunks_usados: int
    chunks_recuperados: List[ChunkResultado] = Field(
        description="Chunks usados por el retriever (para trazabilidad)."
    )
