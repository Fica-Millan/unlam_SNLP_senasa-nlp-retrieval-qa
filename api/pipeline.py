"""
pipeline.py
-----------
Singleton que carga y mantiene los modelos en memoria.

Al arrancar la API se ejecuta una sola vez:
  1. Carga los PDFs
  2. Genera los chunks
  3. Entrena TF-IDF
  4. Entrena Embeddings (real o Mock según disponibilidad)
  5. Construye el Híbrido
  6. Carga el QATransformer (BERT QA español, real o Mock)

Expone get_pipeline() para que los endpoints lo consuman.
"""

import sys
import os
import logging

# Añadir src al path
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

from pdf_loader import cargar_pdfs
from chunker import chunkear_documentos
from retriever_tfidf import TFIDFRetriever

logger = logging.getLogger(__name__)

# Ruta a los PDFs (relativa al directorio del proyecto)
PDFS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "pdfs")


class Pipeline:
    """Contenedor singleton de modelos y datos."""

    def __init__(self):
        self.docs = []
        self.chunks = []
        self.tfidf = None
        self.embedder = None
        self.hybrid = None
        self.qa_transformer = None
        self._ready = False

    def inicializar(self):
        """Carga datos y entrena todos los modelos."""
        logger.info("Iniciando pipeline NLP...")

        # 1. Cargar PDFs
        self.docs = cargar_pdfs(PDFS_DIR)
        logger.info(f"{len(self.docs)} documentos cargados.")

        # 2. Chunkear
        self.chunks = chunkear_documentos(self.docs)
        logger.info(f"{len(self.chunks)} chunks generados.")

        # 3. TF-IDF
        self.tfidf = TFIDFRetriever()
        self.tfidf.fit(self.chunks)
        logger.info("TF-IDF listo.")

        # 4. Embeddings (con fallback offline)
        self.embedder = self._cargar_embedder()

        # 5. Híbrido
        self.hybrid = self._construir_hybrid(alpha=0.8)
        logger.info("Pipeline listo.")

        # 6. QA Transformer (carga lazy: se inicializa en primer uso)
        # Se carga aquí como None y se inicializa en get_qa_transformer()
        # para no bloquear el arranque si el modelo no está disponible.
        self.qa_transformer = None

        self._ready = True

    def _cargar_embedder(self):
        try:
            from retriever_embeddings import EmbeddingRetriever
            r = EmbeddingRetriever()
            r.fit(self.chunks)
            logger.info("EmbeddingRetriever (SentenceTransformer) listo.")
            return r
        except Exception as e:
            logger.warning(
                f"SentenceTransformer no disponible ({e}). "
                "Usando MockEmbeddingRetriever (LSA/SVD)."
            )
            from mock_embeddings import MockEmbeddingRetriever
            r = MockEmbeddingRetriever()
            r.fit(self.chunks)
            return r

    def _construir_hybrid(self, alpha: float):
        """Construye el híbrido reutilizando los sub-modelos ya entrenados."""
        from retriever_hybrid import build_hybrid
        return build_hybrid(self.chunks, tfidf=self.tfidf, embedder=self.embedder, alpha=alpha)

    def get_retriever(self, nombre: str):
        if nombre == "tfidf":
            return self.tfidf
        elif nombre == "embeddings":
            return self.embedder
        elif nombre == "hybrid":
            return self.hybrid
        raise ValueError(f"Modelo desconocido: '{nombre}'")

    def get_qa_transformer(self):
        """
        Retorna el QATransformer, cargándolo en el primer uso (lazy loading).
        Usa fallback a Mock si el modelo no está disponible.
        """
        if self.qa_transformer is None:
            from qa_transformer import get_qa_transformer
            self.qa_transformer = get_qa_transformer()
            logger.info(f"QATransformer listo: {type(self.qa_transformer).__name__}")
        return self.qa_transformer

    @property
    def ready(self):
        return self._ready


# --- Singleton global ---
_pipeline: Pipeline = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None or not _pipeline.ready:
        _pipeline = Pipeline()
        _pipeline.inicializar()
    return _pipeline
