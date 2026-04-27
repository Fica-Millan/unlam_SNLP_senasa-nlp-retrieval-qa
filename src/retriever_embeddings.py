"""
retriever_embeddings.py
-----------------------
Modelo de recuperación semántica basado en embeddings + índice FAISS.

Modelo de embeddings:
    paraphrase-multilingual-MiniLM-L12-v2
    - Multilingüe (incluye español)
    - 384 dimensiones
    - Adecuado para recuperación de párrafos cortos/medianos

Índice vectorial:
    FAISS IndexFlatIP (Inner Product sobre vectores normalizados)
    - Equivalente a similitud coseno cuando los vectores están normalizados
    - Búsqueda exacta (no aproximada) — adecuado para corpus pequeños (~500 chunks)
    - Más eficiente que sklearn.cosine_similarity para corpus grandes

Decisiones de diseño:
    - normalize_embeddings=True → producto punto ≡ coseno
    - faiss.IndexFlatIP → búsqueda exacta, sin pérdida de calidad
    - Regla directa para artículos (bypass FAISS, igual que TF-IDF)
"""

import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer

from utils import normalizar_query, extraer_numero_articulo


class EmbeddingRetriever:
    """
    Recuperador semántico con índice FAISS.

    Pipeline:
        chunks → SentenceTransformer → embeddings normalizados
             → faiss.IndexFlatIP → búsqueda por producto punto (≡ coseno)

    Uso:
        retriever = EmbeddingRetriever()
        retriever.fit(chunks)
        resultados = retriever.buscar("que es brucelosis bovina", top_k=5)
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model      = SentenceTransformer(model_name)
        self.embeddings = None
        self.index      = None   # índice FAISS
        self.chunks     = None
        self._dim       = None

    def fit(self, chunks: List[Dict]):
        """
        Genera embeddings y construye el índice FAISS.

        IndexFlatIP: índice plano con producto interno (Inner Product).
        Con vectores L2-normalizados, IP ≡ similitud coseno.
        """
        self.chunks = chunks
        textos = [c["texto"] for c in chunks]

        # 1. Generar embeddings normalizados
        self.embeddings = self.model.encode(
            textos,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # ||v|| = 1 → IP ≡ coseno
        ).astype("float32")

        # 2. Construir índice FAISS
        self._dim  = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self._dim)  # Inner Product
        self.index.add(self.embeddings)

        print(f"[FAISS] Índice construido: {self.index.ntotal} vectores, dim={self._dim}")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Codifica una query como vector normalizado.
        Usado internamente por HybridRetriever.
        """
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def buscar(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Devuelve los top_k chunks más relevantes usando FAISS.

        Lógica:
        1. Artículo específico → bypass directo (igual que TF-IDF).
        2. Caso general → búsqueda FAISS por producto interno.

        Returns:
            Lista de dicts con 'score' (similitud coseno) y 'chunk'.
        """
        if self.index is None:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")

        query = normalizar_query(query)
        query_article = extraer_numero_articulo(query)

        # --- Regla estructural: artículo específico ---
        if query_article is not None:
            resultados = [
                {"score": 1.0, "chunk": c}
                for c in self.chunks
                if c.get("article_number") == query_article
            ]
            if resultados:
                return resultados[:top_k]

        # --- Búsqueda FAISS ---
        query_emb = self.encode_query(query)                 # shape: (1, dim)
        scores, indices = self.index.search(query_emb, top_k) # scores: (1, k)

        resultados = [
            {
                "score": float(scores[0][i]),
                "chunk": self.chunks[indices[0][i]],
            }
            for i in range(len(indices[0]))
            if indices[0][i] >= 0           # FAISS retorna -1 si no hay suficientes resultados
        ]

        return resultados
