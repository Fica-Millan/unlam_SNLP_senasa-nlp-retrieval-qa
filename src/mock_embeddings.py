"""
mock_embeddings.py
------------------
EmbeddingRetriever simulado con TF-IDF SVD para entornos sin acceso
a Hugging Face (CI, sandboxes, ejecución offline).

Reemplaza SentenceTransformer con TruncatedSVD sobre TF-IDF,
manteniendo la misma interfaz pública (fit / buscar).

NO usar en producción — solo para tests y desarrollo offline.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

from utils import normalizar_query, extraer_numero_articulo


class MockEmbeddingRetriever:
    """
    Mismo contrato que EmbeddingRetriever pero sin dependencia de red.
    Embeddings = TF-IDF reducido con SVD (LSA).
    """

    def __init__(self, n_components: int = 100):
        self.vectorizer = TfidfVectorizer(lowercase=True, max_df=0.9, min_df=1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.embeddings = None
        self.chunks = None

    def fit(self, chunks: List[Dict]):
        self.chunks = chunks
        textos = [c["texto"] for c in chunks]
        tfidf_matrix = self.vectorizer.fit_transform(textos)
        raw = self.svd.fit_transform(tfidf_matrix)
        self.embeddings = normalize(raw, norm="l2")
        print(f"[MockEmbedding] {len(chunks)} chunks → shape={self.embeddings.shape}")

    def encode_query(self, query: str):
        """Codifica una query como vector normalizado (misma interfaz que EmbeddingRetriever)."""
        from sklearn.preprocessing import normalize
        query_vec = self.vectorizer.transform([query])
        raw = self.svd.transform(query_vec)
        return normalize(raw, norm="l2")

    def buscar(self, query: str, top_k: int = 5):
        if self.embeddings is None:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")

        query = normalizar_query(query)
        query_article = extraer_numero_articulo(query)

        if query_article is not None:
            resultados = [
                {"score": 1.0, "chunk": c}
                for c in self.chunks
                if c.get("article_number") == query_article
            ]
            if resultados:
                return resultados[:top_k]

        query_vec = self.vectorizer.transform([query])
        query_emb = normalize(self.svd.transform(query_vec), norm="l2")
        scores = cosine_similarity(query_emb, self.embeddings)[0]

        resultados = [
            {"score": float(scores[idx]), "chunk": self.chunks[idx]}
            for idx in range(len(self.chunks))
        ]
        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:top_k]
