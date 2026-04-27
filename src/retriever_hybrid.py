"""
retriever_hybrid.py
-------------------
Retriever híbrido: TF-IDF (léxico) + Embeddings (semántico).

Estrategia de fusión:
    score_final = alpha * score_emb + (1 - alpha) * score_tfidf

Normalización:
    Ambos scores se normalizan a [0,1] antes de fusionar (min-max),
    para que ningún modelo domine por escala.

Re-ranking heurístico (basado en estructura y contenido del corpus):
    - Penalización de ruido administrativo
    - Boost por artículo exacto (regla estructural)
    - Boost por contenido de definiciones (si la query lo indica)
    - Boost por obligaciones (si la query lo indica)

Selección de alpha:
    El parámetro alpha puede ajustarse con optimizar_alpha().
    Valores más altos priorizan semántica; más bajos priorizan léxico.
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from retriever_tfidf import TFIDFRetriever
from retriever_embeddings import EmbeddingRetriever
from utils import normalizar_query, extraer_numero_articulo, es_ruido


# =========================================================
# 🟢 NORMALIZACIÓN DE SCORES
# =========================================================

def normalizar_scores(scores: np.ndarray) -> np.ndarray:
    """Normaliza a [0,1] con min-max. Si todos son iguales, retorna tal cual."""
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return scores
    return (scores - min_s) / (max_s - min_s)


# =========================================================
# 🟢 MODELO HÍBRIDO
# =========================================================

class HybridRetriever:
    """
    Retriever que combina TF-IDF y embeddings mediante interpolación lineal.

    Args:
        alpha: Peso del modelo semántico (0=solo TF-IDF, 1=solo embeddings).
               Default=0.5. Puede optimizarse con optimizar_alpha().
    """

    def __init__(self, alpha: float = 0.8):  # ✅ es el alpha óptimo encontrado en grid search
        self.alpha = alpha
        self.tfidf = TFIDFRetriever()
        self.embedder = EmbeddingRetriever()
        self.chunks = None

    def fit(self, chunks):
        """Entrena ambos sub-modelos."""
        self.chunks = chunks
        self.tfidf.fit(chunks)
        self.embedder.fit(chunks)

    def buscar(self, query: str, top_k: int = 5):
        """
        Búsqueda híbrida con re-ranking heurístico.
        """
        if self.chunks is None:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")

        query = normalizar_query(query)
        query_article = extraer_numero_articulo(query)

        # --- Regla de artículo específico (bypass híbrido) ---
        if query_article is not None:
            resultados = [
                {"score": 1.0, "chunk": c}
                for c in self.chunks
                if c.get("article_number") == query_article
            ]
            if resultados:
                return resultados[:top_k]

        # --- Scores TF-IDF ---
        tfidf_vec = self.tfidf.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(tfidf_vec, self.tfidf.matrix)[0]

        # --- Scores embeddings ---
        query_emb = self.embedder.encode_query(query)
        emb_scores = cosine_similarity(query_emb, self.embedder.embeddings)[0]

        # --- Normalización min-max (obligatoria antes de fusionar) ---
        tfidf_scores = normalizar_scores(tfidf_scores)
        emb_scores   = normalizar_scores(emb_scores)

        # --- Detectar intención de query para heurísticas ---
        q_lower = query.lower()
        busca_definicion = any(x in q_lower for x in [
            "definicion", "definición", "que es", "que son",
            "se entiende", "significa"
        ])
        busca_obligacion = any(x in q_lower for x in [
            "obligatorio", "obligatoria", "medidas", "requisitos", "deben"
        ])

        resultados = []

        for idx, chunk in enumerate(self.chunks):
            score = self.alpha * emb_scores[idx] + (1 - self.alpha) * tfidf_scores[idx]

            texto = chunk["texto"]

            # Heurística 1: penalizar ruido administrativo
            if es_ruido(texto):
                score *= 0.3

            # Heurística 2: boost por contenido de definiciones
            if busca_definicion and chunk.get("es_definicion"):
                score *= 1.5

            # Heurística 3: boost por contenido de obligaciones
            if busca_obligacion and chunk.get("es_obligacion"):
                score *= 1.3

            resultados.append({"score": float(score), "chunk": chunk})

        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:top_k]


# =========================================================
# 🟢 GRID SEARCH DE ALPHA
# =========================================================

def optimizar_alpha(chunks, eval_queries, tfidf, embedder, alphas=None, k=5):
    """
    Busca el valor de alpha que maximiza el MRR sobre un conjunto
    de queries de evaluación.

    Reutiliza los sub-modelos ya entrenados (tfidf y embedder) para
    no re-entrenar en cada iteración — solo varía el peso alpha.

    Args:
        chunks:       Lista de chunks indexados.
        eval_queries: Lista de dicts con 'query' y 'relevantes'.
        tfidf:        TFIDFRetriever ya entrenado.
        embedder:     EmbeddingRetriever (real o Mock) ya entrenado.
        alphas:       Lista de valores a probar (default: 0.0 a 1.0, paso 0.1).
        k:            Número de resultados a evaluar.

    Returns:
        Dict con mejor_alpha y tabla {alpha: MRR}.
    """
    from evaluation import evaluar_modelo_detallado 

    if alphas is None:
        alphas = [round(a * 0.1, 1) for a in range(0, 11)]

    resultados_alpha = {}

    print("Grid search de alpha (con modelos reales):")
    print("-" * 45)

    for alpha in alphas:
        retriever = build_hybrid(chunks, tfidf=tfidf, embedder=embedder, alpha=alpha)
        eval_result = evaluar_modelo_detallado(retriever, eval_queries, k=k)
        mrr = eval_result["metricas"]["MRR"]
        resultados_alpha[alpha] = mrr
        print(f"  alpha={alpha:.1f} → MRR={mrr:.3f}")

    mejor_alpha = max(resultados_alpha, key=resultados_alpha.get)
    print(f"\nMejor alpha: {mejor_alpha} (MRR={resultados_alpha[mejor_alpha]:.3f})")

    return {
        "mejor_alpha": mejor_alpha,
        "tabla": resultados_alpha,
    }


# =========================================================
# 🟢 FACTORY: construye híbrido reutilizando sub-modelos
# =========================================================

def build_hybrid(chunks, tfidf, embedder, alpha=0.8): #✅ usa el alpha óptimo encontrado en grid search
    """
    Construye un HybridRetriever reutilizando sub-modelos ya entrenados.

    Evita re-entrenar TF-IDF y embeddings en cada variación de alpha.
    Funciona con EmbeddingRetriever real o MockEmbeddingRetriever.

    Args:
        chunks:   Lista de chunks indexados.
        tfidf:    TFIDFRetriever ya entrenado.
        embedder: EmbeddingRetriever (real o Mock) ya entrenado.
        alpha:    Peso del modelo semántico [0, 1].

    Returns:
        HybridRetriever listo para usar.
    """
    h = HybridRetriever.__new__(HybridRetriever)
    h.alpha = alpha
    h.chunks = chunks
    h.tfidf = tfidf
    h.embedder = embedder
    return h


def build_hybrid_offline(chunks, alpha=0.5):
    """
    Construye un HybridRetriever con MockEmbeddingRetriever
    para entornos sin acceso a Hugging Face (tests, CI).

    En producción usar build_hybrid() con EmbeddingRetriever real.
    """
    from mock_embeddings import MockEmbeddingRetriever

    tfidf = TFIDFRetriever()
    tfidf.fit(chunks)
    embedder = MockEmbeddingRetriever()
    embedder.fit(chunks)
    return build_hybrid(chunks, tfidf=tfidf, embedder=embedder, alpha=alpha)
