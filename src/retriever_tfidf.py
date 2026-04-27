"""
retriever_tfidf.py
------------------
Modelo de recuperación léxica basado en TF-IDF + similitud coseno.

Decisiones de diseño:
- ngram_range=(1,3): captura frases normativas ("producto veterinario")
- Stopwords personalizadas en español
- Regla directa para búsqueda por artículo (bypass TF-IDF)
- Sin boosts ad-hoc: el ranking depende únicamente de TF-IDF coseno
  más la regla estructural de artículo
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import normalizar_query, extraer_numero_articulo


STOPWORDS_ES = [
    "el", "la", "los", "las", "lo",
    "de", "del", "y", "a", "en",
    "que", "por", "con", "para",
    "un", "una", "unos", "unas",
    "se", "es", "son", "fue", "ser",
    "al", "como", "más", "pero", "sus", "le", "ya",
    "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay",
    "donde", "quien", "quienes", "cual", "cuales",
    "dice", "dicen", "hacer", "hace",
]


class TFIDFRetriever:
    """
    Recuperador basado en TF-IDF con similitud coseno.

    Uso:
        retriever = TFIDFRetriever()
        retriever.fit(chunks)
        resultados = retriever.buscar("brucelosis bovina", top_k=5)
    """

    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.chunks = None

    def fit(self, chunks):
        """Indexa los chunks con TF-IDF."""
        self.chunks = chunks
        textos = [c["texto"] for c in chunks]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=STOPWORDS_ES,
            ngram_range=(1, 3),
            max_df=0.9,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(textos)

    def buscar(self, query: str, top_k: int = 5):
        """
        Devuelve los top_k chunks más relevantes para la query.

        Lógica:
        1. Si la query menciona un artículo específico, retorna
           directamente los chunks de ese artículo (matching estructural).
        2. En caso contrario, usa coseno TF-IDF puro sin boosts.
        """
        if self.vectorizer is None:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")

        query = normalizar_query(query)
        query_article = extraer_numero_articulo(query)

        # --- Regla de artículo específico ---
        if query_article is not None:
            resultados = [
                {"score": 1.0, "chunk": c}
                for c in self.chunks
                if c.get("article_number") == query_article
            ]
            if resultados:
                return resultados[:top_k]

        # --- TF-IDF coseno ---
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        resultados = [
            {"score": float(scores[idx]), "chunk": self.chunks[idx]}
            for idx in range(len(self.chunks))
        ]

        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:top_k]
