"""
rag_distilbert.py
-----------------
Pipeline RAG (Retrieval-Augmented Generation) con DistilBERT.

Arquitectura:
    query → Retriever (FAISS + embeddings) → top-k chunks
          → contexto concatenado
          → DistilBERT QA extractivo → span exacto

Modelo:
    mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es
    - Arquitectura: DistilBERT destilado del modelo BERT español
    - Fine-tuning: SQuAD 2.0 traducido al español con knowledge distillation
    - Idioma: español
    - Tarea: QA extractivo (extrae el span exacto que responde la pregunta)
    - 40% menos parámetros que BERT-base, ~60% más rápido en inferencia

Diferencia con qa_transformer.py:
    qa_transformer.py evalúa cada chunk por separado y toma el mejor span.
    Este módulo concatena los chunks en un único contexto y aplica DistilBERT
    una sola vez.

Instalación:
    pip install transformers torch faiss-cpu sentence-transformers

Descarga del modelo (una sola vez):
    python -c "
    from transformers import pipeline
    pipeline('question-answering',
             model='mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es')
    "
"""

from typing import List, Dict, Optional

MODELO_DISTILBERT_ES = (
    "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
)

# Máximo de caracteres del contexto que DistilBERT puede procesar.
# DistilBERT tiene límite de 512 tokens ≈ ~1500-2000 caracteres en español.
MAX_CONTEXTO_CHARS = 1800

# Score mínimo de confianza para aceptar la respuesta.
SCORE_MINIMO = 0.01


# =========================================================
# 🟢 PIPELINE RAG
# =========================================================

class RAGDistilBERT:
    """
    Pipeline RAG completo: recuperación con FAISS + generación con DistilBERT.

    Flujo:
        1. query → retriever.buscar() → chunks relevantes
        2. chunks → contexto concatenado (truncado a MAX_CONTEXTO_CHARS)
        3. (query, contexto) → DistilBERT QA → span exacto

    Uso:
        rag = RAGDistilBERT(retriever)
        resultado = rag.preguntar("¿qué es la brucelosis bovina?")
    """

    def __init__(self, retriever, model_name: str = MODELO_DISTILBERT_ES):
        """
        Args:
            retriever:   EmbeddingRetriever ya entrenado con FAISS.
            model_name:  modelo DistilBERT fine-tuneado en QA español.
        """
        self.retriever = retriever
        self._model_name = model_name
        self._pipeline = self._cargar_pipeline(model_name)

    def _cargar_pipeline(self, model_name: str):
        """Carga el pipeline de HuggingFace. Retorna None si no está disponible."""
        try:
            from transformers import pipeline as hf_pipeline
            print(f"[RAGDistilBERT] Cargando modelo '{model_name}'...")
            pipe = hf_pipeline(
                "question-answering",
                model=model_name,
                device=-1,
            )
            print("[RAGDistilBERT] Modelo listo.")
            return pipe
        except Exception as e:
            print(f"[RAGDistilBERT] Modelo no disponible ({type(e).__name__}). "
                  "Usando modo fallback (extractivo heurístico).")
            return None

    def _construir_contexto(self, chunks: List[Dict]) -> str:
        """
        Concatena los chunks en un único contexto truncado.

        DistilBERT tiene límite de 512 tokens. Truncamos el contexto
        para evitar errores y mantener la información más relevante
        (los chunks ya vienen ordenados por score descendente).
        """
        contexto = "\n\n".join(c["texto"] for c in chunks)
        if len(contexto) > MAX_CONTEXTO_CHARS:
            contexto = contexto[:MAX_CONTEXTO_CHARS]
        return contexto

    def preguntar(self, query: str, top_k: int = 4) -> Dict:
        """
        Responde una pregunta usando RAG completo.

        Args:
            query:  pregunta en lenguaje natural.
            top_k:  chunks a recuperar (default: 4)

        Returns:
            Dict con:
                - respuesta:         span extraído por DistilBERT
                - score_qa:          confianza del modelo (0-1)
                - contexto:          texto usado como contexto
                - chunks_usados:     lista de chunks recuperados
                - fuentes:           doc_id + sección de cada chunk
                - metodo:            "rag_distilbert"
                - modelo:            nombre del modelo
        """
        # 1. Recuperar chunks con FAISS
        resultados = self.retriever.buscar(query, top_k=top_k)

        if not resultados:
            return self._respuesta_vacia(query)

        # 2. Construir contexto
        chunks_recuperados = [r["chunk"] for r in resultados]
        contexto = self._construir_contexto(chunks_recuperados)

        fuentes = [
            {
                "doc_id":  c.get("doc_id", "?"),
                "seccion": c.get("seccion", "?"),
                "score":   round(r["score"], 4),
            }
            for r, c in zip(resultados, chunks_recuperados)
        ]

        # 3. Aplicar DistilBERT QA sobre el contexto completo
        if self._pipeline is not None:
            try:
                raw = self._pipeline(
                    question=query,
                    context=contexto,
                    handle_impossible_answer=True,
                )
                respuesta  = raw.get("answer", "").strip()
                score_qa   = raw.get("score", 0.0)

                if not respuesta or score_qa < SCORE_MINIMO:
                    respuesta = self._fallback_extractivo(contexto, query)
                    score_qa  = 0.0

            except Exception as e:
                print(f"[RAGDistilBERT] Error en inferencia: {e}")
                respuesta = self._fallback_extractivo(contexto, query)
                score_qa  = 0.0
        else:
            # Fallback si el modelo no está disponible
            respuesta = self._fallback_extractivo(contexto, query)
            score_qa  = 0.0

        return {
            "respuesta":      respuesta,
            "score_qa":       round(score_qa, 4),
            "contexto":       contexto,
            "chunks_usados":  chunks_recuperados,
            "fuentes":        fuentes,
            "metodo":         "rag_distilbert",
            "modelo":         self._model_name,
        }

    def _fallback_extractivo(self, contexto: str, query: str) -> str:
        """
        Extracción heurística cuando DistilBERT no está disponible
        o no encontró respuesta confiable.
        """
        import re
        palabras_query = {
            w.lower() for w in re.findall(r"\w+", query)
            if len(w) >= 3
        }
        oraciones = re.split(r"(?<=[.!?])\s+", contexto.strip())
        mejor = ""
        mejor_score = -1
        for o in oraciones:
            if len(o.split()) < 6:
                continue
            palabras = {w.lower() for w in re.findall(r"\w+", o)}
            score = len(palabras_query & palabras)
            if score > mejor_score:
                mejor_score = score
                mejor = o
        return mejor.strip() or contexto[:300]

    def _respuesta_vacia(self, query: str) -> Dict:
        return {
            "respuesta":     "No se encontraron fragmentos relevantes para la consulta.",
            "score_qa":      0.0,
            "contexto":      "",
            "chunks_usados": [],
            "fuentes":       [],
            "metodo":        "rag_distilbert",
            "modelo":        self._model_name,
        }


# =========================================================
# 🟢 DEMO INTERACTIVO
# =========================================================

def demo_interactivo(rag: RAGDistilBERT):
    """
    Loop interactivo de preguntas y respuestas.
    Permite probar el pipeline RAG con DistilBERT sin necesidad de levantar la API.
    """
    print("\n" + "=" * 60)
    print("Sistema RAG — DistilBERT + FAISS")
    print("Escribí 'salir' para terminar.")
    print("=" * 60)

    while True:
        query = input("\nHacé una pregunta: ").strip()
        if query.lower() in ("salir", "exit", "quit"):
            print("Saliendo.")
            break
        if not query:
            continue

        resultado = rag.preguntar(query)

        print(f"\nContexto usado ({len(resultado['contexto'])} chars):")
        print(f"  {resultado['contexto'][:300]}...")
        print(f"\nPregunta: {query}")
        print(f"Respuesta: {resultado['respuesta']}")
        print(f"Score DistilBERT: {resultado['score_qa']:.4f}")
        print("\nFuentes:")
        for f in resultado["fuentes"]:
            print(f"  - {f['doc_id']} | {f['seccion']} | score_retrieval={f['score']}")
