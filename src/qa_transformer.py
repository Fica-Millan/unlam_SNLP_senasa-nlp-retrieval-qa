"""
qa_transformer.py
-----------------
QA extractivo basado en Transformer (BERT fine-tuneado en SQuAD en español).

Pipeline:
    query + chunks → selección del mejor chunk → BERT QA → span exacto

Modelo utilizado:
    mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es
    - Arquitectura: BERT-base (12 capas, 110M parámetros)
    - Fine-tuning: SQuAD 2.0 traducido al español
    - Idioma: español
    - Tarea: extractive QA (question answering)

Diferencia con qa_engine.py:
    qa_engine.py usa heurísticas (overlap léxico + reglas) para construir
    la respuesta. Este módulo usa un Transformer entrenado para encontrar
    el span exacto del texto que responde la pregunta.

Ventaja:
    Mayor precisión en la extracción del fragmento exacto que responde
    la pregunta, especialmente en consultas de tipo definicional.

Limitación:
    La calidad de la respuesta depende directamente de que el retriever
    haya devuelto el chunk correcto en los primeros resultados.
    Si el chunk relevante no está en top-k, el Transformer no puede
    compensar esa falla.

Instalación:
    pip install transformers torch

Uso offline (sin internet):
    Descargar el modelo una vez con:
        python -c "from transformers import pipeline; pipeline('question-answering',
        model='mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es')"
    Luego usar TRANSFORMERS_OFFLINE=1 para evitar llamadas a HuggingFace.
"""

from typing import List, Dict, Optional

from torch import device

# Nombre del modelo por defecto — puede sobreescribirse en QATransformer.__init__
MODELO_QA_ES = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"

# Score mínimo de confianza para aceptar la respuesta del Transformer.
# Por debajo de este umbral se considera que el modelo no encontró
# una respuesta confiable en el chunk.
SCORE_MINIMO = 0.05


# =========================================================
# 🟢 MOTOR QA CON TRANSFORMER
# =========================================================

class QATransformer:
    """
    Motor de QA basado en Transformer (BERT fine-tuneado en SQuAD español).

    Carga el modelo una sola vez en __init__ para evitar re-cargar
    los pesos en cada consulta (carga ~1-2 segundos en CPU).

    Uso:
        qa = QATransformer()
        resultado = qa.responder(resultados_retriever, "qué es la brucelosis")
    """

    def __init__(self, model_name: str = MODELO_QA_ES, device: int = -1):
        """
        Inicializa el pipeline de QA.

        Args:
            model_name: nombre del modelo en HuggingFace Hub.
            device:     -1 = CPU, 0 = GPU (si disponible).
        """
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "Instalar transformers: pip install transformers torch"
            )

        print(f"[QATransformer] Cargando modelo '{model_name}'...")
        self._pipeline = hf_pipeline(
            "question-answering",
            model=model_name,
            device=device,
        )
        self._model_name = model_name
        print("[QATransformer] Modelo listo.")

    def responder(
        self,
        resultados: List[Dict],
        query: str,
        top_n: int = 3,
        score_minimo: float = SCORE_MINIMO,
    ) -> Dict:
        """
        Genera una respuesta usando el Transformer sobre los top_n chunks.

        Estrategia multi-chunk:
            1. Aplica el pipeline de QA sobre cada uno de los top_n chunks.
            2. Selecciona el span con mayor score de confianza.
            3. Si ningún chunk supera el score mínimo, retorna fallback.

        Args:
            resultados:    salida del retriever [{score, chunk}, ...]
            query:         consulta del usuario
            top_n:         chunks candidatos a evaluar (default: 3)
            score_minimo:  confianza mínima para aceptar la respuesta

        Returns:
            Dict con:
                - respuesta:     span exacto extraído por el Transformer
                - score_qa:      confianza del modelo (0-1)
                - score_retrieval: relevancia del chunk devuelto por el retriever
                - fuente:        doc_id del chunk fuente
                - seccion:       sección normativa del chunk
                - contexto:      texto completo del chunk usado como contexto
                - metodo:        "transformer"
                - modelo:        nombre del modelo usado
        """
        if not resultados:
            return _respuesta_vacia(query, self._model_name)

        top = resultados[:top_n]
        mejor_resultado = None
        mejor_score_qa = -1.0

        for r in top:
            chunk = r["chunk"]
            contexto = chunk.get("texto", "").strip()

            if len(contexto.split()) < 10:
                continue

            try:
                raw = self._pipeline(
                    question=query,
                    context=contexto,
                    handle_impossible_answer=True,
                )
            except Exception:
                continue

            score_qa = raw.get("score", 0.0)
            answer   = raw.get("answer", "").strip()

            # Ignorar respuestas vacías o de muy baja confianza
            if not answer or score_qa < score_minimo:
                continue

            if score_qa > mejor_score_qa:
                mejor_score_qa = score_qa
                mejor_resultado = {
                    "respuesta":          answer,
                    "score_qa":           round(score_qa, 4),
                    "score_retrieval":    round(r["score"], 4),
                    "fuente":             chunk.get("doc_id", "desconocido"),
                    "seccion":            chunk.get("seccion", ""),
                    "article_number":     chunk.get("article_number"),
                    "contexto":           contexto,
                    "metodo":             "transformer",
                    "modelo":             self._model_name,
                }

        if mejor_resultado is None:
            return _respuesta_vacia(query, self._model_name)

        return mejor_resultado


# =========================================================
# 🟢 FALLBACK
# =========================================================

def _respuesta_vacia(query: str, model_name: str) -> Dict:
    """Respuesta cuando el Transformer no encontró un span confiable."""
    return {
        "respuesta":       "El modelo no encontró una respuesta con suficiente confianza en los fragmentos recuperados.",
        "score_qa":        0.0,
        "score_retrieval": 0.0,
        "fuente":          None,
        "seccion":         "",
        "article_number":  None,
        "contexto":        "",
        "metodo":          "transformer",
        "modelo":          model_name,
    }


# =========================================================
# 🟢 MOCK PARA TESTS OFFLINE
# =========================================================

class MockQATransformer:
    """
    Implementación mock de QATransformer para tests sin acceso a HuggingFace.

    Simula la interfaz exacta de QATransformer usando extracción de span
    por overlap léxico (sin modelo neuronal). Garantiza que los tests
    validen el contrato de la interfaz sin depender de descargas de red.

    NO usar en producción.
    """

    def __init__(self, model_name: str = MODELO_QA_ES, device: int = -1):
        self._model_name = f"MOCK({model_name})"
        print(f"[MockQATransformer] Usando mock (sin modelo real).")

    def responder(
        self,
        resultados: List[Dict],
        query: str,
        top_n: int = 3,
        score_minimo: float = SCORE_MINIMO,
    ) -> Dict:
        """Simula QA extrayendo la oración con mayor overlap léxico."""
        import re

        if not resultados:
            return _respuesta_vacia(query, self._model_name)

        palabras_query = {
            w.lower() for w in re.findall(r"\w+", query)
            if len(w) >= 3
        }

        mejor_span = None
        mejor_score = -1.0
        mejor_chunk = None
        mejor_r = None

        for r in resultados[:top_n]:
            chunk = r["chunk"]
            texto = chunk.get("texto", "")
            oraciones = re.split(r"(?<=[.!?])\s+", texto.strip())

            for oracion in oraciones:
                if len(oracion.split()) < 5:
                    continue
                palabras = {w.lower() for w in re.findall(r"\w+", oracion)}
                score = len(palabras_query & palabras) / max(len(palabras_query), 1)
                if score > mejor_score:
                    mejor_score = score
                    mejor_span = oracion.strip()
                    mejor_chunk = chunk
                    mejor_r = r

        if mejor_span is None or mejor_score < score_minimo:
            return _respuesta_vacia(query, self._model_name)

        return {
            "respuesta":       mejor_span,
            "score_qa":        round(min(mejor_score, 1.0), 4),
            "score_retrieval": round(mejor_r["score"], 4),
            "fuente":          mejor_chunk.get("doc_id", "desconocido"),
            "seccion":         mejor_chunk.get("seccion", ""),
            "article_number":  mejor_chunk.get("article_number"),
            "contexto":        mejor_chunk.get("texto", ""),
            "metodo":          "transformer",
            "modelo":          self._model_name,
        }


# =========================================================
# 🟢 FACTORY (real o mock según disponibilidad)
# =========================================================

def get_qa_transformer(
    model_name: str = MODELO_QA_ES,
    device: int = -1,
    forzar_mock: bool = False,
) -> "QATransformer | MockQATransformer":
    """
    Retorna un QATransformer real o Mock según disponibilidad del modelo.

    Args:
        model_name:   modelo de HuggingFace a usar.
        device:       -1=CPU, 0=GPU.
        forzar_mock:  True para forzar el mock (tests offline).

    Returns:
        QATransformer si el modelo está disponible, MockQATransformer si no.
    """
    if forzar_mock:
        return MockQATransformer(model_name, device)
    try:
        return QATransformer(model_name, device)
    except Exception as e:
        print(f"[QATransformer] No disponible ({type(e).__name__}). Usando Mock.")
        return MockQATransformer(model_name, device)
