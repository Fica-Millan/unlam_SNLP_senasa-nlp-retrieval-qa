"""
qa_engine.py
------------
Responsabilidad: dado un conjunto de chunks recuperados, generar
una respuesta en lenguaje natural a la consulta del usuario.

Estrategia: extractive QA + síntesis heurística.

Pipeline:
    query + chunks → selección de candidatos → extracción de respuesta

Este módulo es agnóstico al retriever: funciona igual con TF-IDF,
Embeddings o Híbrido como fuente de chunks.

Dos niveles de respuesta:
    1. Extractivo puro     → devuelve el pasaje más relevante sin modificar
    2. Síntesis heurística → combina múltiples chunks en una respuesta
                             estructurada (sin modelo generativo)

Dependencias de utils.py (punto único de verdad):
    - clasificar_query()       → tipo de intención del usuario
    - extraer_numero_articulo()→ número de artículo en la query
    - es_ruido()               → detección de ruido administrativo
"""
 
import re
from typing import List, Dict

from utils import clasificar_query, extraer_numero_articulo, es_ruido


# =========================================================
# 🟢 STOPWORDS PARA SCORING DE ORACIONES
# =========================================================

# Palabras a ignorar al calcular overlap léxico entre query y oración.
# Umbral de longitud reducido a 3 para no perder términos normativos
# cortos relevantes: "ley", "res", "uso", "ova", etc.
STOPWORDS_SCORING = {
    "que", "qué", "con", "del", "los", "las", "una", "uno",
    "para", "por", "como", "esto", "esta", "este",
}


# =========================================================
# 🟢 EXTRACCIÓN DE ORACIÓN MÁS RELEVANTE
# =========================================================

def extraer_oracion_relevante(texto: str, query: str) -> str:
    """
    Busca la oración más relevante dentro de un chunk.

    Estrategia: score por cobertura de palabras de la query.
    Devuelve la oración con mayor overlap léxico.

    Umbral mínimo de longitud: 3 caracteres (para no perder
    términos normativos cortos como 'ley', 'res', 'uso').

    Args:
        texto:  texto del chunk
        query:  consulta del usuario

    Returns:
        str: la oración más relevante del chunk
    """
    palabras_query = {
        w.lower() for w in re.findall(r"\w+", query)
        if len(w) >= 3 and w.lower() not in STOPWORDS_SCORING
    }

    oraciones = re.split(r"(?<=[.!?])\s+", texto.strip())

    if not oraciones:
        return ""

    mejor_oracion = oraciones[0]
    mejor_score = -1

    for oracion in oraciones:
        if len(oracion.split()) < 6:
            continue
        palabras_oracion = {w.lower() for w in re.findall(r"\w+", oracion)}
        score = len(palabras_query & palabras_oracion)
        if score > mejor_score:
            mejor_score = score
            mejor_oracion = oracion

    return mejor_oracion.strip()


# =========================================================
# 🟢 QA EXTRACTIVO PURO
# =========================================================

def qa_extractivo(resultados: List[Dict], query: str) -> Dict:
    """
    QA extractivo: devuelve el pasaje más relevante sin modificar.

    Útil como baseline de QA y para validación del retriever.

    Args:
        resultados: salida del retriever (lista de {score, chunk})
        query:      consulta original

    Returns:
        Dict con respuesta, fuente, score y metadata
    """
    if not resultados:
        return {
            "respuesta": "No se encontraron fragmentos relevantes para la consulta.",
            "fuente": None,
            "score": 0.0,
            "metodo": "extractivo",
        }

    mejor = resultados[0]
    chunk = mejor["chunk"]
    oracion = extraer_oracion_relevante(chunk["texto"], query)

    return {
        "respuesta": oracion,
        "pasaje_completo": chunk["texto"],
        "fuente": chunk.get("doc_id", "desconocido"),
        "seccion": chunk.get("seccion", ""),
        "score": round(mejor["score"], 4),
        "metodo": "extractivo",
    }


# =========================================================
# 🟢 QA POR SÍNTESIS HEURÍSTICA
# =========================================================

def _limpiar_prefijo(texto: str) -> str:
    """Elimina prefijos estructurales normativos que no aportan a la respuesta."""
    # Eliminar encabezado CONSIDERANDO (con uno o más 'Que' seguidos)
    texto = re.sub(
        r"^CONSIDERANDO:\s*(Que\s*)+",
        "",
        texto,
        flags=re.IGNORECASE,
    )
    # Eliminar encabezado ARTÍCULO N
    texto = re.sub(
        r"^ARTÍCULO\s+\d+[\s°.:-]*",
        "",
        texto,
        flags=re.IGNORECASE,
    )
    return texto.strip()


def _es_ruido_oracion(oracion: str) -> bool:
    """
    Detecta si una oración no aporta valor como respuesta.

    Combina la detección de ruido administrativo de utils.es_ruido()
    con heurísticas estructurales (oraciones muy cortas, títulos en
    mayúsculas, líneas que terminan en dos puntos).
    """
    if es_ruido(oracion):
        return True

    palabras = oracion.split()

    if len(palabras) < 8:
        return True

    if oracion.strip().endswith(":"):
        return True

    # Títulos en mayúsculas (sin contenido real)
    if oracion.isupper():
        return True

    return False


def qa_sintetico(resultados: List[Dict], query: str, top_n: int = 3) -> Dict:
    """
    Genera una respuesta sintética combinando los top_n chunks.

    Estrategia:
        1. Detectar tipo de consulta (desde utils.clasificar_query)
        2. Filtrar chunks según tipo usando flags semánticos del corpus
        3. Extraer oración clave de cada chunk
        4. Armar respuesta estructurada según tipo

    Args:
        resultados: salida del retriever
        query:      consulta original
        top_n:      chunks a considerar

    Returns:
        Dict con respuesta sintetizada y fuentes
    """
    if not resultados:
        return {
            "respuesta": "No se encontraron fragmentos relevantes para la consulta.",
            "fuentes": [],
            "metodo": "sintetico",
        }

    tipo = clasificar_query(query)

    # =========================================
    # FILTRO POR TIPO USANDO FLAGS DEL CORPUS
    # =========================================
    # Usa los flags es_definicion / es_obligacion generados en chunker.py,
    # no búsqueda de texto crudo, para consistencia con el pipeline.

    if tipo == "definicion":
        filtrados = [r for r in resultados if r["chunk"].get("es_definicion")]
        top = (filtrados or resultados)[:top_n]

    elif tipo == "articulo":
        num = extraer_numero_articulo(query)
        if num is not None:
            filtrados = [
                r for r in resultados
                if r["chunk"].get("article_number") == num
            ]
            top = (filtrados or resultados)[:top_n]
        else:
            top = resultados[:top_n]

    elif tipo == "procedimiento":
        filtrados = [r for r in resultados if r["chunk"].get("es_procedimiento")]
        top = (filtrados or resultados)[:top_n]

    else:
        top = resultados[:top_n]

    # =========================================
    # EXTRACCIÓN Y LIMPIEZA DE ORACIONES
    # =========================================

    oraciones_limpias = []
    fuentes = []

    for r in top:
        if "chunk" not in r:
            continue
        chunk = r["chunk"]
        oracion = extraer_oracion_relevante(chunk["texto"], query)
        if not oracion:
            continue
        limpia = _limpiar_prefijo(oracion)
        if not _es_ruido_oracion(limpia):
            oraciones_limpias.append(limpia)
            fuentes.append({
                "doc_id":  chunk.get("doc_id", "?"),
                "seccion": chunk.get("seccion", "?"),
                "score":   round(r["score"], 4),
            })

    # =========================================
    # ARMADO DE RESPUESTA SEGÚN TIPO
    # =========================================

    if tipo == "definicion" and oraciones_limpias:
        respuesta = f"Definición: {oraciones_limpias[0]}"
        if len(oraciones_limpias) > 1:
            respuesta += f" {oraciones_limpias[1]}"

    elif tipo == "articulo" and oraciones_limpias:
        respuesta = f"El artículo consultado establece: {oraciones_limpias[0]}"

    elif tipo == "procedimiento" and oraciones_limpias:
        partes = [f"({i+1}) {o}" for i, o in enumerate(oraciones_limpias)]
        respuesta = "La normativa establece: " + " ".join(partes)

    elif oraciones_limpias:
        respuesta = oraciones_limpias[0]

    else:
        # Fallback: texto crudo del mejor chunk sin prefijo
        raw = resultados[0]["chunk"]["texto"][:300] if resultados else ""
        respuesta = _limpiar_prefijo(raw) or "No se encontró información específica."

    return {
        "respuesta": respuesta,
        "tipo_query": tipo,
        "fuentes": fuentes,
        "n_chunks_usados": len(oraciones_limpias),
        "metodo": "sintetico",
    }


# =========================================================
# 🟢 INTERFAZ PRINCIPAL
# =========================================================

def responder(
    resultados: List[Dict],
    query: str,
    modo: str = "sintetico",
    top_n: int = 3,
    qa_transformer_instance=None,
) -> Dict:
    """
    Punto de entrada del módulo QA.

    Args:
        resultados:            salida del retriever
        query:                 consulta del usuario
        modo:                  "extractivo" | "sintetico" | "transformer"
        top_n:                 chunks a considerar (modos sintético y transformer)
        qa_transformer_instance: instancia de QATransformer o MockQATransformer.
                                 Requerido cuando modo="transformer".
                                 Si es None y modo="transformer", se intenta
                                 cargar el modelo automáticamente.

    Returns:
        Dict con respuesta y metadata.
        En modo "transformer" incluye además score_qa y contexto.
    """
    if modo == "extractivo":
        return qa_extractivo(resultados, query)

    elif modo == "sintetico":
        return qa_sintetico(resultados, query, top_n=top_n)

    elif modo == "transformer":
        if qa_transformer_instance is None:
            from qa_transformer import get_qa_transformer
            qa_transformer_instance = get_qa_transformer()
        return qa_transformer_instance.responder(resultados, query, top_n=top_n)

    else:
        raise ValueError(
            f"Modo no válido: {modo!r}. "
            "Usar 'extractivo', 'sintetico' o 'transformer'."
        )
