"""
utils.py
--------
Funciones compartidas entre módulos del pipeline NLP.

Centraliza:
- Normalización de queries
- Clasificación semántica de chunks
- Detección de ruido normativo
- Parsing de número de artículo
"""

import re
from typing import Optional


# =========================================================
# 🟢 NORMALIZACIÓN DE QUERY
# =========================================================

def normalizar_query(q: str) -> str:
    """
    Normaliza una query de usuario para mejorar el matching:
    - Minúsculas
    - Tildes en interrogativos
    - Normalización de 'articulo' → 'artículo'
    - Normalización de 'definicion' → 'definición'
    """
    q = q.lower().strip()
    q = q.replace("qué", "que")
    q = q.replace("cómo", "como")
    q = q.replace("cuál", "cual")
    q = re.sub(r"\barticulo\b", "artículo", q)
    q = re.sub(r"\bdefinicion\b", "definición", q)
    q = re.sub(r"\binscripcion\b", "inscripción", q)
    q = re.sub(r"\bidentificacion\b", "identificación", q)
    return q


# =========================================================
# 🟢 PARSING DE NÚMERO DE ARTÍCULO
# =========================================================

def extraer_numero_articulo(query: str) -> Optional[int]:
    """
    Extrae el número de artículo mencionado en la query, si existe.

    Ejemplos:
        "articulo 3"  → 3
        "artículo 20" → 20
        "brucelosis"  → None
    """
    match = re.search(r"art[ií]culo\s+(\d+)", query.lower())
    if match:
        return int(match.group(1))
    return None


# =========================================================
# 🟢 CLASIFICACIÓN SEMÁNTICA DE CHUNKS
# =========================================================

KEYWORDS_DEFINICION = [
    "definiciones",
    "se entiende por",
    "se define como",
    "a los fines de la presente",
    "a los efectos de",
    "se denominará",
    "se denominan",
]

KEYWORDS_OBLIGACION = [
    "debe",
    "deben",
    "deberá",
    "deberán",
    "obligatorio",
    "obligatoria",
    "obligatoriamente",
    "se establece",
    "se dispone",
    "queda prohibido",
    "está prohibido",
]

KEYWORDS_PROCEDIMIENTO = [
    "procedimiento",
    "trámite",
    "solicitud",
    "registro",
    "inscripción",
    "habilitación",
    "autorización",
    "presentar",
    "presentará",
]

KEYWORDS_SANCION = [
    "infracción",
    "sanción",
    "multa",
    "penalidad",
    "clausura",
]


def clasificar_chunk(texto: str) -> dict:
    """
    Detecta categorías semánticas presentes en el texto.

    Returns:
        dict con flags booleanos: es_definicion, es_obligacion,
        es_procedimiento, es_sancion
    """
    t = texto.lower()
    return {
        "es_definicion":    any(k in t for k in KEYWORDS_DEFINICION),
        "es_obligacion":    any(k in t for k in KEYWORDS_OBLIGACION),
        "es_procedimiento": any(k in t for k in KEYWORDS_PROCEDIMIENTO),
        "es_sancion":       any(k in t for k in KEYWORDS_SANCION),
    }


# =========================================================
# 🟢 DETECCIÓN DE RUIDO ADMINISTRATIVO
# =========================================================

KEYWORDS_RUIDO = [
    "comuníquese",
    "comuniquese",
    "publíquese",
    "publiquese",
    "archívese",
    "archivese",
    "registro oficial",
    "boletín oficial",
    "boletin oficial",
    "dése al",
    "pase a",
    "intervenga",
]


def es_ruido(texto: str) -> bool:
    """
    Detecta si un chunk es ruido administrativo (cierre formal
    de resolución sin contenido normativo relevante).
    """
    t = texto.lower()
    return any(k in t for k in KEYWORDS_RUIDO)


# =========================================================
# 🟢 DETECCIÓN DE TIPO DE QUERY
# =========================================================

def detectar_tipo_query(query: str) -> str:
    """
    Clasifica la query en una de tres categorías para orientar
    el re-ranking:
        'lexica'   → búsqueda de artículo específico
        'semantica'→ definiciones, explicaciones
        'mixta'    → obligaciones, procedimientos, requisitos
    """
    q = query.lower()

    if re.search(r"art[ií]culo\s+\d+", q):
        return "lexica"

    if any(x in q for x in [
        "que es", "que son", "definicion", "definición",
        "se entiende", "significa", "explica"
    ]):
        return "semantica"

    if any(x in q for x in [
        "obligatorio", "obligatoria", "requisito", "requisitos",
        "medida", "medidas", "como se", "procedimiento", "trámite"
    ]):
        return "mixta"

    return "semantica"  # default


# =========================================================
# 🟢 CLASIFICACIÓN DE QUERY PARA QA (granularidad fina)
# =========================================================

PATRONES_QA = {
    "definicion": [
        r"\bqué es\b", r"\bque es\b", r"\bqué son\b", r"\bque son\b",
        r"\bdefinición\b", r"\bdefinicion\b",
        r"\bsignifica\b", r"\bconsiste en\b",
        r"\bse entiende\b",
    ],
    "articulo": [
        r"\bart[ií]culo\s+\d+\b",
        r"\bque dice el art\b",
        r"\bart\.\s*\d+\b",
    ],
    "procedimiento": [
        r"\bcómo\b", r"\bcomo\b",
        r"\bpasos\b", r"\bprocedimiento\b",
        r"\brequisitos\b", r"\bobligacion\b",
        r"\bdebe\b", r"\btiene que\b",
        r"\bse debe\b", r"\bhay que\b",
    ],
}


def clasificar_query(query: str) -> str:
    """
    Clasifica la intención de la query con granularidad fina para QA.

    Returns:
        "definicion" | "articulo" | "procedimiento" | "general"

    Nota: más granular que detectar_tipo_query(), que usa categorías
    de IR (lexica/semantica/mixta). Usar esta función en qa_engine.py
    y detectar_tipo_query() en retrievers y API.
    """
    q = query.lower()
    for tipo, patrones in PATRONES_QA.items():
        for patron in patrones:
            if re.search(patron, q):
                return tipo
    return "general"


def detectar_tipo_query(query: str) -> str:
    """
    Clasifica la query en categorías de IR para orientar el re-ranking.

    Returns:
        'lexica' | 'semantica' | 'mixta'

    Mapea internamente desde clasificar_query() para mantener
    un único punto de clasificación.
    """
    tipo_qa = clasificar_query(query)

    if tipo_qa == "articulo":
        return "lexica"
    if tipo_qa == "definicion":
        return "semantica"
    if tipo_qa == "procedimiento":
        return "mixta"

    # Para "general" usar heurísticas adicionales
    q = query.lower()
    if any(x in q for x in [
        "obligatorio", "obligatoria", "requisito", "requisitos",
        "medida", "medidas", "como se", "trámite"
    ]):
        return "mixta"

    return "semantica"  # default
