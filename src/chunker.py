"""
chunker.py
----------
Responsabilidad: dividir documentos en chunks para recuperación eficiente.

Estrategia:
- División primaria por secciones normativas (VISTO, CONSIDERANDO, RESUELVE, ARTÍCULO N)
- CONSIDERANDO subdividido por cláusulas "Que"
- Chunks largos subdivididos por palabras con solapamiento
- Clasificación semántica centralizada en utils.py (sin duplicación)
- Metadata enriquecida: tipo de sección, número de artículo, posición relativa

Filtros de calidad:
- Mínimo 15 palabras
- Debe terminar en oración completa (solo para chunks directos, no subchunks finales)
"""

import re
from typing import List, Dict

from utils import clasificar_chunk


# =========================================================
# 🟢 DIVISIÓN POR SECCIONES NORMATIVAS
# =========================================================

def dividir_por_secciones(texto: str) -> List[str]:
    """
    Divide el texto según las secciones típicas de normativa SENASA.
    """
    patron = r"(VISTO|CONSIDERANDO|RESUELVE|ART[IÍ]CULO\s+\d+)"
    partes = re.split(patron, texto, flags=re.IGNORECASE)

    secciones = []
    for i in range(1, len(partes), 2):
        titulo = partes[i]
        contenido = partes[i + 1] if i + 1 < len(partes) else ""
        secciones.append(f"{titulo} {contenido}".strip())

    return secciones


def dividir_considerando(texto: str) -> List[str]:
    """
    Divide la sección CONSIDERANDO en subbloques por cláusula 'Que'.
    Solo devuelve fragmentos con contenido suficiente.
    """
    partes = re.split(r"(?<=\.)\s+(?=Que\b)", texto)

    chunks = []
    for p in partes:
        p = p.strip()
        if len(p) > 50 and not re.match(r"^\s*(considerando|visto)", p, re.IGNORECASE):
            p = re.sub(r"^(Que\s+)+", "", p, flags=re.IGNORECASE)
            chunks.append(f"CONSIDERANDO: Que {p}")

    return chunks


# =========================================================
# 🟢  SUBDIVISIÓN EN SUBCHUNKS
# =========================================================

def termina_en_oracion(texto: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", texto.strip()))


def generar_chunks_parrafo(parrafo: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Divide un párrafo largo en subchunks por palabras con solapamiento.
    """
    palabras = parrafo.split()
    chunks = []
    start = 0

    while start < len(palabras):
        end = start + chunk_size
        chunks.append(" ".join(palabras[start:end]))
        start += max(1, chunk_size - overlap)

    return chunks


# =========================================================
# 🟢 METADATA DE SECCIÓN
# =========================================================

def extraer_metadata_seccion(seccion: str) -> Dict:
    """
    Extrae el tipo de sección y el número de artículo (si aplica).
    """
    match = re.match(r"(VISTO|CONSIDERANDO|RESUELVE|ART[IÍ]CULO\s+\d+)", seccion, re.IGNORECASE)
    titulo_seccion = match.group(1).upper() if match else "OTRA"

    article_number = None
    art_match = re.match(r"ART[IÍ]CULO\s+(\d+)", titulo_seccion)
    if art_match:
        article_number = int(art_match.group(1))

    return {"titulo_seccion": titulo_seccion, "article_number": article_number}


# =========================================================
# 🟢 PIPELINE PRINCIPAL
# =========================================================

def chunkear_documentos(documentos: List[Dict], chunk_size: int = 120, overlap: int = 20) -> List[Dict]:
    """
    Genera chunks a partir de una lista de documentos.

    Cada chunk incluye:
    - doc_id, chunk_id, titulo, seccion, article_number
    - texto
    - flags semánticos: es_definicion, es_obligacion, es_procedimiento, es_sancion
    """
    chunks = []

    for doc in documentos:
        secciones = dividir_por_secciones(doc["texto"])

        for i, seccion in enumerate(secciones):
            meta = extraer_metadata_seccion(seccion)
            titulo_seccion = meta["titulo_seccion"]
            article_number = meta["article_number"]

            # Subdividir CONSIDERANDO por cláusulas Que
            if "CONSIDERANDO" in titulo_seccion:
                sub_secciones = dividir_considerando(seccion) or [seccion]
            else:
                sub_secciones = [seccion]

            for k, sub in enumerate(sub_secciones):
                palabras = sub.split()

                # Filtro mínimo de longitud
                if len(palabras) < 15:
                    continue

                # --- Chunk directo (no necesita subdivisión) ---
                if len(palabras) <= chunk_size:
                    if not termina_en_oracion(sub):
                        continue

                    flags = clasificar_chunk(sub)
                    chunks.append({
                        "doc_id":         doc["id"],
                        "chunk_id":       f"{doc['id']}_sec{i}_sub{k}",
                        "titulo":         doc["titulo"],
                        "seccion":        titulo_seccion,
                        "article_number": article_number,
                        "subseccion":     k,
                        "texto":          sub,
                        **flags,
                    })

                # --- Subchunks por palabras ---
                else:
                    subchunks = generar_chunks_parrafo(sub, chunk_size, overlap)

                    for j, sc in enumerate(subchunks):
                        palabras_sc = sc.split()

                        if len(palabras_sc) < 15:
                            continue

                        # Solo el último subchunk puede quedar incompleto;
                        # los intermedios siempre terminan en corte de palabras.
                        # Aplicamos el filtro de oración solo al último.
                        es_ultimo = (j == len(subchunks) - 1)
                        if es_ultimo and not termina_en_oracion(sc):
                            continue

                        flags = clasificar_chunk(sc)
                        chunks.append({
                            "doc_id":         doc["id"],
                            "chunk_id":       f"{doc['id']}_sec{i}_sub{k}_c{j}",
                            "titulo":         doc["titulo"],
                            "seccion":        titulo_seccion,
                            "article_number": article_number,
                            "subseccion":     k,
                            "chunk_num":      j,
                            "texto":          sc,
                            **flags,
                        })

    return chunks
