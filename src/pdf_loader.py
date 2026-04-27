"""
pdf_loader.py
-------------
Responsabilidad: cargar y limpiar resoluciones SENASA desde archivos PDF.

Pipeline:
PDF → extracción → limpieza básica → documento estructurado

Mejoras implementadas:
- Metadata enriquecida (número de resolución, año)
- Advertencia explícita cuando el texto extraído queda vacío
  (PDF escaneado o protegido)
- Pipeline de limpieza con orden corregido
"""

import os
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def extraer_texto_pdf(ruta_pdf: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Instalar pdfplumber: pip install pdfplumber")

    if not os.path.exists(ruta_pdf):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta_pdf}")

    texto_paginas = []
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text()
            if texto is not None and texto.strip():
                texto_paginas.append(texto)

    texto_completo = "\n".join(texto_paginas)

    if len(texto_completo.strip()) < 200:
        logger.warning(
            f"[pdf_loader] Texto muy corto en '{os.path.basename(ruta_pdf)}' "
            f"({len(texto_completo)} chars). Puede ser un PDF escaneado."
        )

    return texto_completo


def limpiar_texto_normativa(texto: str) -> str:
    """
    Pipeline de limpieza ordenado correctamente:
    1. Colapsar saltos múltiples
    2. Eliminar encabezados institucionales
    3. Cortar desde VISTO
    4. Eliminar números de página aislados
    5. Normalizar espacios horizontales
    6. Unir líneas que empiezan en minúscula
    7. Unir líneas que no terminan en punto
    8. Corregir palabras pegadas
    9. Eliminar VISTO residual al inicio
    """
    # 1
    texto = re.sub(r"\n{2,}", "\n", texto)
    # 2
    texto = re.sub(r"(?i)^.*servicio nacional de sanidad.*$", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"(?i)republica argentina.*\n", "", texto)
    # 3
    match = re.search(r"VISTO.*", texto, re.IGNORECASE | re.DOTALL)
    if match:
        texto = match.group(0)
    # 4
    texto = re.sub(r"\n\s*\d+\s*\n", "\n", texto)
    # 5
    texto = re.sub(r"[ \t]+", " ", texto)
    # 6
    texto = re.sub(r"\n([a-záéíóúüñ])", r" \1", texto)
    # 7
    texto = re.sub(r"(?<![.!?])\n", " ", texto)
    # 8
    for kw in ["VISTO", "CONSIDERANDO", "RESUELVE", "ARTÍCULO"]:
        texto = re.sub(rf"({kw})([a-záéíóúüñ])", rf"\1 \2", texto)
    # 9
    texto = re.sub(r"(?i)^VISTO\s*", "", texto)

    return texto.strip()


def extraer_metadata(texto: str, nombre_archivo: str) -> Dict:
    nombre_base = os.path.basename(nombre_archivo).replace(".pdf", "")
    doc_id = nombre_base

    numero_resolucion: Optional[str] = None
    anio: Optional[int] = None

    match_nombre = re.match(r"(?i)res[_\-]?(\d+)[_\-](\d{4})", nombre_base)
    if match_nombre:
        numero_resolucion = match_nombre.group(1)
        anio = int(match_nombre.group(2))

    lineas = [l.strip() for l in texto.split("\n") if l.strip()]
    titulo = lineas[0] if lineas else doc_id

    if numero_resolucion is None:
        match_texto = re.search(r"resoluci[oó]n\s+(?:n[°º]?\s*)?(\d+)", texto, re.IGNORECASE)
        if match_texto:
            numero_resolucion = match_texto.group(1)

    return {
        "id": doc_id,
        "numero_resolucion": numero_resolucion,
        "anio": anio,
        "titulo": titulo,
    }


def cargar_pdfs(directorio: str) -> List[Dict]:
    if not os.path.exists(directorio):
        raise FileNotFoundError(f"Directorio no encontrado: {directorio}")

    documentos = []
    archivos_pdf = sorted([f for f in os.listdir(directorio) if f.endswith(".pdf")])

    if not archivos_pdf:
        logger.warning(f"[pdf_loader] No se encontraron PDFs en: {directorio}")
        return documentos

    for archivo in archivos_pdf:
        ruta = os.path.join(directorio, archivo)
        texto_crudo = extraer_texto_pdf(ruta)
        texto_limpio = limpiar_texto_normativa(texto_crudo)
        metadata = extraer_metadata(texto_limpio, archivo)

        if not texto_limpio:
            logger.warning(f"[pdf_loader] Texto vacío tras limpieza: '{archivo}'. Omitido.")
            continue

        documentos.append({
            "id":                metadata["id"],
            "numero_resolucion": metadata["numero_resolucion"],
            "anio":              metadata["anio"],
            "titulo":            metadata["titulo"],
            "texto":             texto_limpio,
        })

    return documentos
