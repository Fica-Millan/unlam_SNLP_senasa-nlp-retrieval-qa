"""
test_chunking_quality.py
------------------------
Tests de calidad del chunker.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
import numpy as np
from collections import Counter
from pdf_loader import cargar_pdfs
from chunker import chunkear_documentos

RUTA_PDFS = "../data/raw/pdfs"


def cargar_chunks():
    docs = cargar_pdfs(RUTA_PDFS)
    return chunkear_documentos(docs)


def test_chunks_generados():
    chunks = cargar_chunks()
    assert len(chunks) > 0, "No se generaron chunks"
    assert len(chunks) >= 50, f"Muy pocos chunks: {len(chunks)}"
    print(f"[OK] test_chunks_generados: {len(chunks)} chunks")


def test_longitud_minima():
    chunks = cargar_chunks()
    cortos = [c for c in chunks if len(c["texto"].split()) < 15]
    assert len(cortos) == 0, \
        f"{len(cortos)} chunks tienen menos de 15 palabras"
    print(f"[OK] test_longitud_minima: ningún chunk menor a 15 palabras")


def test_sin_duplicados_exactos():
    chunks = cargar_chunks()
    textos = [c["texto"] for c in chunks]
    unique = set(textos)
    dup_pct = (1 - len(unique) / len(textos)) * 100
    assert dup_pct < 5.0, \
        f"Duplicación excesiva: {dup_pct:.1f}%"
    print(f"[OK] test_sin_duplicados_exactos: duplicación={dup_pct:.1f}%")


def test_campos_obligatorios():
    chunks = cargar_chunks()
    campos = ["doc_id", "chunk_id", "titulo", "seccion", "texto",
              "es_definicion", "es_obligacion"]
    for c in chunks[:10]:
        for campo in campos:
            assert campo in c, f"Campo '{campo}' ausente en chunk '{c.get('chunk_id')}'"
    print("[OK] test_campos_obligatorios: todos los campos presentes")


def test_flags_semanticos_son_bool():
    chunks = cargar_chunks()
    for c in chunks:
        for flag in ["es_definicion", "es_obligacion", "es_procedimiento", "es_sancion"]:
            assert isinstance(c[flag], bool), \
                f"Flag '{flag}' no es bool en chunk '{c['chunk_id']}'"
    print("[OK] test_flags_semanticos_son_bool")


def test_secciones_conocidas():
    chunks = cargar_chunks()
    secciones = set(c["seccion"] for c in chunks)
    # Debe haber al menos CONSIDERANDO y algún ARTÍCULO
    assert any("CONSIDERANDO" in s for s in secciones), "Sin chunks CONSIDERANDO"
    assert any("ARTÍCULO" in s for s in secciones), "Sin chunks ARTÍCULO"
    print(f"[OK] test_secciones_conocidas: {len(secciones)} secciones distintas")


def test_article_number_consistente():
    chunks = cargar_chunks()
    for c in chunks:
        seccion = c.get("seccion", "")
        num = c.get("article_number")
        if "ARTÍCULO" in seccion:
            assert num is not None, \
                f"article_number es None en sección '{seccion}' (chunk '{c['chunk_id']}')"
            match = re.search(r"ARTÍCULO\s+(\d+)", seccion)
            if match:
                assert num == int(match.group(1)), \
                    f"article_number={num} no coincide con '{seccion}'"
    print("[OK] test_article_number_consistente")


def test_sin_mezcla_de_secciones():
    """Ningún chunk debe contener múltiples encabezados normativos."""
    chunks = cargar_chunks()
    patrones = ["VISTO", "CONSIDERANDO", "RESUELVE"]
    mezclados = []
    for c in chunks:
        encontrados = [p for p in patrones if p in c["texto"].upper()]
        if len(encontrados) > 1:
            mezclados.append(c["chunk_id"])
    pct = len(mezclados) / len(chunks) * 100
    assert pct < 5.0, \
        f"{pct:.1f}% de chunks mezclan secciones: {mezclados[:3]}"
    print(f"[OK] test_sin_mezcla_de_secciones: mezcla={pct:.1f}%")


def test_distribucion_tamanios():
    chunks = cargar_chunks()
    lengths = [len(c["texto"].split()) for c in chunks]
    promedio = np.mean(lengths)
    mediana = np.median(lengths)
    print(f"[INFO] Distribución tamaños: min={min(lengths)}, max={max(lengths)}, "
          f"prom={promedio:.1f}, mediana={mediana}")
    assert promedio > 20, f"Promedio muy bajo: {promedio:.1f}"
    assert max(lengths) <= 130, f"Chunks demasiado largos: max={max(lengths)}"
    print("[OK] test_distribucion_tamanios")


if __name__ == "__main__":
    print("=" * 50)
    print("TEST CHUNKING QUALITY")
    print("=" * 50)
    test_chunks_generados()
    test_longitud_minima()
    test_sin_duplicados_exactos()
    test_campos_obligatorios()
    test_flags_semanticos_son_bool()
    test_secciones_conocidas()
    test_article_number_consistente()
    test_sin_mezcla_de_secciones()
    test_distribucion_tamanios()
    print("\n✅ Todos los tests pasaron.")
