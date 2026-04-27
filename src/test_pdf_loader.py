"""
test_pdf_loader.py
------------------
Tests del módulo pdf_loader.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pdf_loader import cargar_pdfs, limpiar_texto_normativa, extraer_metadata

RUTA_PDFS = "../data/raw/pdfs"


def test_carga_documentos():
    docs = cargar_pdfs(RUTA_PDFS)
    assert len(docs) == 10, f"Se esperaban 10 documentos, se obtuvieron {len(docs)}"
    print(f"[OK] test_carga_documentos: {len(docs)} documentos cargados")


def test_no_documentos_vacios():
    docs = cargar_pdfs(RUTA_PDFS)
    for doc in docs:
        assert len(doc["texto"]) > 100, \
            f"Texto demasiado corto en '{doc['id']}': {len(doc['texto'])} chars"
    print(f"[OK] test_no_documentos_vacios: todos los documentos tienen texto suficiente")


def test_campos_obligatorios():
    docs = cargar_pdfs(RUTA_PDFS)
    campos = ["id", "titulo", "texto", "numero_resolucion", "anio"]
    for doc in docs:
        for campo in campos:
            assert campo in doc, f"Campo '{campo}' ausente en '{doc['id']}'"
    print(f"[OK] test_campos_obligatorios: todos los campos presentes")


def test_metadata_enriquecida():
    docs = cargar_pdfs(RUTA_PDFS)
    for doc in docs:
        assert doc["numero_resolucion"] is not None, \
            f"numero_resolucion ausente en '{doc['id']}'"
        assert doc["anio"] is not None, \
            f"anio ausente en '{doc['id']}'"
        assert isinstance(doc["anio"], int), \
            f"anio no es entero en '{doc['id']}'"
    print(f"[OK] test_metadata_enriquecida: número y año extraídos correctamente")


def test_limpieza_elimina_encabezados():
    texto_crudo = "SERVICIO NACIONAL DE SANIDAD Y CALIDAD AGROALIMENTARIA\nVISTO el Expediente\nCONSIDERANDO."
    texto_limpio = limpiar_texto_normativa(texto_crudo)
    assert "SERVICIO NACIONAL" not in texto_limpio.upper(), \
        "El encabezado institucional no fue eliminado"
    print(f"[OK] test_limpieza_elimina_encabezados")


def test_limpieza_corta_desde_visto():
    texto = "Texto previo sin relevancia\nVISTO el expediente N° 123.\nCONSIDERANDO algo."
    texto_limpio = limpiar_texto_normativa(texto)
    assert texto_limpio.upper().startswith("EL EXPEDIENTE") or \
           "EXPEDIENTE" in texto_limpio.upper(), \
        f"No cortó desde VISTO correctamente. Resultado: {texto_limpio[:100]}"
    print(f"[OK] test_limpieza_corta_desde_visto")


def test_directorio_inexistente():
    try:
        cargar_pdfs("/ruta/que/no/existe")
        assert False, "Debería haber lanzado FileNotFoundError"
    except FileNotFoundError:
        print("[OK] test_directorio_inexistente: FileNotFoundError lanzado correctamente")


if __name__ == "__main__":
    print("=" * 50)
    print("TEST PDF LOADER")
    print("=" * 50)
    test_carga_documentos()
    test_no_documentos_vacios()
    test_campos_obligatorios()
    test_metadata_enriquecida()
    test_limpieza_elimina_encabezados()
    test_limpieza_corta_desde_visto()
    test_directorio_inexistente()
    print("\n✅ Todos los tests pasaron.")
