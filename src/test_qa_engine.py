"""
test_qa_engine.py
-----------------
Tests del módulo qa_engine.py.

Estrategia:
- Tests unitarios de funciones internas (clasificación, extracción).
- Tests de integración end-to-end: retriever → QA → respuesta.
- Usa TFIDFRetriever para tests rápidos (sin descarga de modelo).
- build_hybrid() reutiliza sub-modelos ya entrenados (consistente
  con el resto del proyecto).

Nota metodológica:
    El QA implementado es extractivo/sintético heurístico, sin modelo
    generativo. Esto garantiza trazabilidad total de la información
    (cada respuesta apunta a una fuente exacta) pero limita la fluidez
    y capacidad de abstracción respecto a un LLM.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pdf_loader import cargar_pdfs
from chunker import chunkear_documentos
from retriever_tfidf import TFIDFRetriever
from retriever_hybrid import build_hybrid
from utils import clasificar_query, extraer_numero_articulo
from qa_engine import (
    responder,
    qa_extractivo,
    qa_sintetico,
    extraer_oracion_relevante,
    _limpiar_prefijo,
)

RUTA_PDFS = "../data/raw/pdfs"


# =========================================================
# 🟢 SETUP (compartido entre tests)
# =========================================================

def build_pipeline():
    docs = cargar_pdfs(RUTA_PDFS)
    chunks = chunkear_documentos(docs)
    tfidf = TFIDFRetriever()
    tfidf.fit(chunks)
    return tfidf, chunks


# =========================================================
# 🟢 TESTS UNITARIOS
# =========================================================

def test_clasificar_query():
    """Verifica que clasificar_query está importado de utils (no reimplementado)."""
    assert clasificar_query("qué es la brucelosis") == "definicion"
    assert clasificar_query("que dice el articulo 3") == "articulo"
    assert clasificar_query("cómo se debe vacunar") == "procedimiento"
    assert clasificar_query("brucelosis bovina") == "general"
    print("[OK] test_clasificar_query")


def test_extraer_oracion_relevante():
    """La oración devuelta debe contener alguna palabra de la query."""
    texto = (
        "La brucelosis bovina es una enfermedad infecciosa causada por Brucella abortus. "
        "Afecta principalmente al ganado bovino. "
        "Puede transmitirse a los seres humanos."
    )
    query = "qué es la brucelosis bovina"
    oracion = extraer_oracion_relevante(texto, query)
    assert len(oracion) > 0, "No devolvió ninguna oración"
    assert any(w in oracion.lower() for w in ["brucelosis", "bovina", "enfermedad"]), \
        f"Oración irrelevante: {oracion}"
    print(f"[OK] test_extraer_oracion_relevante: '{oracion[:80]}...'")


def test_extraer_oracion_texto_vacio():
    """Con texto vacío debe devolver string vacío sin explotar."""
    assert extraer_oracion_relevante("", "query") == ""
    print("[OK] test_extraer_oracion_texto_vacio")


def test_limpiar_prefijo():
    """Verifica eliminación de prefijos normativos estructurales."""
    casos = [
        ("CONSIDERANDO: Que la ley establece...",     "la ley establece..."),
        ("CONSIDERANDO: Que Que la brucelosis...",     "la brucelosis..."),
        ("ARTÍCULO 3.- Los establecimientos...",       "Los establecimientos..."),
        ("texto sin prefijo",                          "texto sin prefijo"),
    ]
    for entrada, esperado in casos:
        resultado = _limpiar_prefijo(entrada)
        assert resultado == esperado, \
            f"Entrada: '{entrada}'\nEsperado: '{esperado}'\nObtenido: '{resultado}'"
    print("[OK] test_limpiar_prefijo")


def test_responder_modo_invalido():
    """Modo no válido debe lanzar ValueError."""
    try:
        responder([], "query", modo="gpt4")
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass
    print("[OK] test_responder_modo_invalido")


def test_responder_sin_resultados():
    """Con lista vacía debe devolver mensaje de error, no explotar."""
    for modo in ["extractivo", "sintetico"]:
        resp = responder([], "query test", modo=modo)
        assert "respuesta" in resp
        assert len(resp["respuesta"]) > 0
        assert resp["metodo"] == modo
    print("[OK] test_responder_sin_resultados")


# =========================================================
# 🟢 TESTS DE INTEGRACIÓN (retriever → QA)
# =========================================================

def test_qa_extractivo_devuelve_campos():
    """El QA extractivo debe incluir todos los campos requeridos."""
    tfidf, _ = build_pipeline()
    resultados = tfidf.buscar("brucelosis bovina", top_k=5)
    resp = qa_extractivo(resultados, "qué es la brucelosis bovina")

    for campo in ["respuesta", "fuente", "score", "metodo", "pasaje_completo", "seccion"]:
        assert campo in resp, f"Campo '{campo}' ausente"
    assert resp["metodo"] == "extractivo"
    assert resp["score"] > 0
    assert len(resp["respuesta"]) > 0
    print(f"[OK] test_qa_extractivo_devuelve_campos: '{resp['respuesta'][:80]}...'")


def test_qa_sintetico_devuelve_campos():
    """El QA sintético debe incluir todos los campos requeridos."""
    tfidf, _ = build_pipeline()
    resultados = tfidf.buscar("brucelosis bovina", top_k=5)
    resp = qa_sintetico(resultados, "qué es la brucelosis bovina", top_n=3)

    for campo in ["respuesta", "fuentes", "metodo", "tipo_query", "n_chunks_usados"]:
        assert campo in resp, f"Campo '{campo}' ausente"
    assert resp["metodo"] == "sintetico"
    assert isinstance(resp["fuentes"], list)
    assert len(resp["respuesta"]) > 0
    print(f"[OK] test_qa_sintetico_devuelve_campos: '{resp['respuesta'][:80]}...'")


def test_qa_tipo_detectado_correctamente():
    """El tipo detectado en la respuesta sintética debe coincidir con utils."""
    tfidf, _ = build_pipeline()
    casos = [
        ("qué es la brucelosis bovina",   "definicion"),
        ("cómo se deben vacunar las terneras", "procedimiento"),
    ]
    for query, tipo_esperado in casos:
        resultados = tfidf.buscar(query, top_k=5)
        resp = qa_sintetico(resultados, query)
        assert resp["tipo_query"] == tipo_esperado, \
            f"Query '{query}': esperado '{tipo_esperado}', obtenido '{resp['tipo_query']}'"
    print("[OK] test_qa_tipo_detectado_correctamente")


def test_qa_articulo_usa_numero_correcto():
    """Para queries de artículo, el QA debe usar los chunks de ese artículo."""
    tfidf, chunks = build_pipeline()
    query = "que dice el articulo 3"
    resultados = tfidf.buscar(query, top_k=5)
    resp = qa_sintetico(resultados, query, top_n=3)

    # Si encontró fuentes, deben ser del artículo 3
    for fuente in resp["fuentes"]:
        seccion = fuente.get("seccion", "")
        assert "ARTÍCULO 3" in seccion or seccion == "?", \
            f"Fuente de artículo incorrecto: {seccion}"
    print(f"[OK] test_qa_articulo_usa_numero_correcto: {len(resp['fuentes'])} fuentes")


def test_qa_sintetico_no_devuelve_ruido():
    """La respuesta sintética no debe contener frases administrativas de cierre."""
    tfidf, _ = build_pipeline()
    queries = [
        "brucelosis bovina que es",
        "medidas sanitarias obligatorias",
        "como se registra un producto veterinario",
    ]
    ruido_keywords = ["comuníquese", "publíquese", "archívese", "boletín oficial"]
    for query in queries:
        resultados = tfidf.buscar(query, top_k=5)
        resp = qa_sintetico(resultados, query)
        for kw in ruido_keywords:
            assert kw not in resp["respuesta"].lower(), \
                f"Ruido en respuesta para '{query}': encontrado '{kw}'"
    print("[OK] test_qa_sintetico_no_devuelve_ruido")


def test_qa_fuentes_tienen_campos():
    """Cada fuente en la respuesta sintética debe tener doc_id, seccion y score."""
    tfidf, _ = build_pipeline()
    resultados = tfidf.buscar("definicion producto veterinario", top_k=5)
    resp = qa_sintetico(resultados, "definicion producto veterinario", top_n=3)
    for fuente in resp["fuentes"]:
        for campo in ["doc_id", "seccion", "score"]:
            assert campo in fuente, f"Campo '{campo}' ausente en fuente: {fuente}"
        assert isinstance(fuente["score"], float)
    print(f"[OK] test_qa_fuentes_tienen_campos: {len(resp['fuentes'])} fuentes validadas")


def test_qa_end_to_end_queries_curadas():
    """Verifica que el pipeline completo produce respuestas no vacías en queries reales."""
    tfidf, _ = build_pipeline()
    queries = [
        "qué es la brucelosis bovina",
        "definicion animal positivo",
        "como se deben identificar los animales",
        "medidas sanitarias obligatorias",
        "que riesgos tiene la brucelosis",
    ]
    for query in queries:
        resultados = tfidf.buscar(query, top_k=5)
        for modo in ["extractivo", "sintetico"]:
            resp = responder(resultados, query, modo=modo)
            assert len(resp["respuesta"]) > 10, \
                f"Respuesta vacía para '{query}' en modo '{modo}'"
    print(f"[OK] test_qa_end_to_end_queries_curadas: {len(queries)} queries × 2 modos")


def test_qa_con_hybrid():
    """Verifica que el QA funciona correctamente con el HybridRetriever."""
    docs = cargar_pdfs(RUTA_PDFS)
    chunks = chunkear_documentos(docs)

    tfidf = TFIDFRetriever()
    tfidf.fit(chunks)

    try:
        from retriever_embeddings import EmbeddingRetriever
        embedder = EmbeddingRetriever()
        embedder.fit(chunks)
    except Exception:
        from mock_embeddings import MockEmbeddingRetriever
        embedder = MockEmbeddingRetriever()
        embedder.fit(chunks)

    hybrid = build_hybrid(chunks, tfidf=tfidf, embedder=embedder, alpha=0.6)
    resultados = hybrid.buscar("brucelosis bovina que es", top_k=5)
    resp = responder(resultados, "brucelosis bovina que es", modo="sintetico")

    assert len(resp["respuesta"]) > 10
    assert resp["metodo"] == "sintetico"
    print(f"[OK] test_qa_con_hybrid: '{resp['respuesta'][:80]}...'")


# =========================================================
# 🟢 DEMO (solo cuando se ejecuta directamente)
# =========================================================

def demo_respuestas():
    """Muestra respuestas comparativas extractivo vs sintético."""
    tfidf, _ = build_pipeline()
    queries = [
        "qué es la brucelosis bovina",
        "definicion animal positivo",
        "que dice el articulo 3",
        "cómo se debe vacunar a las terneras",
        "que riesgos tiene la brucelosis para las personas",
        "medidas sanitarias obligatorias",
    ]
    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print(f"TIPO:  {clasificar_query(query)}")
        resultados = tfidf.buscar(query, top_k=5)

        print("\n--- TOP 3 CHUNKS ---")
        for r in resultados[:3]:
            print(f"  Score={r['score']:.4f} | {r['chunk'].get('seccion')}")
            print(f"  {r['chunk']['texto'][:150]}")

        for modo in ["extractivo", "sintetico"]:
            resp = responder(resultados, query, modo=modo)
            print(f"\n[{modo.upper()}]")
            print(f"  Respuesta: {resp['respuesta']}")
            if "fuentes" in resp:
                for f in resp["fuentes"][:2]:
                    print(f"  Fuente: {f['doc_id']} | {f['seccion']} | score={f['score']}")
            elif "fuente" in resp:
                print(f"  Fuente: {resp['fuente']} | {resp.get('seccion','')} | score={resp['score']}")


# =========================================================
# 🟢 MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST QA ENGINE")
    print("=" * 60)

    # Tests unitarios
    test_clasificar_query()
    test_extraer_oracion_relevante()
    test_extraer_oracion_texto_vacio()
    test_limpiar_prefijo()
    test_responder_modo_invalido()
    test_responder_sin_resultados()

    # Tests de integración
    test_qa_extractivo_devuelve_campos()
    test_qa_sintetico_devuelve_campos()
    test_qa_tipo_detectado_correctamente()
    test_qa_articulo_usa_numero_correcto()
    test_qa_sintetico_no_devuelve_ruido()
    test_qa_fuentes_tienen_campos()
    test_qa_end_to_end_queries_curadas()
    test_qa_con_hybrid()

    print("\n✅ Todos los tests pasaron.")
    print()

    # Demo comparativa (comentar si solo se quieren los tests)
    demo_respuestas()
