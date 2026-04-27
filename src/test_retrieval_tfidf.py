"""
test_retrieval_tfidf.py
-----------------------
Tests del TFIDFRetriever.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pdf_loader import cargar_pdfs
from chunker import chunkear_documentos
from retriever_tfidf import TFIDFRetriever

RUTA_PDFS = "../data/raw/pdfs"


def setup():
    docs = cargar_pdfs(RUTA_PDFS)
    chunks = chunkear_documentos(docs)
    retriever = TFIDFRetriever()
    retriever.fit(chunks)
    return retriever, chunks


def test_fit_inicializa_correctamente():
    retriever, chunks = setup()
    assert retriever.vectorizer is not None
    assert retriever.matrix is not None
    assert retriever.matrix.shape[0] == len(chunks)
    print(f"[OK] test_fit_inicializa_correctamente: matrix={retriever.matrix.shape}")


def test_buscar_devuelve_top_k():
    retriever, _ = setup()
    for k in [1, 3, 5]:
        resultados = retriever.buscar("brucelosis bovina", top_k=k)
        assert len(resultados) == k, f"Se esperaban {k} resultados, se obtuvieron {len(resultados)}"
    print("[OK] test_buscar_devuelve_top_k")


def test_scores_entre_0_y_1():
    retriever, _ = setup()
    resultados = retriever.buscar("medidas sanitarias", top_k=5)
    for r in resultados:
        assert 0.0 <= r["score"] <= 1.01, f"Score fuera de rango: {r['score']}"
    print("[OK] test_scores_entre_0_y_1")


def test_ordenados_por_score():
    retriever, _ = setup()
    resultados = retriever.buscar("producto veterinario definicion", top_k=5)
    scores = [r["score"] for r in resultados]
    assert scores == sorted(scores, reverse=True), "Resultados no ordenados por score"
    print("[OK] test_ordenados_por_score")


def test_busqueda_articulo_especifico():
    retriever, _ = setup()
    resultados = retriever.buscar("articulo 3", top_k=5)
    assert len(resultados) > 0, "Sin resultados para 'articulo 3'"
    for r in resultados:
        assert r["chunk"].get("article_number") == 3, \
            f"Se esperaba artículo 3, se obtuvo: {r['chunk'].get('article_number')}"
    print(f"[OK] test_busqueda_articulo_especifico: {len(resultados)} chunks del artículo 3")


def test_sin_entrenar_lanza_error():
    retriever = TFIDFRetriever()
    try:
        retriever.buscar("test")
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        print("[OK] test_sin_entrenar_lanza_error")


def test_cada_resultado_tiene_chunk():
    retriever, _ = setup()
    resultados = retriever.buscar("requisitos productores", top_k=5)
    for r in resultados:
        assert "chunk" in r
        assert "texto" in r["chunk"]
        assert len(r["chunk"]["texto"]) > 0
    print("[OK] test_cada_resultado_tiene_chunk")


if __name__ == "__main__":
    print("=" * 50)
    print("TEST RETRIEVAL TF-IDF")
    print("=" * 50)
    test_fit_inicializa_correctamente()
    test_buscar_devuelve_top_k()
    test_scores_entre_0_y_1()
    test_ordenados_por_score()
    test_busqueda_articulo_especifico()
    test_sin_entrenar_lanza_error()
    test_cada_resultado_tiene_chunk()
    print("\n✅ Todos los tests pasaron.")
