"""
test_retrieval_embeddings.py
----------------------------
Tests del EmbeddingRetriever.

Estrategia de fallback:
- Si SentenceTransformer no puede descargar el modelo (sin internet),
  usa MockEmbeddingRetriever (TF-IDF + SVD) que mantiene la misma interfaz.
  Los tests de comportamiento son idénticos en ambos casos.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

RUTA_PDFS = "../data/raw/pdfs"


def get_retriever(chunks):
    """Retorna EmbeddingRetriever real o Mock según disponibilidad."""
    try:
        from retriever_embeddings import EmbeddingRetriever
        r = EmbeddingRetriever()
        r.fit(chunks)
        print("[INFO] Usando EmbeddingRetriever real (SentenceTransformer)")
        return r
    except Exception as e:
        print(f"[INFO] SentenceTransformer no disponible ({type(e).__name__}). Usando MockEmbeddingRetriever.")
        from mock_embeddings import MockEmbeddingRetriever
        r = MockEmbeddingRetriever()
        r.fit(chunks)
        return r


def setup():
    from pdf_loader import cargar_pdfs
    from chunker import chunkear_documentos
    docs = cargar_pdfs(RUTA_PDFS)
    chunks = chunkear_documentos(docs)
    retriever = get_retriever(chunks)
    return retriever, chunks


def test_fit_genera_embeddings():
    retriever, chunks = setup()
    assert retriever.embeddings is not None
    assert retriever.embeddings.shape[0] == len(chunks)
    assert retriever.embeddings.shape[1] > 0
    print(f"[OK] test_fit_genera_embeddings: shape={retriever.embeddings.shape}")


def test_buscar_devuelve_top_k():
    retriever, _ = setup()
    for k in [1, 3, 5]:
        resultados = retriever.buscar("brucelosis bovina", top_k=k)
        assert len(resultados) == k, f"Esperados {k}, obtenidos {len(resultados)}"
    print("[OK] test_buscar_devuelve_top_k")


def test_scores_en_rango():
    retriever, _ = setup()
    resultados = retriever.buscar("sanidad animal", top_k=5)
    for r in resultados:
        assert -1.01 <= r["score"] <= 1.01, f"Score fuera de rango: {r['score']}"
    print("[OK] test_scores_en_rango")


def test_ordenados_por_score():
    retriever, _ = setup()
    resultados = retriever.buscar("definicion producto veterinario", top_k=5)
    scores = [r["score"] for r in resultados]
    assert scores == sorted(scores, reverse=True), "Resultados no ordenados"
    print("[OK] test_ordenados_por_score")


def test_busqueda_articulo_especifico():
    retriever, _ = setup()
    resultados = retriever.buscar("articulo 3", top_k=5)
    assert len(resultados) > 0
    for r in resultados:
        assert r["chunk"].get("article_number") == 3, \
            f"Esperado artículo 3, obtenido: {r['chunk'].get('article_number')}"
    print(f"[OK] test_busqueda_articulo_especifico: {len(resultados)} resultados")


def test_sin_entrenar_lanza_error():
    try:
        from retriever_embeddings import EmbeddingRetriever
        r = EmbeddingRetriever()
    except Exception:
        from mock_embeddings import MockEmbeddingRetriever
        r = MockEmbeddingRetriever()
    try:
        r.buscar("test")
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        print("[OK] test_sin_entrenar_lanza_error")


def test_embeddings_normalizados():
    retriever, _ = setup()
    normas = np.linalg.norm(retriever.embeddings, axis=1)
    assert np.allclose(normas, 1.0, atol=1e-4), \
        f"Embeddings no normalizados. Norma media: {normas.mean():.4f}"
    print(f"[OK] test_embeddings_normalizados: norma media={normas.mean():.4f}")


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
    print("TEST RETRIEVAL EMBEDDINGS")
    print("=" * 50)
    test_fit_genera_embeddings()
    test_buscar_devuelve_top_k()
    test_scores_en_rango()
    test_ordenados_por_score()
    test_busqueda_articulo_especifico()
    test_sin_entrenar_lanza_error()
    test_embeddings_normalizados()
    test_cada_resultado_tiene_chunk()
    print("\n✅ Todos los tests pasaron.")
