"""
test_retrieval_hybrid.py
------------------------
Tests del HybridRetriever.
Usa build_hybrid_offline() si no hay acceso a Hugging Face.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from retriever_hybrid import normalizar_scores

RUTA_PDFS = "../data/raw/pdfs"


def build_retriever(chunks, alpha=0.5):
    try:
        from retriever_hybrid import HybridRetriever
        r = HybridRetriever(alpha=alpha)
        r.fit(chunks)
        print("[INFO] Usando HybridRetriever real")
        return r
    except Exception as e:
        print(f"[INFO] Fallback a build_hybrid_offline ({type(e).__name__})")
        from retriever_hybrid import build_hybrid_offline
        return build_hybrid_offline(chunks, alpha=alpha)


def setup(alpha=0.5):
    from pdf_loader import cargar_pdfs
    from chunker import chunkear_documentos
    docs = cargar_pdfs(RUTA_PDFS)
    chunks = chunkear_documentos(docs)
    retriever = build_retriever(chunks, alpha=alpha)
    return retriever, chunks


def test_fit_inicializa_submodelos():
    retriever, chunks = setup()
    assert retriever.tfidf.matrix is not None
    assert retriever.embedder.embeddings is not None
    print("[OK] test_fit_inicializa_submodelos")


def test_buscar_devuelve_top_k():
    retriever, _ = setup()
    for k in [1, 3, 5]:
        resultados = retriever.buscar("brucelosis bovina", top_k=k)
        assert len(resultados) == k, f"Esperados {k}, obtenidos {len(resultados)}"
    print("[OK] test_buscar_devuelve_top_k")


def test_scores_en_rango():
    retriever, _ = setup()
    resultados = retriever.buscar("medidas sanitarias", top_k=5)
    for r in resultados:
        assert r["score"] >= 0.0, f"Score negativo: {r['score']}"
    print("[OK] test_scores_en_rango")


def test_ordenados_por_score():
    retriever, _ = setup()
    resultados = retriever.buscar("definicion producto biologico", top_k=5)
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


def test_penalizacion_ruido():
    retriever, _ = setup()
    resultados = retriever.buscar("resolucion senasa normativa", top_k=10)
    ruido_posiciones = [
        i for i, r in enumerate(resultados)
        if any(k in r["chunk"]["texto"].lower()
               for k in ["comuníquese", "publíquese", "archívese"])
    ]
    for pos in ruido_posiciones:
        assert pos >= len(resultados) // 2, \
            f"Chunk de ruido en posición alta: {pos}"
    print(f"[OK] test_penalizacion_ruido: posiciones ruido={ruido_posiciones}")


def test_normalizar_scores():
    arr = np.array([0.1, 0.5, 0.9])
    norm = normalizar_scores(arr)
    assert abs(norm.min()) < 1e-6
    assert abs(norm.max() - 1.0) < 1e-6

    arr_const = np.array([0.5, 0.5, 0.5])
    norm_const = normalizar_scores(arr_const)
    assert not np.any(np.isnan(norm_const))
    print("[OK] test_normalizar_scores")


def test_alpha_extremos():
    from pdf_loader import cargar_pdfs
    from chunker import chunkear_documentos
    docs = cargar_pdfs(RUTA_PDFS)
    chunks = chunkear_documentos(docs)
    for alpha in [0.0, 1.0]:
        r = build_retriever(chunks, alpha=alpha)
        resultados = r.buscar("sanidad animal", top_k=3)
        assert len(resultados) == 3, f"alpha={alpha}: esperados 3 resultados"
    print("[OK] test_alpha_extremos")


if __name__ == "__main__":
    print("=" * 50)
    print("TEST RETRIEVAL HYBRID")
    print("=" * 50)
    test_fit_inicializa_submodelos()
    test_buscar_devuelve_top_k()
    test_scores_en_rango()
    test_ordenados_por_score()
    test_busqueda_articulo_especifico()
    test_penalizacion_ruido()
    test_normalizar_scores()
    test_alpha_extremos()
    print("\n✅ Todos los tests pasaron.")
