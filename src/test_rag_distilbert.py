"""
test_rag_distilbert.py
----------------------
Tests del pipeline RAG con FAISS + DistilBERT.

Verifica:
- Construcción del índice FAISS
- Pipeline RAG completo (recuperación + generación)
- Contrato de interfaz (campos de respuesta)
- Comportamiento con modelo no disponible (fallback)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pdf_loader import cargar_pdfs
from chunker import chunkear_documentos
from retriever_embeddings import EmbeddingRetriever
from rag_distilbert import RAGDistilBERT, MODELO_DISTILBERT_ES

RUTA_PDFS = "../data/raw/pdfs"

CAMPOS_REQUERIDOS = [
    "respuesta", "score_qa", "contexto",
    "chunks_usados", "fuentes", "metodo", "modelo",
]


def setup():
    docs   = cargar_pdfs(RUTA_PDFS)
    chunks = chunkear_documentos(docs)
    retriever = EmbeddingRetriever()
    retriever.fit(chunks)
    return retriever, chunks


# =========================================================
# 🟢 TESTS — FAISS
# =========================================================

def test_faiss_index_construido():
    """El índice FAISS debe construirse con el número correcto de vectores."""
    retriever, chunks = setup()
    assert retriever.index is not None, "Índice FAISS no construido"
    assert retriever.index.ntotal == len(chunks), \
        f"FAISS tiene {retriever.index.ntotal} vectores, esperados {len(chunks)}"
    print(f"[OK] test_faiss_index_construido: {retriever.index.ntotal} vectores, dim={retriever._dim}")


def test_faiss_dimension_correcta():
    """La dimensión del índice debe coincidir con los embeddings."""
    retriever, _ = setup()
    assert retriever._dim == 384, f"Dimensión esperada 384, obtenida {retriever._dim}"
    assert retriever.embeddings.shape[1] == 384
    print(f"[OK] test_faiss_dimension_correcta: dim={retriever._dim}")


def test_faiss_embeddings_normalizados():
    """Los embeddings deben tener norma ≈ 1 (requerido para IP ≡ coseno)."""
    retriever, _ = setup()
    normas = np.linalg.norm(retriever.embeddings, axis=1)
    assert np.allclose(normas, 1.0, atol=1e-4), \
        f"Embeddings no normalizados. Norma media: {normas.mean():.4f}"
    print(f"[OK] test_faiss_embeddings_normalizados: norma media={normas.mean():.4f}")


def test_faiss_scores_entre_0_y_1():
    """Los scores FAISS (producto interno sobre vectores normalizados) deben estar en [0,1]."""
    retriever, _ = setup()
    resultados = retriever.buscar("brucelosis bovina", top_k=5)
    for r in resultados:
        assert -0.01 <= r["score"] <= 1.01, f"Score fuera de rango: {r['score']}"
    print(f"[OK] test_faiss_scores_entre_0_y_1: {[round(r['score'],3) for r in resultados]}")


def test_faiss_ordenado_por_score():
    """Los resultados deben estar ordenados de mayor a menor score."""
    retriever, _ = setup()
    resultados = retriever.buscar("definicion producto veterinario", top_k=5)
    scores = [r["score"] for r in resultados]
    assert scores == sorted(scores, reverse=True), "Resultados FAISS no ordenados"
    print(f"[OK] test_faiss_ordenado_por_score")


def test_faiss_top_k():
    """FAISS debe devolver exactamente top_k resultados."""
    retriever, _ = setup()
    for k in [1, 3, 5]:
        resultados = retriever.buscar("sanidad animal", top_k=k)
        assert len(resultados) == k, f"Esperados {k}, obtenidos {len(resultados)}"
    print("[OK] test_faiss_top_k")


def test_faiss_articulo_bypass():
    """Queries con artículo deben usar bypass, no FAISS."""
    retriever, _ = setup()
    resultados = retriever.buscar("articulo 3", top_k=5)
    assert len(resultados) > 0
    for r in resultados:
        assert r["chunk"].get("article_number") == 3, \
            f"Esperado artículo 3, obtenido {r['chunk'].get('article_number')}"
    print(f"[OK] test_faiss_articulo_bypass: {len(resultados)} chunks del artículo 3")


# =========================================================
# 🟢 TESTS — RAG DistilBERT
# =========================================================

def test_rag_devuelve_campos_requeridos():
    """La respuesta RAG debe incluir todos los campos del contrato."""
    retriever, _ = setup()
    rag = RAGDistilBERT(retriever)
    resultado = rag.preguntar("qué es la brucelosis bovina")
    for campo in CAMPOS_REQUERIDOS:
        assert campo in resultado, f"Campo '{campo}' ausente"
    assert resultado["metodo"] == "rag_distilbert"
    assert isinstance(resultado["score_qa"], float)
    assert isinstance(resultado["fuentes"], list)
    assert isinstance(resultado["chunks_usados"], list)
    print("[OK] test_rag_devuelve_campos_requeridos")


def test_rag_respuesta_no_vacia():
    """El RAG debe devolver una respuesta no vacía."""
    retriever, _ = setup()
    rag = RAGDistilBERT(retriever)
    resultado = rag.preguntar("qué es la brucelosis bovina")
    assert len(resultado["respuesta"]) > 5, \
        f"Respuesta vacía: '{resultado['respuesta']}'"
    print(f"[OK] test_rag_respuesta_no_vacia: '{resultado['respuesta'][:80]}...'")


def test_rag_contexto_contiene_chunks():
    """El contexto debe contener texto de los chunks recuperados."""
    retriever, _ = setup()
    rag = RAGDistilBERT(retriever)
    resultado = rag.preguntar("brucelosis bovina")
    assert len(resultado["contexto"]) > 50
    assert len(resultado["chunks_usados"]) > 0
    print(f"[OK] test_rag_contexto_contiene_chunks: {len(resultado['contexto'])} chars")


def test_rag_fuentes_tienen_campos():
    """Cada fuente debe tener doc_id, seccion y score."""
    retriever, _ = setup()
    rag = RAGDistilBERT(retriever)
    resultado = rag.preguntar("medidas sanitarias obligatorias")
    for f in resultado["fuentes"]:
        for campo in ["doc_id", "seccion", "score"]:
            assert campo in f, f"Campo '{campo}' ausente en fuente"
    print(f"[OK] test_rag_fuentes_tienen_campos: {len(resultado['fuentes'])} fuentes")


def test_rag_sin_resultados_fallback():
    """Con retriever que devuelve vacío, RAG debe manejar el caso sin explotar."""
    from unittest.mock import MagicMock
    retriever_mock = MagicMock()
    retriever_mock.buscar.return_value = []
    rag = RAGDistilBERT(retriever_mock)
    resultado = rag.preguntar("query imposible")
    assert "respuesta" in resultado
    assert resultado["score_qa"] == 0.0
    assert resultado["fuentes"] == []
    print("[OK] test_rag_sin_resultados_fallback")


def test_rag_contexto_truncado():
    """El contexto no debe superar MAX_CONTEXTO_CHARS."""
    from rag_distilbert import MAX_CONTEXTO_CHARS
    retriever, _ = setup()
    rag = RAGDistilBERT(retriever)
    resultado = rag.preguntar("brucelosis bovina normativa sanidad animal", top_k=10)
    assert len(resultado["contexto"]) <= MAX_CONTEXTO_CHARS + 10, \
        f"Contexto demasiado largo: {len(resultado['contexto'])} chars"
    print(f"[OK] test_rag_contexto_truncado: {len(resultado['contexto'])} chars ≤ {MAX_CONTEXTO_CHARS}")


def test_rag_queries_curadas():
    """Pipeline RAG completo sobre queries representativas del dominio."""
    retriever, _ = setup()
    rag = RAGDistilBERT(retriever)
    queries = [
        "qué es la brucelosis bovina",
        "definicion animal positivo",
        "medidas sanitarias obligatorias",
        "como se registra un producto veterinario",
        "que dice el articulo 3",
    ]
    for query in queries:
        resultado = rag.preguntar(query)
        assert len(resultado["respuesta"]) > 5, \
            f"Respuesta vacía para: '{query}'"
    print(f"[OK] test_rag_queries_curadas: {len(queries)} queries OK")


# =========================================================
# 🟢 DEMO (solo cuando se ejecuta directamente)
# =========================================================

def demo_comparacion():
    """Muestra FAISS + DistilBERT vs TF-IDF para las mismas queries."""
    from retriever_tfidf import TFIDFRetriever
    from qa_engine import responder

    docs   = cargar_pdfs(RUTA_PDFS)
    chunks = chunkear_documentos(docs)

    tfidf = TFIDFRetriever()
    tfidf.fit(chunks)

    retriever = EmbeddingRetriever()
    retriever.fit(chunks)
    rag = RAGDistilBERT(retriever)

    queries = [
        "qué es la brucelosis bovina",
        "definicion animal reaccionante positivo",
        "medidas sanitarias obligatorias en brucelosis",
        "como se registra un producto veterinario",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")

        # TF-IDF + extractivo
        res_tfidf = tfidf.buscar(query, top_k=4)
        resp_ext  = responder(res_tfidf, query, modo="sintetico")
        print(f"\n  [TF-IDF + Sintético]")
        print(f"  {resp_ext['respuesta'][:200]}")

        # FAISS + DistilBERT RAG
        res_rag = rag.preguntar(query, top_k=4)
        print(f"\n  [FAISS + DistilBERT RAG] score_qa={res_rag['score_qa']:.4f}")
        print(f"  {res_rag['respuesta'][:200]}")
        print(f"  Contexto: {len(res_rag['contexto'])} chars")
        print(f"  Fuentes: {[f['doc_id'] for f in res_rag['fuentes']]}")


# =========================================================
# 🟢 MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST RAG DISTILBERT + FAISS")
    print("=" * 60)

    # Tests FAISS
    test_faiss_index_construido()
    test_faiss_dimension_correcta()
    test_faiss_embeddings_normalizados()
    test_faiss_scores_entre_0_y_1()
    test_faiss_ordenado_por_score()
    test_faiss_top_k()
    test_faiss_articulo_bypass()

    # Tests RAG
    test_rag_devuelve_campos_requeridos()
    test_rag_respuesta_no_vacia()
    test_rag_contexto_contiene_chunks()
    test_rag_fuentes_tienen_campos()
    test_rag_sin_resultados_fallback()
    test_rag_contexto_truncado()
    test_rag_queries_curadas()

    print("\n✅ Todos los tests pasaron.")
    print()

    demo_comparacion()
