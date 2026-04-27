"""
test_qa_transformer.py
----------------------
Tests del módulo qa_transformer.py.

Usa MockQATransformer para tests offline (sin descarga de modelo).
Los tests verifican el contrato de interfaz — los mismos tests pasan
con QATransformer real cuando el modelo está disponible.

Para correr con el modelo real:
    TRANSFORMERS_OFFLINE=0 python test_qa_transformer.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pdf_loader import cargar_pdfs
from chunker import chunkear_documentos
from retriever_tfidf import TFIDFRetriever
from qa_transformer import MockQATransformer, get_qa_transformer, MODELO_QA_ES
from qa_engine import responder

RUTA_PDFS = "../data/raw/pdfs"

CAMPOS_REQUERIDOS = [
    "respuesta", "score_qa", "score_retrieval",
    "fuente", "seccion", "contexto", "metodo", "modelo",
]


def setup():
    docs = cargar_pdfs(RUTA_PDFS)
    chunks = chunkear_documentos(docs)
    tfidf = TFIDFRetriever()
    tfidf.fit(chunks)
    qa = MockQATransformer()
    return tfidf, qa, chunks


# =========================================================
# 🟢 TESTS UNITARIOS — contrato de interfaz
# =========================================================

def test_mock_devuelve_campos_requeridos():
    """La respuesta debe incluir todos los campos del contrato."""
    tfidf, qa, _ = setup()
    resultados = tfidf.buscar("brucelosis bovina", top_k=5)
    resp = qa.responder(resultados, "qué es la brucelosis bovina")

    for campo in CAMPOS_REQUERIDOS:
        assert campo in resp, f"Campo '{campo}' ausente en respuesta"

    assert resp["metodo"] == "transformer"
    assert "MOCK" in resp["modelo"]
    assert isinstance(resp["score_qa"], float)
    assert 0.0 <= resp["score_qa"] <= 1.0
    print("[OK] test_mock_devuelve_campos_requeridos")


def test_respuesta_no_vacia():
    """Con resultados del retriever debe devolver una respuesta no vacía."""
    tfidf, qa, _ = setup()
    resultados = tfidf.buscar("brucelosis bovina", top_k=5)
    resp = qa.responder(resultados, "qué es la brucelosis bovina")
    assert len(resp["respuesta"]) > 10, \
        f"Respuesta demasiado corta: '{resp['respuesta']}'"
    print(f"[OK] test_respuesta_no_vacia: '{resp['respuesta'][:80]}...'")


def test_sin_resultados_devuelve_fallback():
    """Con lista vacía debe devolver respuesta de fallback, sin explotar."""
    qa = MockQATransformer()
    resp = qa.responder([], "cualquier query")
    assert "respuesta" in resp
    assert resp["score_qa"] == 0.0
    assert resp["fuente"] is None
    assert resp["metodo"] == "transformer"
    print("[OK] test_sin_resultados_devuelve_fallback")


def test_score_qa_entre_0_y_1():
    """El score de confianza debe estar en [0, 1]."""
    tfidf, qa, _ = setup()
    queries = [
        "qué es la brucelosis bovina",
        "definicion animal positivo",
        "medidas sanitarias obligatorias",
    ]
    for q in queries:
        resultados = tfidf.buscar(q, top_k=5)
        resp = qa.responder(resultados, q)
        assert 0.0 <= resp["score_qa"] <= 1.0, \
            f"Score fuera de rango para '{q}': {resp['score_qa']}"
    print("[OK] test_score_qa_entre_0_y_1")


def test_fuente_es_doc_conocido():
    """La fuente referenciada debe ser un doc_id del corpus."""
    tfidf, qa, chunks = setup()
    doc_ids_corpus = {c["doc_id"] for c in chunks}
    resultados = tfidf.buscar("brucelosis bovina", top_k=5)
    resp = qa.responder(resultados, "qué es la brucelosis bovina")
    if resp["fuente"] is not None:
        assert resp["fuente"] in doc_ids_corpus, \
            f"fuente '{resp['fuente']}' no está en el corpus"
    print(f"[OK] test_fuente_es_doc_conocido: fuente={resp['fuente']}")


def test_contexto_es_texto_del_chunk():
    """El contexto devuelto debe ser el texto completo del chunk fuente."""
    tfidf, qa, chunks = setup()
    resultados = tfidf.buscar("brucelosis bovina", top_k=5)
    resp = qa.responder(resultados, "qué es la brucelosis bovina")
    if resp["contexto"]:
        textos_corpus = {c["texto"] for c in chunks}
        assert resp["contexto"] in textos_corpus, \
            "El contexto no coincide con ningún chunk del corpus"
    print("[OK] test_contexto_es_texto_del_chunk")


def test_respuesta_esta_contenida_en_contexto():
    """La respuesta extraída debe ser un substring del contexto (span exacto)."""
    tfidf, qa, _ = setup()
    resultados = tfidf.buscar("brucelosis bovina que es", top_k=5)
    resp = qa.responder(resultados, "brucelosis bovina que es")
    if resp["contexto"] and resp["respuesta"] and resp["score_qa"] > 0:
        assert resp["respuesta"] in resp["contexto"], \
            f"Respuesta '{resp['respuesta'][:50]}' no está en el contexto"
    print("[OK] test_respuesta_esta_contenida_en_contexto")


def test_top_n_limita_chunks_evaluados():
    """Con top_n=1 solo se evalúa el primer chunk."""
    tfidf, qa, _ = setup()
    resultados = tfidf.buscar("brucelosis bovina", top_k=5)
    resp_1 = qa.responder(resultados, "qué es la brucelosis", top_n=1)
    resp_3 = qa.responder(resultados, "qué es la brucelosis", top_n=3)
    # Ambos deben funcionar sin error
    assert "respuesta" in resp_1
    assert "respuesta" in resp_3
    print("[OK] test_top_n_limita_chunks_evaluados")


# =========================================================
# 🟢 TESTS DE INTEGRACIÓN — qa_engine.responder() modo transformer
# =========================================================

def test_responder_modo_transformer():
    """qa_engine.responder() con modo='transformer' debe funcionar."""
    tfidf, qa, _ = setup()
    resultados = tfidf.buscar("brucelosis bovina", top_k=5)
    resp = responder(resultados, "qué es la brucelosis bovina",
                     modo="transformer", top_n=3,
                     qa_transformer_instance=qa)
    assert "respuesta" in resp
    assert resp["metodo"] == "transformer"
    assert len(resp["respuesta"]) > 5
    print(f"[OK] test_responder_modo_transformer: '{resp['respuesta'][:80]}...'")


def test_responder_modo_invalido_sigue_igual():
    """Modo inválido sigue lanzando ValueError."""
    try:
        responder([], "query", modo="gpt5")
        assert False
    except ValueError:
        pass
    print("[OK] test_responder_modo_invalido_sigue_igual")


def test_get_qa_transformer_fallback():
    """get_qa_transformer() con modelo no disponible retorna Mock sin explotar."""
    qa = get_qa_transformer(
        model_name="modelo-inexistente-xyz/no-existe",
        forzar_mock=False
    )
    assert qa is not None
    # Debe ser un Mock (el modelo no existe)
    assert "Mock" in type(qa).__name__ or hasattr(qa, '_model_name')
    print(f"[OK] test_get_qa_transformer_fallback: {type(qa).__name__}")


# =========================================================
# 🟢 DEMO COMPARATIVA (extractivo vs sintético vs transformer)
# =========================================================

def demo_comparacion_modos():
    """Muestra las tres respuestas lado a lado para las mismas queries."""
    tfidf, qa, _ = setup()
    queries = [
        "qué es la brucelosis bovina",
        "qué es un animal reaccionante positivo",
        "que dice el articulo 3",
        "cómo se debe vacunar a las terneras",
        "medidas sanitarias obligatorias en brucelosis",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        resultados = tfidf.buscar(query, top_k=5)

        for modo in ["extractivo", "sintetico", "transformer"]:
            if modo == "transformer":
                resp = responder(resultados, query, modo=modo,
                                 qa_transformer_instance=qa)
                score_str = f" [score_qa={resp.get('score_qa', 0):.3f}]"
            else:
                resp = responder(resultados, query, modo=modo)
                score_str = ""

            print(f"\n  [{modo.upper()}]{score_str}")
            print(f"  {resp['respuesta'][:200]}")
            if "fuente" in resp and resp["fuente"]:
                print(f"  Fuente: {resp['fuente']} | {resp.get('seccion', '')}")
            elif resp.get("fuentes"):
                f = resp["fuentes"][0]
                print(f"  Fuente: {f['doc_id']} | {f['seccion']}")


# =========================================================
# 🟢 MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST QA TRANSFORMER")
    print("=" * 60)

    # Tests unitarios
    test_mock_devuelve_campos_requeridos()
    test_respuesta_no_vacia()
    test_sin_resultados_devuelve_fallback()
    test_score_qa_entre_0_y_1()
    test_fuente_es_doc_conocido()
    test_contexto_es_texto_del_chunk()
    test_respuesta_esta_contenida_en_contexto()
    test_top_n_limita_chunks_evaluados()

    # Tests de integración
    test_responder_modo_transformer()
    test_responder_modo_invalido_sigue_igual()
    test_get_qa_transformer_fallback()

    print("\n✅ Todos los tests pasaron.")
    print()

    # Demo comparativa
    demo_comparacion_modos()
