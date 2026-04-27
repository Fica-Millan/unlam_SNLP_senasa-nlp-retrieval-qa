"""
test_api.py
-----------
Tests de la API REST usando TestClient de FastAPI.
No requiere servidor corriendo — ejecuta in-process.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_raiz():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "nombre" in data
    assert "/buscar" in data["endpoints"]
    print("[OK] test_raiz")


def test_estado():
    r = client.get("/estado")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["chunks_indexados"] > 0
    assert data["documentos_cargados"] == 10
    assert set(data["modelos_disponibles"]) == {"tfidf", "embeddings", "hybrid"}
    print(f"[OK] test_estado: {data['chunks_indexados']} chunks, {data['documentos_cargados']} docs")


def test_buscar_basico():
    r = client.post("/buscar", json={"query": "brucelosis bovina que es"})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "brucelosis bovina que es"
    assert data["modelo"] == "hybrid"
    assert len(data["resultados"]) == 5
    assert data["total_resultados"] == 5
    print(f"[OK] test_buscar_basico: {len(data['resultados'])} resultados")


def test_buscar_modelo_tfidf():
    r = client.post("/buscar", json={"query": "articulo 3", "modelo": "tfidf", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["modelo"] == "tfidf"
    assert len(data["resultados"]) == 3
    for res in data["resultados"]:
        assert res["article_number"] == 3
    print("[OK] test_buscar_modelo_tfidf: artículo 3 encontrado correctamente")


def test_buscar_modelo_embeddings():
    r = client.post("/buscar", json={"query": "definicion producto veterinario", "modelo": "embeddings"})
    assert r.status_code == 200
    data = r.json()
    assert data["modelo"] == "embeddings"
    assert len(data["resultados"]) > 0
    print("[OK] test_buscar_modelo_embeddings")


def test_buscar_top_k_variable():
    for k in [1, 5, 10]:
        r = client.post("/buscar", json={"query": "sanidad animal", "top_k": k})
        assert r.status_code == 200
        assert len(r.json()["resultados"]) == k
    print("[OK] test_buscar_top_k_variable")


def test_buscar_filtro_articulos():
    r = client.post("/buscar", json={
        "query": "requisitos habilitacion",
        "solo_articulos": True,
        "top_k": 5,
    })
    assert r.status_code == 200
    data = r.json()
    for res in data["resultados"]:
        assert "ARTÍCULO" in res["seccion"], \
            f"Sección no es ARTÍCULO: {res['seccion']}"
    print(f"[OK] test_buscar_filtro_articulos: {len(data['resultados'])} resultados de artículos")


def test_buscar_filtro_definiciones():
    r = client.post("/buscar", json={
        "query": "definicion producto veterinario",
        "solo_definiciones": True,
        "top_k": 5,
    })
    assert r.status_code == 200
    data = r.json()
    for res in data["resultados"]:
        assert res["es_definicion"] is True, \
            f"Chunk no marcado como definición: {res['texto_preview'][:80]}"
    print(f"[OK] test_buscar_filtro_definiciones: {len(data['resultados'])} resultados")


def test_buscar_tipo_query_detectado():
    casos = [
        ("articulo 5", "lexica"),
        ("que es brucelosis", "semantica"),
        ("medidas sanitarias obligatorias", "mixta"),
    ]
    for query, tipo_esperado in casos:
        r = client.post("/buscar", json={"query": query})
        assert r.status_code == 200
        assert r.json()["tipo_query"] == tipo_esperado, \
            f"Query '{query}': esperado '{tipo_esperado}', obtenido '{r.json()['tipo_query']}'"
    print("[OK] test_buscar_tipo_query_detectado")


def test_buscar_modelo_invalido():
    r = client.post("/buscar", json={"query": "test", "modelo": "gpt4"})
    assert r.status_code == 422  # Pydantic validation error
    print("[OK] test_buscar_modelo_invalido: 422 correctamente")


def test_buscar_query_corta():
    r = client.post("/buscar", json={"query": "ab"})
    assert r.status_code == 422
    print("[OK] test_buscar_query_corta: 422 correctamente")


def test_resultado_tiene_campos():
    r = client.post("/buscar", json={"query": "producto veterinario"})
    assert r.status_code == 200
    for res in r.json()["resultados"]:
        for campo in ["chunk_id", "doc_id", "score", "seccion", "texto", "texto_preview",
                      "es_definicion", "es_obligacion"]:
            assert campo in res, f"Campo '{campo}' ausente en resultado"
    print("[OK] test_resultado_tiene_campos")


def test_comparar():
    r = client.post("/comparar", json={"query": "brucelosis bovina", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert len(data["comparacion"]) == 3
    modelos = {m["modelo"] for m in data["comparacion"]}
    assert modelos == {"tfidf", "embeddings", "hybrid"}
    for modelo_result in data["comparacion"]:
        assert len(modelo_result["resultados"]) == 3
    print("[OK] test_comparar: 3 modelos con 3 resultados cada uno")


def test_documento_existente():
    r = client.get("/documento/Res_067-2019")
    assert r.status_code == 200
    data = r.json()
    assert data["doc_id"] == "Res_067-2019"
    assert data["total_chunks"] > 0
    print(f"[OK] test_documento_existente: {data['total_chunks']} chunks")


def test_documento_inexistente():
    r = client.get("/documento/Res_999-9999")
    assert r.status_code == 404
    print("[OK] test_documento_inexistente: 404 correctamente")


def test_documento_filtro_seccion():
    r = client.get("/documento/Res_067-2019?seccion=ARTÍCULO")
    assert r.status_code == 200
    data = r.json()
    for c in data["chunks"]:
        assert "ARTÍCULO" in c["seccion"]
    print(f"[OK] test_documento_filtro_seccion: {data['total_chunks']} artículos")


def test_listar_documentos():
    r = client.get("/documentos")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 10
    for doc in data["documentos"]:
        for campo in ["id", "numero_resolucion", "anio", "chunks"]:
            assert campo in doc
    print(f"[OK] test_listar_documentos: {data['total']} documentos")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST API REST — NORMATIVA SENASA")
    print("=" * 60)
    test_raiz()
    test_estado()
    test_buscar_basico()
    test_buscar_modelo_tfidf()
    test_buscar_modelo_embeddings()
    test_buscar_top_k_variable()
    test_buscar_filtro_articulos()
    test_buscar_filtro_definiciones()
    test_buscar_tipo_query_detectado()
    test_buscar_modelo_invalido()
    test_buscar_query_corta()
    test_resultado_tiene_campos()
    test_comparar()
    test_documento_existente()
    test_documento_inexistente()
    test_documento_filtro_seccion()
    test_listar_documentos()
    print("\n✅ Todos los tests de la API pasaron.")
