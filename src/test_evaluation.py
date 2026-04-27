"""
test_evaluation.py
------------------
Evaluación comparativa TF-IDF / Embeddings / Híbrido.

Dataset de evaluación — criterio gold_ids:
  Las queries y gold_ids se cargan desde data/eval_queries.json,
  gold standard unificado de 30 queries verificadas manualmente
  contra el corpus real (15 originales + 15 del test previo).

  gold_ids se obtuvieron inspeccionando el corpus con inspect_corpus.py
  y seleccionando los chunks que realmente responden cada pregunta.

  total_relevantes indica cuántos chunks en todo el corpus son
  aceptables para esa query (para calcular Recall real).
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from pdf_loader import cargar_pdfs
from chunker import chunkear_documentos
from retriever_tfidf import TFIDFRetriever
from retriever_hybrid import build_hybrid, optimizar_alpha
from evaluation import (
    evaluar_modelo_detallado,
    deduplicar_top_k,
    es_relevante, es_relevante_por_id,
    precision_at_k, hit_rate_at_k, recall_at_k,
    reciprocal_rank, f1_score,
)
from visualization import (
    generar_resumen_ejecutivo,
    plot_alpha_vs_mrr,
    plot_correlacion_modelos,
    plot_metricas_globales,
    plot_mrr_por_tipo,
    plot_f1_por_tipo,
)

# ── Ruta al gold standard unificado ──────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, "..", "data")
_GOLD_PATH = os.path.join(_DATA_DIR, "eval_queries.json")

def _cargar_eval_queries():
    with open(_GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)

# Carga global para que los tests la usen directamente
EVAL_QUERIES = _cargar_eval_queries()



def get_embedding_retriever(chunks):
    try:
        from retriever_embeddings import EmbeddingRetriever
        r = EmbeddingRetriever(); r.fit(chunks)
        print("[INFO] EmbeddingRetriever real (SentenceTransformer) cargado.")
        return r
    except Exception as e:
        print(f"[INFO] Fallback a MockEmbeddingRetriever ({type(e).__name__}).")
        from mock_embeddings import MockEmbeddingRetriever
        r = MockEmbeddingRetriever(); r.fit(chunks)
        return r


# =========================================================
# 🟢 TESTS UNITARIOS DE MÉTRICAS
# =========================================================

def test_metricas_unitarias():
    """Verifica el comportamiento correcto de cada función de métrica."""
    fake = [
        {"score": 0.9, "chunk": {"chunk_id": "doc1_sec0", "texto": "brucelosis bovina enfermedad"}},
        {"score": 0.7, "chunk": {"chunk_id": "doc1_sec1", "texto": "producto veterinario registro"}},
        {"score": 0.3, "chunk": {"chunk_id": "doc1_sec2", "texto": "texto irrelevante general"}},
    ]
    gold = ["doc1_sec0"]

    # Precision@3: 1 de 3 relevante → 0.333
    p = precision_at_k(fake, gold, k=3, usar_ids=True)
    assert abs(p - 1/3) < 1e-6, f"P@3={p:.3f}, esperado 0.333"

    # Hit Rate: hay al menos uno → 1.0
    hr = hit_rate_at_k(fake, gold, k=3, usar_ids=True)
    assert hr == 1.0

    # Hit Rate con gold inexistente → 0.0
    hr0 = hit_rate_at_k(fake, ["inexistente_xyz"], k=3, usar_ids=True)
    assert hr0 == 0.0

    # RR: primer resultado relevante en posición 1 → 1.0
    rr = reciprocal_rank(fake, gold, usar_ids=True)
    assert abs(rr - 1.0) < 1e-6

    # RR con relevante en posición 2 → 0.5
    rr2 = reciprocal_rank(fake, ["doc1_sec1"], usar_ids=True)
    assert abs(rr2 - 0.5) < 1e-6

    # Recall: 1 encontrado de 2 totales → 0.5
    r = recall_at_k(fake, gold, k=3, total_relevantes=2, usar_ids=True)
    assert abs(r - 0.5) < 1e-6, f"Recall={r:.3f}, esperado 0.5"

    # F1: p=0.333, r=0.5 → F1 = 2*(0.333*0.5)/(0.333+0.5) ≈ 0.4
    f = f1_score(p, r)
    expected_f1 = 2 * (p * r) / (p + r)
    assert abs(f - expected_f1) < 1e-6, f"F1={f:.4f}, esperado {expected_f1:.4f}"

    # F1 con p=r=0 → 0.0 (sin división por cero)
    assert f1_score(0.0, 0.0) == 0.0

    print("[OK] test_metricas_unitarias (gold_ids + recall + f1)")


def test_deduplicacion():
    """Verifica que deduplicar_top_k elimina chunk_ids repetidos."""
    duplicados = [
        {"score": 0.9, "chunk": {"chunk_id": "A", "texto": "x"}},
        {"score": 0.8, "chunk": {"chunk_id": "A", "texto": "x"}},   # duplicado
        {"score": 0.7, "chunk": {"chunk_id": "B", "texto": "y"}},
        {"score": 0.6, "chunk": {"chunk_id": "C", "texto": "z"}},
    ]
    dedup = deduplicar_top_k(duplicados, k=3)
    ids = [r["chunk"]["chunk_id"] for r in dedup]
    assert ids == ["A", "B", "C"], f"Esperado [A,B,C], obtenido {ids}"
    assert len(dedup) == 3

    # Precision no debe inflarse con duplicados
    p_con_dup = precision_at_k(duplicados, ["A"], k=3, usar_ids=True)
    assert abs(p_con_dup - 1/3) < 1e-6, \
        f"Precision con duplicados={p_con_dup:.3f}, esperado 0.333 (no 0.667)"

    print("[OK] test_deduplicacion")


def test_recall_requiere_total():
    """Verifica que recall_at_k lanza error si total_relevantes <= 0."""
    fake = [{"score": 1.0, "chunk": {"chunk_id": "A", "texto": "x"}}]
    try:
        recall_at_k(fake, ["A"], k=1, total_relevantes=0, usar_ids=True)
        assert False, "Debería lanzar ValueError"
    except ValueError:
        pass
    print("[OK] test_recall_requiere_total")


def test_es_relevante_numeros():
    """Verifica matching exacto de números (sin falsos positivos)."""
    texto = "El artículo 33 establece los requisitos de habilitación."
    assert not es_relevante(texto, ["artículo 3"]), "Falso positivo: '3' en '33'"
    assert es_relevante(texto, ["artículo 33"]),    "No detectó '33'"
    print("[OK] test_es_relevante_numeros")


def test_dataset_verificado():
    """Verifica que todos los gold_ids existen en el corpus actual."""
    docs = cargar_pdfs("../data/raw/pdfs")
    chunks = chunkear_documentos(docs)
    ids_corpus = {c["chunk_id"] for c in chunks}

    errores = []
    for q in EVAL_QUERIES:
        for gid in q.get("gold_ids", []):
            if gid not in ids_corpus:
                errores.append(f"  ✗ '{gid}' (query: '{q['query']}')")

    if errores:
        raise AssertionError("gold_ids no encontrados:\n" + "\n".join(errores))

    print(f"[OK] test_dataset_verificado: todos los gold_ids existen en {len(chunks)} chunks")


# =========================================================
# 🟢 SETUP
# =========================================================

def build_models():
    docs = cargar_pdfs("../data/raw/pdfs")
    chunks = chunkear_documentos(docs)
    print(f"Chunks totales: {len(chunks)}")

    tfidf = TFIDFRetriever()
    tfidf.fit(chunks)
    embedder = get_embedding_retriever(chunks)
    hybrid = build_hybrid(chunks, tfidf=tfidf, embedder=embedder, alpha=0.8)
    return tfidf, embedder, hybrid, chunks


# =========================================================
# 🟢 EVALUACIÓN COMPARATIVA
# =========================================================

def test_evaluacion_comparativa():
    tfidf, embedder, hybrid, _ = build_models()

    eval_tfidf  = evaluar_modelo_detallado(tfidf,    EVAL_QUERIES, k=5)
    eval_emb    = evaluar_modelo_detallado(embedder, EVAL_QUERIES, k=5)
    eval_hybrid = evaluar_modelo_detallado(hybrid,   EVAL_QUERIES, k=5)

    print_comparacion(eval_tfidf, eval_emb, eval_hybrid)
    print_ganador_por_query(eval_tfidf, eval_emb, eval_hybrid)

    plot_metricas_globales(eval_tfidf, eval_emb, eval_hybrid)
    plot_mrr_por_tipo(eval_tfidf, eval_emb, eval_hybrid)
    plot_f1_por_tipo(eval_tfidf, eval_emb, eval_hybrid)

    for nombre, ev in [("TF-IDF", eval_tfidf), ("Embeddings", eval_emb), ("Hybrid", eval_hybrid)]:
        mrr = ev["metricas"]["MRR"]
        assert mrr > 0.0, f"{nombre} tiene MRR=0 — pipeline roto."
        print(
            f"[OK] {nombre}: "
            f"MRR={mrr:.3f} | "
            f"HR={ev['metricas']['hit_rate@k']:.3f} | "
            f"P={ev['metricas']['precision@k']:.3f} | "
            f"F1={ev['metricas']['F1@k']:.3f}"
        )

    return eval_tfidf, eval_emb, eval_hybrid


# =========================================================
# 🟢 GRID SEARCH DE ALPHA
# =========================================================

def test_grid_search_alpha():
    docs   = cargar_pdfs("../data/raw/pdfs")
    chunks = chunkear_documentos(docs)
    tfidf  = TFIDFRetriever(); tfidf.fit(chunks)
    embedder = get_embedding_retriever(chunks)

    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    resultado = optimizar_alpha(
        chunks, EVAL_QUERIES,
        tfidf=tfidf, embedder=embedder,
        alphas=alphas, k=5,
    )
    assert resultado["mejor_alpha"] in alphas
    mejor = resultado["mejor_alpha"]
    print(f"[OK] test_grid_search_alpha: alpha={mejor} (MRR={resultado['tabla'][mejor]:.3f})")

    plot_alpha_vs_mrr(resultado)


# =========================================================
# 🟢 HELPERS DE PRESENTACIÓN
# =========================================================

def print_comparacion(e_t, e_e, e_h):
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 60)
    modelos = {
        "TF-IDF":     e_t["metricas"],
        "Embeddings": e_e["metricas"],
        "Hybrid":     e_h["metricas"],
    }
    for m in ["precision@k", "hit_rate@k", "recall@k", "MRR", "F1@k"]:
        print(f"\n{m.upper()}")
        for nombre, met in modelos.items():
            val = met.get(m)
            if val is not None:
                print(f"  {nombre:<12}: {val:.3f}")
            else:
                print(f"  {nombre:<12}: N/A")

    print("\nMRR POR TIPO DE QUERY")
    for tipo in ["lexica", "semantica", "mixta"]:
        print(f"\n  {tipo.upper()}")
        for nombre, met in modelos.items():
            mrr_t = met.get(f"MRR_{tipo}")
            f1_t  = met.get(f"F1_{tipo}")
            if mrr_t is not None:
                f1_str = f" | F1={f1_t:.3f}" if f1_t is not None else ""
                print(f"    {nombre:<12}: MRR={mrr_t:.3f}{f1_str}")


def print_ganador_por_query(e_t, e_e, e_h):
    print("\n" + "=" * 60)
    print("GANADOR POR QUERY (según RR | F1)")
    print("=" * 60)
    for r_t, r_e, r_h in zip(
        e_t["por_query"], e_e["por_query"], e_h["por_query"]
    ):
        scores  = {"TF-IDF": r_t["rr"], "Emb": r_e["rr"], "Hybrid": r_h["rr"]}
        ganador = max(scores, key=scores.get)
        print(
            f"  {r_t['query']:<50} "
            f"TF={r_t['rr']:.2f}(F1={r_t['f1']:.2f})|"
            f"Em={r_e['rr']:.2f}(F1={r_e['f1']:.2f})|"
            f"Hy={r_h['rr']:.2f}(F1={r_h['f1']:.2f}) → {ganador}"
        )


# =========================================================
# 🟢 MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST EVALUATION — COMPARACIÓN DE MODELOS")
    print("=" * 60)

    test_metricas_unitarias()
    test_deduplicacion()
    test_recall_requiere_total()
    test_es_relevante_numeros()
    test_dataset_verificado()

    eval_tfidf, eval_emb, eval_hybrid = test_evaluacion_comparativa()

    test_grid_search_alpha()

    plot_correlacion_modelos(eval_tfidf, eval_emb, eval_hybrid)
    generar_resumen_ejecutivo(eval_tfidf, eval_emb, eval_hybrid)

    print("\n✅ Todos los tests pasaron.")
