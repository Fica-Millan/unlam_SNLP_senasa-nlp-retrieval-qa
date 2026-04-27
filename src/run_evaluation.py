"""
run_evaluation.py
-----------------
Script de evaluación completo y reproducible del sistema de recuperación
y comprensión de normativa SENASA.

Evalúa dos dimensiones:
    1. RETRIEVAL:  Precision@k, Recall@k, Hit Rate@k, MRR, F1@k
                   desagregados por tipo de query (léxica / semántica / mixta)
    2. QA:         F1 de tokens y Exact Match sobre respuestas generadas
                   por los tres modos: extractivo, sintético y transformer

Modelos evaluados:
    - TF-IDF + similitud coseno (modelo clásico)
    - Embeddings semánticos LSA/SVD (MockEmbeddingRetriever, proxy offline)
    - Híbrido TF-IDF + Embeddings con alpha=0.8 (modelo combinado)

Modos QA evaluados:
    - Extractivo: oración más relevante por overlap léxico
    - Sintético:  síntesis heurística multi-chunk
    - Transformer: span exacto por MockQATransformer (proxy offline de BERT)

Salidas:
    outputs/eval_retrieval.json      → métricas de retrieval por modelo y tipo
    outputs/eval_qa.json             → métricas QA (EM + F1 tokens) por modelo y modo
    outputs/eval_resumen.json        → tabla consolidada para el informe
    outputs/eval_rag_distilbert.json → métricas del RAG DistilBERT (solo --modo real)

Ejecutar desde la raíz del proyecto:
    python src/run_evaluation.py              # modo mock (default, sin HuggingFace)
    python src/run_evaluation.py --modo real  # modo real (requiere HuggingFace)
    python src/run_evaluation.py --modo mock  # modo mock explícito

Modos disponibles:
    mock  → MockEmbeddingRetriever (LSA/SVD) + MockQATransformer (overlap léxico)
            Reproducible sin descarga de modelos. RAGDistilBERT se omite.
    real  → EmbeddingRetriever (SentenceTransformer) + QATransformer (BERT español)
            + RAGDistilBERT (DistilBERT con contexto concatenado, 1 inferencia)
            Requiere descarga previa de modelos HuggingFace. Produce métricas definitivas.
"""

import sys
import os
import json
import re
import argparse
from statistics import mean

# Path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from pdf_loader          import cargar_pdfs
from chunker             import chunkear_documentos
from retriever_tfidf     import TFIDFRetriever
from mock_embeddings     import MockEmbeddingRetriever
from retriever_hybrid    import build_hybrid
from evaluation          import evaluar_modelo_detallado
from qa_engine           import responder
from qa_transformer      import MockQATransformer


# =========================================================
# 🟢 MÉTRICAS QA: Exact Match y F1 de tokens (estilo SQuAD)
# =========================================================

def _normalizar(texto: str) -> str:
    """
    Normaliza texto para comparación:
    - Minúsculas
    - Elimina acentos
    - Elimina puntuación
    """
    texto = texto.lower()
    for a, b in [("áàä","a"),("éèë","e"),("íìï","i"),("óòö","o"),("úùü","u")]:
        for c in a:
            texto = texto.replace(c, b)
    texto = re.sub(r"[^a-z0-9 ]", "", texto)
    return texto.strip()


def exact_match(pred: str, gold: str) -> int:
    """1 si la predicción normalizada es igual al gold normalizado, 0 si no."""
    return int(_normalizar(pred) == _normalizar(gold))


def f1_tokens(pred: str, gold: str) -> float:
    """
    F1 de tokens al estilo SQuAD:
    Precision = tokens comunes / tokens predicción
    Recall    = tokens comunes / tokens gold
    F1        = media armónica de P y R
    """
    pred_tokens = _normalizar(pred).split()
    gold_tokens = _normalizar(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    comunes = set(pred_tokens) & set(gold_tokens)
    if not comunes:
        return 0.0
    p = len(comunes) / len(pred_tokens)
    r = len(comunes) / len(gold_tokens)
    return 2 * p * r / (p + r)


# =========================================================
# 🟢 EVALUACIÓN QA
# =========================================================

def evaluar_qa(retriever, nombre_modelo: str, eval_queries: list,
               qa_mock: MockQATransformer, top_k: int = 5) -> dict:
    """
    Evalúa los tres modos QA (extractivo, sintético, transformer)
    sobre el conjunto de evaluación.

    Returns:
        Dict con métricas agregadas y detalle por query.
    """
    modos = ["extractivo", "sintetico", "transformer"]
    acumuladores = {m: {"EM": [], "F1": []} for m in modos}
    por_query = []

    for q in eval_queries:
        query = q["query"]
        gold  = q["respuesta_esperada"]
        resultados_retriever = retriever.buscar(query, top_k=top_k)

        detalle = {"id": q["id"], "query": query, "tipo": q["tipo"], "gold": gold}

        for modo in modos:
            kwargs = {}
            if modo == "transformer":
                kwargs["qa_transformer_instance"] = qa_mock

            resp = responder(resultados_retriever, query, modo=modo, top_n=3, **kwargs)
            pred = resp["respuesta"]

            em = exact_match(pred, gold)
            f1 = f1_tokens(pred, gold)

            acumuladores[modo]["EM"].append(em)
            acumuladores[modo]["F1"].append(f1)

            detalle[modo] = {
                "respuesta": pred,
                "EM":        em,
                "F1":        round(f1, 4),
            }

        por_query.append(detalle)

    metricas = {}
    for modo in modos:
        metricas[f"EM_{modo}"]  = round(mean(acumuladores[modo]["EM"]), 4)
        metricas[f"F1_{modo}"]  = round(mean(acumuladores[modo]["F1"]), 4)

    return {"modelo": nombre_modelo, "metricas": metricas, "por_query": por_query}


# =========================================================
# 🟢 MAIN
# =========================================================

def main():
    # --------------------------------------------------
    # 0. Parsear argumentos
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description="Evaluación del sistema de normativa SENASA")
    parser.add_argument(
        "--modo",
        choices=["mock", "real"],
        default="mock",
        help="mock: LSA/SVD + overlap léxico (sin HuggingFace). real: SentenceTransformer + BERT español."
    )
    args = parser.parse_args()
    modo_real = args.modo == "real"

    data_dir    = os.path.join(ROOT_DIR, "data")
    outputs_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Cargar corpus
    # --------------------------------------------------
    print("=" * 60)
    print("EVALUACIÓN DEL SISTEMA DE NORMATIVA SENASA")
    print(f"Modo: {'REAL (SentenceTransformer + BERT)' if modo_real else 'MOCK (LSA/SVD + overlap léxico)'}")
    print("=" * 60)
    print("\n[1/5] Cargando corpus...")
    docs   = cargar_pdfs(os.path.join(data_dir, "raw", "pdfs"))
    chunks = chunkear_documentos(docs)
    print(f"      {len(docs)} documentos | {len(chunks)} chunks")

    # --------------------------------------------------
    # 2. Entrenar modelos
    # --------------------------------------------------
    print("\n[2/5] Entrenando modelos...")

    tfidf = TFIDFRetriever()
    tfidf.fit(chunks)
    print("      ✓ TF-IDF")

    if modo_real:
        try:
            from retriever_embeddings import EmbeddingRetriever
            embedder = EmbeddingRetriever()
            embedder.fit(chunks)
            print("      ✓ Embeddings (SentenceTransformer — paraphrase-multilingual-MiniLM-L12-v2)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"      ⚠️  EmbeddingRetriever falló ({e}). Usando Mock como fallback.")
            embedder = MockEmbeddingRetriever()
            embedder.fit(chunks)
            modo_real = False
    else:
        embedder = MockEmbeddingRetriever()
        embedder.fit(chunks)
        print("      ✓ Embeddings (Mock LSA/SVD — proxy offline)")

    hybrid = build_hybrid(chunks, tfidf=tfidf, embedder=embedder, alpha=0.8)
    print("      ✓ Híbrido (alpha=0.8)")

    if modo_real:
        try:
            from qa_transformer import QATransformer
            qa_engine = QATransformer()
            print("      ✓ QATransformer (BERT español fine-tuneado en SQuAD)")
        except Exception as e:
            print(f"      ⚠️  QATransformer falló ({e}). Usando Mock como fallback.")
            qa_engine = MockQATransformer()
    else:
        qa_engine = MockQATransformer()
        print("      ✓ QATransformer (Mock — proxy offline de BERT)")

    modelos = {
        "tfidf":      tfidf,
        "embeddings": embedder,
        "hybrid":     hybrid,
    }

    # --------------------------------------------------
    # 3. Cargar gold standard
    # --------------------------------------------------
    print("\n[3/5] Cargando gold standard...")
    eval_path = os.path.join(data_dir, "eval_queries.json")
    with open(eval_path, encoding="utf-8") as f:
        eval_queries = json.load(f)
    print(f"      {len(eval_queries)} queries ({sum(1 for q in eval_queries if q['tipo']=='lexica')} léxicas, "
          f"{sum(1 for q in eval_queries if q['tipo']=='semantica')} semánticas, "
          f"{sum(1 for q in eval_queries if q['tipo']=='mixta')} mixtas)")

    # --------------------------------------------------
    # 4. Evaluación RETRIEVAL
    # --------------------------------------------------
    print("\n[4/5] Evaluación de Retrieval (k=5)...")
    K = 5
    retrieval_resultados = {}

    for nombre, retriever in modelos.items():
        res = evaluar_modelo_detallado(retriever, eval_queries, k=K)
        retrieval_resultados[nombre] = res
        m = res["metricas"]
        print(f"\n  [{nombre.upper()}]")
        print(f"    P@{K}={m['precision@k']:.3f} | HR@{K}={m['hit_rate@k']:.3f} | "
              f"R@{K}={m.get('recall@k',0):.3f} | MRR={m['MRR']:.3f} | F1={m['F1@k']:.3f}")
        for tipo in ["lexica", "semantica", "mixta"]:
            if f"MRR_{tipo}" in m:
                print(f"    {tipo:10s} → MRR={m[f'MRR_{tipo}']:.3f} | "
                      f"HR={m[f'HitRate_{tipo}']:.3f} | F1={m[f'F1_{tipo}']:.3f}")

    with open(os.path.join(outputs_dir, "eval_retrieval.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_resultados, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------
    # 5. Evaluación QA (modos extractivo / sintético / transformer)
    # --------------------------------------------------
    print("\n[5/6] Evaluación de QA — modos heurístico y transformer (EM + F1 tokens)...")
    qa_resultados = {}

    for nombre, retriever in modelos.items():
        res = evaluar_qa(retriever, nombre, eval_queries, qa_engine, top_k=5)
        qa_resultados[nombre] = res
        m = res["metricas"]
        print(f"\n  [{nombre.upper()}]")
        print(f"    Extractivo  → EM={m['EM_extractivo']:.4f} | F1={m['F1_extractivo']:.4f}")
        print(f"    Sintético   → EM={m['EM_sintetico']:.4f} | F1={m['F1_sintetico']:.4f}")
        print(f"    Transformer → EM={m['EM_transformer']:.4f} | F1={m['F1_transformer']:.4f}")

    with open(os.path.join(outputs_dir, "eval_qa.json"), "w", encoding="utf-8") as f:
        json.dump(qa_resultados, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------
    # 5b. Evaluación RAG DistilBERT
    # --------------------------------------------------
    print("\n[6/6] Evaluación de QA — RAG DistilBERT (contexto concatenado)...")
    rag_resultados = None

    if modo_real:
        try:
            from rag_distilbert import RAGDistilBERT
            rag = RAGDistilBERT(retriever=embedder)
            ems_rag, f1s_rag = [], []
            rag_por_query = []

            for q in eval_queries:
                query = q["query"]
                gold  = q["respuesta_esperada"]
                res   = rag.preguntar(query, top_k=5)
                pred  = res["respuesta"]
                em    = exact_match(pred, gold)
                f1    = f1_tokens(pred, gold)
                ems_rag.append(em)
                f1s_rag.append(f1)
                rag_por_query.append({
                    "id": q["id"], "query": query, "tipo": q["tipo"],
                    "gold": gold, "respuesta": pred,
                    "score_qa": res["score_qa"],
                    "EM": em, "F1": round(f1, 4),
                })

            rag_resultados = {
                "modelo": "rag_distilbert",
                "retriever": "embeddings (SentenceTransformer + FAISS)",
                "metricas": {
                    "EM_medio": round(mean(ems_rag), 4),
                    "F1_medio": round(mean(f1s_rag), 4),
                },
                "por_query": rag_por_query,
            }
            print(f"\n  [RAG DISTILBERT]")
            print(f"    EM={rag_resultados['metricas']['EM_medio']:.4f} | "
                  f"F1={rag_resultados['metricas']['F1_medio']:.4f}")

        except Exception as e:
            print(f"      ⚠️  RAGDistilBERT no disponible ({e}). Omitiendo.")
    else:
        print("      ℹ️  RAGDistilBERT solo se evalúa en --modo real (requiere HuggingFace).")

    if rag_resultados:
        with open(os.path.join(outputs_dir, "eval_rag_distilbert.json"), "w", encoding="utf-8") as f:
            json.dump(rag_resultados, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------
    # 6. Tabla resumen consolidada
    # --------------------------------------------------
    resumen = {}
    for nombre in modelos:
        rm = retrieval_resultados[nombre]["metricas"]
        qm = qa_resultados[nombre]["metricas"]
        resumen[nombre] = {
            "retrieval": {
                "Precision@5":    round(rm["precision@k"], 4),
                "Recall@5":       round(rm.get("recall@k", 0), 4),
                "HitRate@5":      round(rm["hit_rate@k"], 4),
                "MRR":            round(rm["MRR"], 4),
                "F1@5":           round(rm["F1@k"], 4),
                "MRR_lexica":     round(rm.get("MRR_lexica", 0), 4),
                "MRR_semantica":  round(rm.get("MRR_semantica", 0), 4),
                "MRR_mixta":      round(rm.get("MRR_mixta", 0), 4),
            },
            "qa": {
                "EM_extractivo":   qm["EM_extractivo"],
                "F1_extractivo":   qm["F1_extractivo"],
                "EM_sintetico":    qm["EM_sintetico"],
                "F1_sintetico":    qm["F1_sintetico"],
                "EM_transformer":  qm["EM_transformer"],
                "F1_transformer":  qm["F1_transformer"],
            },
        }

    if rag_resultados:
        resumen["rag_distilbert"] = {
            "retrieval": "N/A — usa embedder internamente",
            "qa": {
                "EM_rag":  rag_resultados["metricas"]["EM_medio"],
                "F1_rag":  rag_resultados["metricas"]["F1_medio"],
            },
        }

    with open(os.path.join(outputs_dir, "eval_resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------
    # 7. Imprimir tablas finales
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TABLA RESUMEN — RETRIEVAL")
    print("=" * 60)
    header = f"{'Modelo':<12} {'P@5':>6} {'R@5':>6} {'HR@5':>6} {'MRR':>6} {'F1@5':>6} {'MRR-lex':>8} {'MRR-sem':>8} {'MRR-mix':>8}"
    print(header)
    print("-" * len(header))
    for nombre in modelos:
        r = resumen[nombre]["retrieval"]
        print(f"{nombre:<12} {r['Precision@5']:>6.4f} {r['Recall@5']:>6.4f} "
              f"{r['HitRate@5']:>6.4f} {r['MRR']:>6.4f} {r['F1@5']:>6.4f} "
              f"{r['MRR_lexica']:>8.4f} {r['MRR_semantica']:>8.4f} {r['MRR_mixta']:>8.4f}")

    print("\n" + "=" * 60)
    print("TABLA RESUMEN — QA (F1 tokens)")
    print("=" * 60)
    print(f"{'Modelo':<12} {'F1-ext':>8} {'F1-syn':>8} {'F1-tra':>8}")
    print("-" * 40)
    for nombre in modelos:
        q = resumen[nombre]["qa"]
        print(f"{nombre:<12} {q['F1_extractivo']:>8.4f} {q['F1_sintetico']:>8.4f} {q['F1_transformer']:>8.4f}")

    if rag_resultados:
        print("\n" + "=" * 60)
        print("TABLA RESUMEN — RAG DistilBERT (contexto concatenado)")
        print("=" * 60)
        rm = rag_resultados["metricas"]
        print(f"{'rag_distilbert':<14} EM={rm['EM_medio']:.4f} | F1={rm['F1_medio']:.4f}")
        print("  Retriever: embeddings (SentenceTransformer + FAISS)")
        print("  Modelo QA: mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")
        print("  Diferencia vs transformer: contexto concatenado (1 inferencia) vs chunk por chunk (N inferencias)")

    print(f"\n✅ Archivos guardados en: {outputs_dir}/")
    print("   eval_retrieval.json | eval_qa.json | eval_resumen.json", end="")
    if rag_resultados:
        print(" | eval_rag_distilbert.json", end="")
    print()


if __name__ == "__main__":
    main()
