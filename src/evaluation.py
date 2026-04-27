"""
evaluation.py
-------------
Evaluación de modelos de recuperación (TF-IDF / Embeddings / Híbrido).

Métricas implementadas:
- Precision@k:  fracción de resultados relevantes entre los top-k (sin duplicados)
- Hit Rate@k:   1.0 si al menos un chunk_id gold está en top-k
- Recall@k:     relevantes_encontrados / total_relevantes (requiere total_relevantes)
- MRR:          Mean Reciprocal Rank
- F1@k:         media armónica de Precision@k y Recall@k

Criterio de relevancia — DOS modos:
  1. Gold standard por chunk_id (preferido): exacto, sin ambigüedad.
  2. Frase en texto (fallback): útil cuando no se conocen los chunk_ids.

Para el TFI se usa el modo 1 con chunk_ids verificados manualmente.
"""

import re
from typing import List, Optional
from statistics import mean


# =========================================================
# 🟢 DEDUPLICACIÓN DE RESULTADOS
# =========================================================

def deduplicar_top_k(resultados: list, k: int) -> list:
    """
    Retorna hasta k resultados con chunk_id único.
    Evita que un retriever que devuelva duplicados infle artificialmente
    las métricas de precisión.
    """
    vistos = set()
    dedup = []
    for r in resultados:
        cid = r["chunk"].get("chunk_id")
        if cid not in vistos:
            vistos.add(cid)
            dedup.append(r)
        if len(dedup) == k:
            break
    return dedup


# =========================================================
# 🟢 CRITERIO DE RELEVANCIA — MODO 1: chunk_id exacto
# =========================================================

def es_relevante_por_id(chunk: dict, gold_ids: List[str]) -> bool:
    """
    Relevancia basada en chunk_id exacto.
    Es el criterio más riguroso: no hay falsos positivos posibles.
    """
    return chunk.get("chunk_id") in gold_ids


# =========================================================
# 🟢 CRITERIO DE RELEVANCIA — MODO 2: frase en texto (fallback)
# =========================================================

def contiene_numero_exacto(texto: str, numero: str) -> bool:
    return re.search(rf"\b{re.escape(numero)}\b", texto) is not None


def es_relevante(chunk_texto: str, relevantes: List[str], umbral: float = 0.6) -> bool:
    """
    Relevancia por frase en texto.
    Útil para evaluación exploratoria; preferir es_relevante_por_id
    cuando se dispone del gold standard.
    """
    texto = chunk_texto.lower()
    for ref in relevantes:
        tokens = ref.lower().split()
        numeros = [t for t in tokens if t.isdigit()]
        palabras = [t for t in tokens if not t.isdigit()]

        if numeros:
            if not all(contiene_numero_exacto(texto, n) for n in numeros):
                continue
            if palabras and not all(p in texto for p in palabras):
                continue
            return True
        else:
            if not palabras:
                continue
            hits = sum(1 for p in palabras if p in texto)
            if hits / len(palabras) >= umbral:
                return True
    return False


# =========================================================
# 🟢  MÉTRICAS INDIVIDUALES
# =========================================================

def precision_at_k(resultados, relevantes_o_ids, k, usar_ids=False):
    """
    Fracción de resultados relevantes entre los top-k.
    Aplica deduplicación por chunk_id antes de calcular.
    """
    top_k = deduplicar_top_k(resultados, k)
    if not top_k:
        return 0.0
    if usar_ids:
        hits = sum(1 for r in top_k if es_relevante_por_id(r["chunk"], relevantes_o_ids))
    else:
        hits = sum(1 for r in top_k if es_relevante(r["chunk"]["texto"], relevantes_o_ids))
    return hits / len(top_k)


def hit_rate_at_k(resultados, relevantes_o_ids, k, usar_ids=False):
    """
    1.0 si al menos un resultado relevante aparece en los primeros k.
    0.0 en caso contrario.
    """
    for r in deduplicar_top_k(resultados, k):
        if usar_ids:
            if es_relevante_por_id(r["chunk"], relevantes_o_ids):
                return 1.0
        else:
            if es_relevante(r["chunk"]["texto"], relevantes_o_ids):
                return 1.0
    return 0.0


def recall_at_k(resultados, relevantes_o_ids, k, total_relevantes: int, usar_ids=False):
    """
    Recall real: relevantes_encontrados / total_relevantes.

    Requiere total_relevantes explícito — no tiene fallback silencioso.
    Si no se conoce el total, usar hit_rate_at_k en su lugar.

    Args:
        total_relevantes: número total de chunks relevantes en el corpus
                          para esta query. Debe ser > 0.
    """
    if total_relevantes <= 0:
        raise ValueError("total_relevantes debe ser un entero positivo.")

    top_k = deduplicar_top_k(resultados, k)
    if usar_ids:
        encontrados = sum(1 for r in top_k if es_relevante_por_id(r["chunk"], relevantes_o_ids))
    else:
        encontrados = sum(1 for r in top_k if es_relevante(r["chunk"]["texto"], relevantes_o_ids))
    return encontrados / total_relevantes


def reciprocal_rank(resultados, relevantes_o_ids, usar_ids=False):
    """
    Recíproco de la posición del primer resultado relevante.
    0.0 si ningún resultado es relevante.
    """
    for i, r in enumerate(resultados):
        if usar_ids:
            if es_relevante_por_id(r["chunk"], relevantes_o_ids):
                return 1 / (i + 1)
        else:
            if es_relevante(r["chunk"]["texto"], relevantes_o_ids):
                return 1 / (i + 1)
    return 0.0


def f1_score(p: float, r: float) -> float:
    """
    Media armónica de Precision y Recall.
    Retorna 0.0 si ambos son cero (evita división por cero).
    """
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


# =========================================================
# 🟢 EVALUACIÓN GLOBAL
# =========================================================

def evaluar_modelo(retriever, eval_queries, k=5):
    """
    Evaluación simple con promedios globales.
    Calcula recall solo cuando la query tiene total_relevantes definido.
    """
    precisions, hit_rates, recalls, mrrs, f1s = [], [], [], [], []

    print("=" * 50)
    print(f"EVALUACIÓN — top-{k}")
    print("=" * 50)

    for q in eval_queries:
        query    = q["query"]
        usar_ids = "gold_ids" in q
        refs     = q.get("gold_ids", q.get("relevantes", []))
        total_rel = q.get("total_relevantes")

        resultados = retriever.buscar(query, top_k=k)

        p  = precision_at_k(resultados, refs, k, usar_ids)
        hr = hit_rate_at_k(resultados, refs, k, usar_ids)
        rr = reciprocal_rank(resultados, refs, usar_ids)

        # Recall solo cuando está disponible el total
        r = recall_at_k(resultados, refs, k, total_rel, usar_ids) if total_rel else None
        f = f1_score(p, r) if r is not None else f1_score(p, hr)

        precisions.append(p)
        hit_rates.append(hr)
        mrrs.append(rr)
        f1s.append(f)
        if r is not None:
            recalls.append(r)

        recall_str = f"{r:.3f}" if r is not None else "N/A"
        print(f"\nQuery: {query}")
        print(f"  P@{k}={p:.3f} | HR@{k}={hr:.3f} | R@{k}={recall_str} | RR={rr:.3f} | F1={f:.3f}")

    metricas = {
        "precision@k": mean(precisions),
        "hit_rate@k":  mean(hit_rates),
        "MRR":         mean(mrrs),
        "F1@k":        mean(f1s),
    }
    if recalls:
        metricas["recall@k"] = mean(recalls)

    print(f"\nPromedios: P={metricas['precision@k']:.3f} | HR={metricas['hit_rate@k']:.3f} | "
          f"MRR={metricas['MRR']:.3f} | F1={metricas['F1@k']:.3f}")
    return metricas


# =========================================================
# 🟢 EVALUACIÓN DETALLADA (con desagregación por tipo)
# =========================================================

def evaluar_modelo_detallado(retriever, eval_queries, k=5):
    """
    Evaluación con desagregación por tipo de query: lexica / semantica / mixta.
    Recall solo se calcula cuando total_relevantes está definido en la query.
    F1 usa recall real si disponible, hit_rate como proxy si no.
    """
    resultados_por_query = []

    for q in eval_queries:
        query    = q["query"]
        tipo     = q.get("tipo", "general")
        usar_ids = "gold_ids" in q
        refs     = q.get("gold_ids", q.get("relevantes", []))
        total_rel = q.get("total_relevantes")

        resultados = retriever.buscar(query, top_k=k)

        p  = precision_at_k(resultados, refs, k, usar_ids)
        hr = hit_rate_at_k(resultados, refs, k, usar_ids)
        rr = reciprocal_rank(resultados, refs, usar_ids)
        r  = recall_at_k(resultados, refs, k, total_rel, usar_ids) if total_rel else None
        f  = f1_score(p, r if r is not None else hr)

        resultados_por_query.append({
            "query":     query,
            "tipo":      tipo,
            "precision": p,
            "hit_rate":  hr,
            "recall":    r,       # None si no hay total_relevantes
            "rr":        rr,
            "f1":        f,
        })

    # --- Métricas globales ---
    recalls_validos = [r["recall"] for r in resultados_por_query if r["recall"] is not None]

    metricas = {
        "precision@k": mean(r["precision"] for r in resultados_por_query),
        "hit_rate@k":  mean(r["hit_rate"]  for r in resultados_por_query),
        "MRR":         mean(r["rr"]        for r in resultados_por_query),
        "F1@k":        mean(r["f1"]        for r in resultados_por_query),
    }
    if recalls_validos:
        metricas["recall@k"] = mean(recalls_validos)

    # --- Métricas por tipo ---
    for tipo in ["lexica", "semantica", "mixta"]:
        subset = [r for r in resultados_por_query if r["tipo"] == tipo]
        if subset:
            metricas[f"MRR_{tipo}"]       = mean(r["rr"]        for r in subset)
            metricas[f"HitRate_{tipo}"]   = mean(r["hit_rate"]  for r in subset)
            metricas[f"Precision_{tipo}"] = mean(r["precision"] for r in subset)
            metricas[f"F1_{tipo}"]        = mean(r["f1"]        for r in subset)
            recalls_tipo = [r["recall"] for r in subset if r["recall"] is not None]
            if recalls_tipo:
                metricas[f"Recall_{tipo}"] = mean(recalls_tipo)

    return {"metricas": metricas, "por_query": resultados_por_query}
