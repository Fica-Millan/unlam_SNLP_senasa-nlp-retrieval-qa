"""
visualization.py
----------------
Generación de gráficos para evaluación de modelos RAG.
Exporta imágenes listas para documentación.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "figures")


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 🟢 MÉTRICAS GLOBALES
# =========================================================

def plot_metricas_globales(e_t, e_e, e_h):
    _ensure_dir()

    data = []
    modelos = {
        "TF-IDF": e_t["metricas"],
        "Embeddings": e_e["metricas"],
        "Hybrid": e_h["metricas"]
    }

    for modelo, met in modelos.items():
        for m in ["precision@k", "hit_rate@k", "recall@k", "MRR", "F1@k"]:
            data.append({
                "Modelo": modelo,
                "Métrica": m,
                "Valor": met.get(m, 0)
            })

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Guardamos el gráfico en la variable 'ax' y agregamos la paleta
    ax = sns.barplot(data=df, x="Métrica", y="Valor", hue="Modelo", palette="viridis")

    # --- AGREGAR VALORES EN LAS BARRAS ---
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.title("Comparación Global de Modelos", fontsize=14, pad=20)
    
    # Ajustamos el límite de Y para dar aire a las etiquetas
    plt.ylim(0, df["Valor"].max() * 1.15) 
    
    plt.legend(title="Modelos", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "metricas_globales.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[FIG] Guardado: {path}")


# =========================================================
# 🟢 MRR POR TIPO
# =========================================================

def plot_mrr_por_tipo(e_t, e_e, e_h):
    _ensure_dir()

    data = []
    modelos = {
        "TF-IDF": e_t["metricas"],
        "Embeddings": e_e["metricas"],
        "Hybrid": e_h["metricas"]
    }

    for modelo, met in modelos.items():
        for tipo in ["lexica", "semantica", "mixta"]:
            key = f"MRR_{tipo}"
            if key in met:
                data.append({
                    "Modelo": modelo,
                    "Tipo": tipo,
                    "MRR": met[key]
                })

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6)) # Un poco más ancho para que no se amontonen los números
    sns.set_style("whitegrid") # Agrega un fondo limpio
    
    ax = sns.barplot(data=df, x="Tipo", y="MRR", hue="Modelo", palette="viridis")

    # --- AGREGAR VALORES EN LAS BARRAS ---
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.title("MRR por tipo de query", fontsize=14, pad=15)
    plt.ylim(0, df["MRR"].max() * 1.15) # Damos espacio arriba para que el número no se corte
    plt.ylabel("Mean Reciprocal Rank (MRR)")
    plt.xlabel("Tipo de Consulta")
    plt.legend(title="Modelos", bbox_to_anchor=(1.05, 1), loc='upper left') # Mueve la leyenda afuera
    
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "mrr_por_tipo.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[FIG] Guardado: {path}")


# =========================================================
# 🟢 F1 POR TIPO
# =========================================================

def plot_f1_por_tipo(e_t, e_e, e_h):
    _ensure_dir()
    data = []
    modelos = {"TF-IDF": e_t["metricas"], "Embeddings": e_e["metricas"], "Hybrid": e_h["metricas"]}
    for modelo, met in modelos.items():
        for tipo in ["lexica", "semantica", "mixta"]:
            key = f"F1_{tipo}"
            if key in met:
                data.append({"Modelo": modelo, "Tipo": tipo, "F1": met[key]})
    if not data:
        print("[FIG] plot_f1_por_tipo: sin datos (F1 por tipo no calculado).")
        return
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, x="Tipo", y="F1", hue="Modelo", palette="viridis")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    plt.title("F1 por tipo de query", fontsize=14, pad=15)
    plt.ylim(0, df["F1"].max() * 1.15)
    plt.ylabel("F1@k")
    plt.xlabel("Tipo de Consulta")
    plt.legend(title="Modelos", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "f1_por_tipo.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"[FIG] Guardado: {path}")


# =========================================================
# 🟢 GRID SEARCH ALPHA
# =========================================================

def plot_alpha_vs_mrr(resultado):
    _ensure_dir()
    
    # Extraer el diccionario de alpha -> MRR
    tabla = resultado["tabla"] if "tabla" in resultado else resultado
    
    df = pd.DataFrame({
        "alpha": list(tabla.keys()),
        "MRR": list(tabla.values())
    }).sort_values("alpha")
    
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="alpha", y="MRR", marker="o")
    plt.title("Optimización de alpha (Hybrid)")
    plt.xlabel("Alpha (peso semántico - embeddings)")
    plt.ylabel("MRR@5")
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, "alpha_vs_mrr.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[FIG] Guardado: {path}")
    
    
# =========================================================
# 🟢 Heatmap de Correlación entre Modelos
# =========================================================

def plot_correlacion_modelos(e_t, e_e, e_h):
    """Heatmap de correlación de RR entre modelos."""
    _ensure_dir()
    
    # Extraer RR por query
    rr_data = {
        "TF-IDF": [q["rr"] for q in e_t["por_query"]],
        "Embeddings": [q["rr"] for q in e_e["por_query"]],
        "Hybrid": [q["rr"] for q in e_h["por_query"]]
    }
    
    df_corr = pd.DataFrame(rr_data)
    corr_matrix = df_corr.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", 
                center=0, vmin=-1, vmax=1, 
                square=True, fmt=".3f", linewidths=2)
    plt.title("Correlación de RR entre modelos", fontsize=14)
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, "correlacion_modelos.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[FIG] Guardado: {path}")

 
# =========================================================
# 🟢 Precision@k vs Recall@k (Curva de Compromiso)
# =========================================================

def plot_precision_vs_recall(modelos_con_metricas, ks=[1,3,5,10]):
    """
    modelos_con_metricas: dict con nombre -> dict con metricas por k
    Ej: {"TF-IDF": {1: {"p":0.5, "r":0.3}, 3: {...}}}
    """
    _ensure_dir()
    
    plt.figure(figsize=(10, 8))
    
    for nombre, metricas_por_k in modelos_con_metricas.items():
        precisions = [metricas_por_k[k]["precision"] for k in ks if k in metricas_por_k]
        recalls = [metricas_por_k[k]["recall"] for k in ks if k in metricas_por_k]
        
        plt.plot(recalls, precisions, 'o-', label=nombre, linewidth=2, markersize=8)
        
        # Anotar los valores de k
        for i, k in enumerate(ks):
            if k in metricas_por_k:
                plt.annotate(f'k={k}', (recalls[i], precisions[i]), 
                            xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Recall@k")
    plt.ylabel("Precision@k")
    plt.title("Precision-Recall Trade-off", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, "precision_vs_recall.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[FIG] Guardado: {path}")


# =========================================================
# 🟢 Resumen Ejecutivo Automático
# =========================================================

def generar_resumen_ejecutivo(e_t, e_e, e_h, output_file="resumen_ejecutivo.md"):
    _ensure_dir()
    path = os.path.join(OUTPUT_DIR, output_file)
    with open(path, "w", encoding="utf-8") as f:
        
        f.write("# Resumen Ejecutivo - Evaluación de Retrievers\n\n")
        f.write(f"## Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## 1. Comparación Global\n\n")
        f.write("| Métrica | TF-IDF | Embeddings | Hybrid | Mejor |\n")
        f.write("|---------|--------|------------|--------|-------|\n")
        for metrica in ["precision@k", "hit_rate@k", "recall@k", "MRR", "F1@k"]:
            tv = e_t["metricas"].get(metrica)
            ev = e_e["metricas"].get(metrica)
            hv = e_h["metricas"].get(metrica)
            if tv is None and ev is None and hv is None:
                continue
            vals = [("TF-IDF", tv or 0), ("Embeddings", ev or 0), ("Hybrid", hv or 0)]
            mejor = max(vals, key=lambda x: x[1])[0]
            tv_s = f"{tv:.3f}" if tv is not None else "N/A"
            ev_s = f"{ev:.3f}" if ev is not None else "N/A"
            hv_s = f"{hv:.3f}" if hv is not None else "N/A"
            f.write(f"| {metrica} | {tv_s} | {ev_s} | {hv_s} | {mejor} |\n")
        
        f.write("\n## 2. Mejora del Modelo Híbrido\n\n")
        mrr_t = e_t["metricas"]["MRR"]
        mrr_e = e_e["metricas"]["MRR"]
        mrr_h = e_h["metricas"]["MRR"]
        if mrr_t > 0:
            f.write(f"- **Mejora vs TF-IDF**: {(mrr_h - mrr_t)/mrr_t*100:+.1f}% en MRR\n")
        if mrr_e > 0:
            f.write(f"- **Mejora vs Embeddings**: {(mrr_h - mrr_e)/mrr_e*100:+.1f}% en MRR\n")
        
        f.write("\n## 3. Queries con Peor Rendimiento (Hybrid)\n\n")
        peores = sorted(e_h["por_query"], key=lambda x: x["rr"])[:3]
        for i, q in enumerate(peores, 1):
            f.write(f"{i}. **{q['query']}** (RR={q['rr']:.3f}, F1={q['f1']:.3f})\n")
        
        f.write("\n## 4. Recomendaciones\n\n")
        mejora = (mrr_h - mrr_t) / mrr_t * 100 if mrr_t > 0 else 0
        if mejora > 10:
            f.write("- ✅ **Usar modelo híbrido**: Supera significativamente a TF-IDF y Embeddings\n")
        else:
            f.write("- ⚠️ **Evaluar alpha**: La mejora es marginal, probar otros pesos\n")
        if e_h["metricas"].get("MRR_lexica", 1) < 0.5:
            f.write("- 🔹 **Mejorar queries léxicas**: Considerar stemming o expansión\n")
        if e_h["metricas"].get("MRR_semantica", 1) < 0.5:
            f.write("- 🔹 **Mejorar queries semánticas**: Revisar calidad de embeddings\n")
    
    print(f"[REPORT] Generado: {path}")