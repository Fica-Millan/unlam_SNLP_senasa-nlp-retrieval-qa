# Resumen Ejecutivo - Evaluación de Retrievers

## Fecha: 2026-04-26 19:37

## 1. Comparación Global

| Métrica | TF-IDF | Embeddings | Hybrid | Mejor |
|---------|--------|------------|--------|-------|
| precision@k | 0.187 | 0.147 | 0.207 | Hybrid |
| hit_rate@k | 0.733 | 0.533 | 0.800 | Hybrid |
| recall@k | 0.558 | 0.423 | 0.593 | Hybrid |
| MRR | 0.484 | 0.397 | 0.561 | Hybrid |
| F1@k | 0.259 | 0.202 | 0.284 | Hybrid |

## 2. Mejora del Modelo Híbrido

- **Mejora vs TF-IDF**: +15.7% en MRR
- **Mejora vs Embeddings**: +41.3% en MRR

## 3. Queries con Peor Rendimiento (Hybrid)

1. **objetivo comisiones sanidad bienestar animal** (RR=0.000, F1=0.000)
2. **vacunación antibrucélica obligatoria hembras bovinas** (RR=0.000, F1=0.000)
3. **¿Qué animales deben ser identificados electrónicamente según la normativa vigente?** (RR=0.000, F1=0.000)

## 4. Recomendaciones

- ✅ **Usar modelo híbrido**: Supera significativamente a TF-IDF y Embeddings
- 🔹 **Mejorar queries semánticas**: Revisar calidad de embeddings
