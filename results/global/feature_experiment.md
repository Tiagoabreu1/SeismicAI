# Experimento: mais variáveis ajudam? (perfil global, teste)

47 variáveis base + 22 novas. Mesmos splits temporais; gradient boosting com early stopping.

| Conjunto | Modelo | nº features | ROC-AUC | PR-AUC | Brier | PR-AUC c/ M≥big recente | PR-AUC células quietas |
|---|---|---|---|---|---|---|---|
| base | regressão logística | 47 | 0.8098 | 0.1819 | 0.0265 | 0.464 | 0.018 |
| base | gradient boosting | 47 | 0.8175 | 0.1861 | 0.0263 | 0.465 | 0.020 |
| base + extras | regressão logística | 69 | 0.8135 | 0.1854 | 0.0264 | 0.470 | 0.021 |
| base + extras | gradient boosting | 69 | 0.8176 | 0.1890 | 0.0263 | 0.486 | 0.019 |
| só extras | regressão logística | 22 | 0.8086 | 0.1809 | 0.0264 | 0.469 | 0.020 |
| só extras | gradient boosting | 22 | 0.8108 | 0.1799 | 0.0264 | 0.452 | 0.023 |

**Importância por permutação (gradient boosting, base + extras; queda de ROC-AUC)** — as 15 maiores; (*) = variável nova

| Variável | Queda de ROC-AUC |
|---|---|
| rate_target_train | +0.1440 |
| etas_cell (*) | +0.0040 |
| days_since_any (*) | +0.0039 |
| etas_viz (*) | +0.0035 |
| wsl_any | +0.0027 |
| n_104w (*) | +0.0025 |
| rate_any_train | +0.0025 |
| n_52w | +0.0016 |
| n_260w (*) | +0.0016 |
| meandepth_52w | +0.0015 |
| nseg_1w (*) | +0.0013 |
| meandepth_4w | +0.0011 |
| nviz_52w | +0.0010 |
| logE_52w | +0.0010 |
| maxmag_104w (*) | +0.0008 |

Novas variáveis entre as 15 mais importantes: etas_cell, days_since_any, etas_viz, n_104w, n_260w, nseg_1w, maxmag_104w.
