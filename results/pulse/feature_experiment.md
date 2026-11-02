# Experimento: mais variáveis ajudam? (perfil pulse, teste)

45 variáveis base + 22 novas. Mesmos splits temporais; gradient boosting com early stopping.

| Conjunto | Modelo | nº features | ROC-AUC | PR-AUC | Brier | PR-AUC c/ M≥big recente | PR-AUC células quietas |
|---|---|---|---|---|---|---|---|
| base | regressão logística | 45 | 0.8386 | 0.5650 | 0.1018 | 0.746 | 0.068 |
| base | gradient boosting | 45 | 0.8412 | 0.5694 | 0.1012 | 0.750 | 0.066 |
| base + extras | regressão logística | 67 | 0.8404 | 0.5671 | 0.1014 | 0.747 | 0.071 |
| base + extras | gradient boosting | 67 | 0.8420 | 0.5707 | 0.1011 | 0.752 | 0.073 |
| só extras | regressão logística | 22 | 0.8369 | 0.5566 | 0.1027 | 0.739 | 0.070 |
| só extras | gradient boosting | 22 | 0.8389 | 0.5637 | 0.1019 | 0.744 | 0.071 |

**Importância por permutação (gradient boosting, base + extras; queda de ROC-AUC)** — as 15 maiores; (*) = variável nova

| Variável | Queda de ROC-AUC |
|---|---|
| rate_target_train | +0.0627 |
| wktarget_52w | +0.0077 |
| n_104w (*) | +0.0026 |
| n_12w | +0.0017 |
| n_52w | +0.0013 |
| n_4w | +0.0011 |
| meandepth_52w | +0.0006 |
| etas_cell (*) | +0.0004 |
| n_260w (*) | +0.0004 |
| days_since_any (*) | +0.0004 |
| nviz_4w | +0.0003 |
| nvizbig_52w | +0.0003 |
| mean_n_train | +0.0002 |
| etas_viz (*) | +0.0002 |
| n_1w | +0.0002 |

Novas variáveis entre as 15 mais importantes: n_104w, etas_cell, n_260w, days_since_any, etas_viz.
