### Rede de Hopfield — saída de alerta (teste, alarme = M ≥ 6, modelo mlp)

Protótipos: 3 (NORMAL×1, ATENÇÃO×1, ALARME×1); regra projection; 26 bits; acurácia 0.788; estados espúrios 14.9%; varreduras até convergir (média) 0.5.


**Matriz de confusão (linhas = real, colunas = Hopfield)**

| real \ Hopfield | NORMAL | ATENÇÃO | ALARME |
|---|---|---|---|
| NORMAL | 144,126 | 27,623 | 2,262 |
| ATENÇÃO | 7,517 | 8,003 | 3,216 |
| ALARME | 183 | 233 | 107 |

**Por estado**

| Estado | Taxa real | Precision | Recall | F1 |
|---|---|---|---|---|
| NORMAL | 0.900 | 0.949 | 0.828 | 0.885 |
| ATENÇÃO | 0.097 | 0.223 | 0.427 | 0.293 |
| ALARME | 0.003 | 0.019 | 0.205 | 0.035 |

**Alarme (M ≥ 6 na semana seguinte; taxa base 0.003)**

| Decisor | Precision | Recall | F1 | Fração de alarmes |
|---|---|---|---|---|
| hopfield | 0.019 | 0.205 | 0.035 | 0.029 |
| regra p>=0.311 (máx. F1 na validação) | 0.034 | 0.080 | 0.048 | 0.006 |
| regra p>=0.148 (mesma fração de alarmes) | 0.024 | 0.266 | 0.043 | 0.031 |

**Só semanas com assinatura** (evento M ≥ 6 recente na célula ou nos vizinhos; n = 15,355; 27% dos alarmes reais estão aqui; taxa base 0.009)

| Decisor | Precision | Recall | F1 |
|---|---|---|---|
| hopfield | 0.021 | 0.385 | 0.040 |
| regra p>=0.311 (máx. F1 na validação) | 0.041 | 0.196 | 0.068 |
| regra p>=0.148 (mesma fração de alarmes) | 0.030 | 0.566 | 0.058 |

**Protótipos (bits ativos, +1)**

- NORMAL 1: 
- ATENÇÃO 2: nviz_1w ≥ 1, nviz_4w ≥ 5, rate_target_train ≥ 0.0123203, rate_target_train ≥ 0.0342231
- ALARME 3: p ≥ 0.15, n_1w ≥ 1, n_4w ≥ 2, n_12w ≥ 10, n_52w ≥ 25, maxmag_12w ≥ 6, wsl_big ≤ 8, nviz_1w ≥ 1, nviz_4w ≥ 5, rate_target_train ≥ 0.0123203, rate_target_train ≥ 0.0342231, logE_4w ≥ 0.892854
