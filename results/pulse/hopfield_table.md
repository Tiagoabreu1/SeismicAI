### Rede de Hopfield — saída de alerta (teste, alarme = M ≥ 5, modelo lstm)

Protótipos: 3 (NORMAL×1, ATENÇÃO×1, ALARME×1); regra projection; 26 bits; acurácia 0.747; estados espúrios 10.7%; varreduras até convergir (média) 0.9.


**Matriz de confusão (linhas = real, colunas = Hopfield)**

| real \ Hopfield | NORMAL | ATENÇÃO | ALARME |
|---|---|---|---|
| NORMAL | 126,420 | 14,803 | 12,433 |
| ATENÇÃO | 8,758 | 10,173 | 7,473 |
| ALARME | 1,257 | 1,831 | 1,192 |

**Por estado**

| Estado | Taxa real | Precision | Recall | F1 |
|---|---|---|---|---|
| NORMAL | 0.834 | 0.927 | 0.823 | 0.872 |
| ATENÇÃO | 0.143 | 0.379 | 0.385 | 0.382 |
| ALARME | 0.023 | 0.056 | 0.279 | 0.094 |

**Alarme (M ≥ 5 na semana seguinte; taxa base 0.023)**

| Decisor | Precision | Recall | F1 | Fração de alarmes |
|---|---|---|---|---|
| hopfield | 0.056 | 0.279 | 0.094 | 0.114 |
| regra p>=0.504 (máx. F1 na validação) | 0.113 | 0.336 | 0.169 | 0.069 |
| regra p>=0.381 (mesma fração de alarmes) | 0.092 | 0.483 | 0.155 | 0.121 |

**Só semanas com assinatura** (evento M ≥ 5 recente na célula ou nos vizinhos; n = 86,101; 75% dos alarmes reais estão aqui; taxa base 0.038)

| Decisor | Precision | Recall | F1 |
|---|---|---|---|
| hopfield | 0.058 | 0.338 | 0.099 |
| regra p>=0.504 (máx. F1 na validação) | 0.117 | 0.417 | 0.183 |
| regra p>=0.381 (mesma fração de alarmes) | 0.097 | 0.585 | 0.167 |

**Protótipos (bits ativos, +1)**

- NORMAL 1: nviz_1w ≥ 1
- ATENÇÃO 2: p ≥ 0.15, p ≥ 0.3, n_4w ≥ 2, nbig_52w ≥ 2, nviz_1w ≥ 1, nviz_4w ≥ 5, nvizbig_4w ≥ 1, rate_target_train ≥ 0.0797872, rate_target_train ≥ 0.239362, wktarget_52w ≥ 13
- ALARME 3: p ≥ 0.15, p ≥ 0.3, n_4w ≥ 2, n_52w ≥ 25, maxmag_12w ≥ 5, maxmag_52w ≥ 5.5, nbig_52w ≥ 2, wsl_big ≤ 8, nviz_1w ≥ 1, nviz_4w ≥ 5, nvizbig_4w ≥ 1, rate_target_train ≥ 0.0797872, rate_target_train ≥ 0.239362, wktarget_52w ≥ 13
