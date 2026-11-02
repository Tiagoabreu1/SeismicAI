# Dados

Os dados não são versionados (o catálogo bruto tem ~50 MB e muda a cada download). Para obtê-los:

```bash
python src/download_usgs.py            # ~3 min; catálogo global M >= 4,0 desde 2000-01-01
python src/build_dataset.py            # ~10 s; dataset de modelagem (perfil pulse)
```

**Fonte.** USGS ComCat (Comprehensive Earthquake Catalog), via FDSN Event Web Service:
`https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=...&endtime=...&minmagnitude=4.0`
(o script baixa em blocos anuais e divide o bloco quando passa de 20.000 eventos).

**Arquivos gerados**

| Arquivo | Conteúdo |
|---|---|
| `raw/usgs_m40_full.csv` | todas as colunas do USGS (tempo, posição, profundidade, magnitude, tipo, id, fontes, erros…) |
| `raw/usgs_m40_min.csv` | só as colunas usadas: time, latitude, longitude, depth, mag, magType, type, id |
| `processed/events_clean_pulse.parquet` | catálogo limpo com célula e semana |
| `processed/dataset_pulse.parquet` | uma linha por (célula, semana): 45 features, rótulo, split |
| `processed/panel_pulse.parquet` | 10 variáveis semanais por célula (sequências da LSTM) |
| `processed/cells_pulse.csv` | 709 células ativas: posição, segmento, taxa histórica |
| `processed/summary_pulse.json` | parâmetros do perfil e contagens |

**Contagens esperadas (download de novembro de 2026).** 358.672 eventos M ≥ 4,0 de 2000 a 2025 no
mundo; 709 células ativas no Círculo de Fogo; 629.592 / 111.313 / 184.340 amostras de treino /
validação / teste, com 14,5 % / 15,9 % / 16,6 % de positivos. Pequenas diferenças são normais: o USGS
revisa magnitudes e localizações continuamente.

**Limpeza.** Só `type = earthquake`; sem duplicatas de `id`; sem nulos de tempo, posição, profundidade
ou magnitude; M ≥ 4,0; profundidade negativa truncada em 0; eventos atribuídos a células de 2° × 2° e a
semanas (segunda a domingo, UTC). Detalhes em `README.md` (seção 1) e no notebook da Fase 1.
