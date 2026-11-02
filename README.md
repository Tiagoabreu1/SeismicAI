# SeismicAI Pulse — probabilidade semanal de atividade sísmica por região

Projeto da disciplina de Redes Neurais e Deep Learning (2026). Alice De Muylder, Valmir Cerutti e
Tiago Abreu.

**SeismicAI Pulse** é um protótipo de startup que usa o catálogo sísmico do USGS e redes neurais para
estimar, semana a semana, a probabilidade de terremotos de magnitude ≥ 4,0 em cada área do Círculo de
Fogo do Pacífico.

> **Aviso.** Finalidade acadêmica e experimental. O SeismicAI Pulse **não** é um sistema de alerta nem
> faz previsão determinística de terremotos. Os resultados são estimativas estatísticas produzidas a
> partir dos dados e das hipóteses descritas aqui e não substituem as instituições responsáveis pelo
> monitoramento sísmico.

## 1. Definição do problema (Fase 1)

A definição completa, com motivação, hipóteses, diagnóstico de completude do catálogo, exploração dos
dados e baselines, está no notebook `notebooks/01_fase1_observacoes.ipynb`. Resumo:

| Item | Definição |
|---|---|
| **Pergunta** | Uma rede neural, treinada com o histórico sísmico recente de uma área e de seus vizinhos, estima a probabilidade de ocorrer ≥ 1 terremoto M ≥ 4,0 nessa área na semana seguinte melhor do que estimativas estatísticas simples? |
| **Tipo de problema** | Classificação binária probabilística sobre séries temporais, com classes desbalanceadas (~16 % de positivos). |
| **Público-alvo** | Seguradoras e resseguradoras; defesa civil e operadores de infraestrutura crítica (energia, portos, mineração); jornalismo e divulgação científica. |
| **Região** | Círculo de Fogo do Pacífico, delimitado por 12 caixas de latitude/longitude, dividido em células de 2° × 2°. Entram as células com ≥ 10 eventos M ≥ 4,0 no período de treino (709 células). |
| **Período** | 01/01/2000 a 31/12/2025, agregado em semanas (segunda a domingo, UTC). |
| **Fonte** | USGS ComCat, via FDSN Event Web Service (formato CSV). |
| **Unidade de amostra** | Par (célula, semana *t*): entrada = semanas *t−51 … t*; rótulo = semana *t+1*. |
| **Saída** | `y = 1` se houve ≥ 1 evento M ≥ 4,0 na célula na semana *t+1*. |
| **Divisão** | Temporal, pela semana do rótulo: treino 2001–2017, validação 2018–2020, teste 2021–2025. 2000 serve só de histórico inicial. |
| **Limpeza** | só `type = earthquake`; sem duplicatas de `id`; sem nulos de tempo, posição, profundidade ou magnitude; M ≥ 4,0; células ativas e padronização definidas só com o treino. |

**Entradas da LSTM** (sequência de 52 semanas × 10 variáveis, do painel semanal): nº de eventos,
magnitude máxima e média, log-energia liberada, profundidade média, nº de eventos M ≥ 5 (grandes),
nº M ≥ 6, semanas desde o último evento (teto 260), nº de eventos e nº de eventos grandes nas 8 células
vizinhas. Mais 5 features estáticas da célula: latitude, cos/sin da longitude, taxa histórica de semanas
com evento (treino) e média de eventos por semana (treino).

**Entradas da MLP** (45 features tabulares): as mesmas quantidades agregadas em janelas de 1, 4, 12 e
52 semanas, as semanas desde o último evento (qualquer / grande), o nº de semanas com evento nas
últimas 52 e as features estáticas.

## 2. Fonte dos dados

- https://earthquake.usgs.gov/data/comcat/
- https://earthquake.usgs.gov/fdsnws/event/1/
- https://earthquake.usgs.gov/earthquakes/feed/ (feeds em tempo real, usados pelo protótipo)

## 3. Como reproduzir

```bash
pip install -r requirements.txt            # torch: com GPU NVIDIA, instale a versão CUDA (pytorch.org)

python src/download_usgs.py                # catálogo global M >= 4,0 desde 2000 -> data/raw/usgs_m40_*.csv
python src/build_dataset.py                # perfil pulse -> data/processed/*_pulse.*
python src/train_baselines.py              # persistência, climatologia, taxa recente, combinado, reg. logística
python src/train_nn.py --model mlp         # usa a GPU automaticamente se houver
python src/train_nn.py --model mlp_seq     # MLP com as mesmas entradas da LSTM, achatadas
python src/train_nn.py --model lstm
python src/train_rbm.py                    # RBM: features não supervisionadas + reg. logística; energia livre
python src/train_hopfield.py --model lstm  # Hopfield: camada de saída (NORMAL / ATENÇÃO / ALARME, alarme = M >= 5,0)
python src/make_report.py                  # results/pulse/results_table.md e figuras
python src/analysis_strata.py              # results/pulse/strata_table.md e calibration_table.md
python src/experiment_features.py          # mais variáveis ajudam? -> results/pulse/feature_experiment.md
python src/study_aftershocks.py            # picos = réplicas? produtividade, Omori, Båth -> results/pulse/aftershocks.md
python src/study_cycles.py                 # ciclo sísmico? intervalos x magnitudes, tendências -> results/pulse/cycles.md
python src/experiment_sweep.py             # GPU: 5 sementes + varredura da LSTM + seq. 104 + pos_weight -> results/pulse/sweep.md
```

Ordem obrigatória: `build_dataset` antes de tudo; `train_nn` (ou `train_baselines`) antes de
`train_hopfield`, que usa as probabilidades de validação e teste do modelo escolhido (`--model`).

O notebook da Fase 1 reaproveita `data/raw/usgs_m40_min.csv` se ele existir, e seu dataset
(`data/processed/rof_dataset.npz`) é idêntico ao do perfil `pulse` (mesmas células, amostras e
baselines; verificado).

## 4. Estrutura do repositório

```
seismicai/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_fase1_observacoes.ipynb   # Fase 1 do grupo: definição, limpeza, completude, exploração, baselines
├── src/
│   ├── download_usgs.py     # coleta do catálogo (USGS FDSN)
│   ├── build_dataset.py     # limpeza, grade, agregação semanal, vizinhos, features, rótulo, splits (perfis)
│   ├── seismic_data.py      # carga e padronização dos dados (compartilhado)
│   ├── metrics.py           # métricas de avaliação (compartilhado)
│   ├── train_baselines.py   # baselines estatísticos + regressão logística
│   ├── train_nn.py          # MLP, MLP-seq e LSTM em PyTorch
│   ├── train_rbm.py         # Máquina de Boltzmann Restrita (CD-k) + regressão logística / energia livre
│   ├── train_hopfield.py    # Rede de Hopfield: memória associativa de estados de alerta (saída do projeto)
│   ├── analysis_strata.py   # análise por estratos e calibração
│   ├── make_report.py       # tabela comparativa e figuras
│   ├── predict.py           # motor de inferência do protótipo (features de qualquer semana + modelos + feed USGS)
│   └── publish_firebase.py  # publica as previsões no Cloud Firestore (ou em JSON local, --dry-run)
├── ENTREGAS.md              # mapa das sete entregas pedidas
├── data/README.md           # como obter os dados; data/raw e data/processed não são versionados
├── models/{perfil}/         # modelos treinados
├── results/{perfil}/        # métricas, predições do teste, figuras; results/eda/ = figuras do notebook
├── app/app.py               # protótipo Streamlit (uso local / demonstração)
├── web/                     # site público: Firebase Hosting + Firestore (ver web/README.md)
│   ├── public/              # index.html, app.js, style.css, firebase-config.js
│   ├── firebase.json, .firebaserc, firestore.rules
├── secrets/                 # chave da conta de serviço do Firebase (não versionada)
└── docs/                    # artigo e apresentação (Fase 3)
```

## 5. Fase 2 — Redes neurais

| Modelo | Entrada | Arquitetura | Parâmetros |
|---|---|---|---|
| Persistência (ref.) | última semana | p = 1 se houve evento na célula na semana *t* | — |
| Climatologia (ref.) | taxa histórica | p = fração de semanas de treino com evento na célula | — |
| Taxa recente (ref.) | últimas 52 semanas | p = fração das últimas 52 semanas com evento | — |
| Combinado (ref.) | — | média da climatologia com a taxa recente (melhor baseline da Fase 1) | — |
| Regressão logística (ref.) | 45 features | modelo linear + sigmoide | 46 |
| **MLP** | 45 features | Dense 128 → BatchNorm → ReLU → Dropout 0,2 → Dense 64 → BN → ReLU → Dropout → Dense 32 → BN → ReLU → Dropout → Dense 1 → sigmoide | ~17 mil |
| **MLP-seq** | 52 × 10 + 5, achatadas | Dense 256 → BN → ReLU → Dropout → Dense 64 → BN → ReLU → Dropout → Dense 1 → sigmoide | ~150 mil |
| **LSTM** | sequência 52 × 10 + 5 estáticas | LSTM(64) → concatena estáticas → Dense 64 → ReLU → Dropout 0,2 → Dense 1 → sigmoide | ~24 mil |
| **RBM + reg. logística** | 45 features em [0, 1] | RBM Bernoulli 45 visíveis × 64 ocultas, treinada sem rótulos por divergência contrastiva (CD-1, 10 épocas); h = sigmoid(Wx + c) → regressão logística | ~3 mil + 65 |
| **RBM (energia livre)** | idem | F(x) = −b·x − Σ softplus(Wx + c), calibrada por Platt na validação: score 100 % não supervisionado | — |
| **Hopfield (saída)** | 26 bits: limiares sobre p do modelo, atividade recente, vizinhos e magnitude | memória associativa com 3 protótipos (NORMAL, ATENÇÃO, ALARME), regra de projeção, dinâmica assíncrona até o atrator | 26 × 26 |

**Rede de Hopfield como saída do projeto.** O modelo probabilístico (LSTM) entrega `p`; a rede de
Hopfield transforma `p` e o estado recente da célula em um vetor bipolar de 26 bits e o deixa relaxar
até um dos três atratores. O **limiar de magnitude** (padrão 5,0) define o ALARME: a memória de
alarme é aprendida só nas semanas de alarme com *assinatura* (evento M ≥ 5 recente na célula ou nos
vizinhos), porque um terremoto grande sem precursores no catálogo não tem padrão a ser memorizado.
O alarme só é emitido quando a rede converge exatamente para o protótipo de ALARME. O script
compara o alarme com uma regra simples sobre `p` no mesmo ponto de operação (mesma fração de
alarmes): a Hopfield não supera um limiar bem escolhido (uma memória binária não tem como), mas
entrega estados discretos, legíveis (os protótipos são listas de bits ativos) e robustos a bits
faltantes — e é o que o protótipo do produto mostra no mapa. Saídas: `results/{perfil}/hopfield_table.md`,
`fig_hopfield_*.png`, `preds_hopfield.parquet` e a memória em `models/{perfil}/hopfield.json`.

**Treinamento.** Padronização (log1p nas contagens + z-score) ajustada só no treino; perda binary
cross-entropy (sem ponderação, para manter as probabilidades calibradas; `--pos-weight` disponível, com
correção do logit); Adam (lr 1e-3, weight decay 1e-5), lotes de 2.048, clipping de gradiente em 5,
`ReduceLROnPlateau` e early stopping pela PR-AUC de validação (paciência 5, máx. 30 épocas). Limiar de
classificação = máximo F1 na validação, aplicado ao teste. `--max-train N` subamostra os negativos do
treino (útil em CPU) e corrige o logit com log(taxa de negativos mantidos), preservando a calibração.

**Métricas.** ROC-AUC, PR-AUC (principal, pelo desbalanceamento), Brier score, log loss, precision /
recall / F1 no limiar da validação, precision / recall ao sinalizar os 5 % de (célula, semana) com maior
probabilidade, ROC-AUC por ano do teste, importância por permutação, análise por estratos (semanas após
eventos grandes, vizinhos ativos, células quietas vs. ativas) e tabela de calibração.

**Meta (definida na Fase 1):** superar o baseline Combinado no teste — ROC-AUC > 0,836 e PR-AUC > 0,552 —
com boa calibração.

## 6. Apêndice — formulação global M ≥ 5,0 (perfil `global`)

Antes de alinhar o pipeline à Fase 1 do grupo, o mesmo código foi aplicado a uma formulação mais
difícil: M ≥ 5,0 em células de 3° no mundo todo, 1990–2026, 770 células, ~3 % de positivos
(`python src/download_usgs.py --start 1990-01-01 --minmag 4.5` e `--profile global` nos demais scripts).
Resultados em `results/global/` (baselines, MLP, RBM e Hopfield; a LSTM e a MLP-seq ficam para a
GPU). Achados que valem para o artigo:

- o teto dessa formulação é ROC-AUC ≈ 0,82 (gradient boosting dá o mesmo; features de vizinhança e
  resolução diária não movem o número): a MLP (0,818 / PR-AUC 0,187) já está nele, contra 0,803 / 0,140
  da climatologia;
- o ganho da rede está onde a AUC agregada não mostra: após um M ≥ 6 na célula, a PR-AUC dobra
  (0,236 → 0,472) e as probabilidades ficam calibradas até ~0,75, enquanto a climatologia satura em 0,32;
- a energia livre da RBM, treinada sem nenhum rótulo, atinge ROC-AUC 0,78: a RBM aprende sozinha que
  semana "atípica" é semana de risco; com a regressão logística sobre as ocultas, 0,807;
- na Hopfield, 73 % dos alarmes M ≥ 6 não têm assinatura (nenhum M ≥ 6 recente na célula ou nos
  vizinhos) — nenhum reconhecedor de padrões os antecipa; nos que têm, o alarme da rede recupera 38 %.

## 7. Fase 3 — Protótipo (app/app.py)

```bash
streamlit run app/app.py          # da raiz do projeto, com data/, models/ e results/ gerados
```

O app abre no navegador (http://localhost:8501) e mostra, para cada célula ativa do Círculo de Fogo, a
probabilidade de evento M ≥ 4,0 na semana seguinte (modelo escolhido na barra lateral; LSTM por padrão)
e o estado de alerta da rede de Hopfield (NORMAL / ATENÇÃO / ALARME, alarme associado a M ≥ 5,0).

- **Modo "Ao vivo"**: usa o catálogo do projeto (`data/raw/usgs_m40_min.csv`) e calcula as features da
  última semana completa com o mesmo código do treino (`src/predict.py`). O botão *Atualizar catálogo
  pelo feed do USGS* baixa os eventos novos e refaz a previsão para a semana corrente. Mostra também os
  eventos da última semana, a tabela das células mais prováveis e o detalhe de uma célula (eventos por
  semana e probabilidade prevista nos últimos 2 anos).
- **Modo "Histórico (teste)"**: percorre as semanas de 2021–2025 com as probabilidades gravadas no
  treino e sobrepõe o que de fato aconteceu na semana seguinte (acertos no top 5 %, alarmes seguidos de
  M ≥ 5,0).
- Painéis com a tabela de métricas, as figuras do relatório e a documentação de funcionamento,
  limitações e uso responsável.

`src/predict.py` também roda em linha de comando (`python src/predict.py --model lstm --feed`) e
reproduz exatamente as probabilidades gravadas no treino (verificado: diferença máxima de 1e-9).

## 8. Produto no Firebase (Cloud Firestore + Hosting)

O pipeline publica as previsões no **Cloud Firestore** e um site estático no **Firebase Hosting** as
exibe — a arquitetura de produto: o motor (Python) roda onde for conveniente e o cliente lê só o resultado.

```
USGS (feed) → src/predict.py (LSTM + Hopfield) → src/publish_firebase.py → Firestore
                                                       weeks/{semana}: 709 células (p, estado, atividade) + eventos
                                                       meta/latest: semana corrente, lista de semanas, métricas
                                                                        └→ web/public (Firebase Hosting)
```

```bash
pip install firebase-admin
python src/publish_firebase.py --dry-run --backfill-test 20   # teste local: web/public/data/*.json
python src/publish_firebase.py --feed --backfill-test 52      # publica no Firestore (chave em secrets/serviceAccount.json)
cd web && firebase deploy --only firestore:rules,hosting        # publica o site
```

Passo a passo completo (criar projeto, chave da conta de serviço, config do app Web, deploy) em
`web/README.md`. Cada semana é um documento de ~100 KB; o site faz uma leitura por semana e as regras
do Firestore permitem só leitura (a escrita é feita pelo pipeline com o SDK Admin).

## 9. Entregas finais

O mapa das sete entregas pedidas no enunciado está em `ENTREGAS.md`.

## 10. Documentos finais (`docs/`)

| Arquivo | O que é |
|---|---|
| `docs/analise_resultados.md` | Análise crítica dos resultados do teste (tabela principal, estratos, calibração, importâncias, RBM, Hopfield, limitações) |
| `docs/artigo.md` / `artigo.docx` / `artigo.pdf` | Artigo de divulgação da startup (9 seções do enunciado, 5 páginas) |
| `docs/apresentacao.html` | Apresentação final: arquivo único, abre em qualquer navegador, teclado/toque, notas (`N`), impressão, demo do site embutida |
| `docs/fig/` | Figuras usadas no artigo e na apresentação (cópias de `results/pulse/`) |

Regerar: `node docs/build_artigo_docx.js` (precisa de `docx` e `image-size` via npm) e
`python docs/build_apresentacao.py` (precisa de `segno` e `Pillow`).

## 11. Resultados finais (perfil pulse, teste 2021–2025)

LSTM: ROC-AUC 0,841 · PR-AUC 0,571 · Brier 0,1012 · precisão no top 5 % 76 % — acima da meta da
Fase 1 (Combinado: 0,836 / 0,552). MLP 0,841 / 0,569; regressão logística 0,839 / 0,565; RBM (energia
livre, sem rótulos) 0,792 / 0,402. Tabelas e figuras completas em `results/pulse/`.
