# SeismicAI Pulse — resumo do projeto

Documento de contexto para o grupo: o que foi feito, por quê, o que os números dizem e como apresentar.
Leitura de 15 minutos. Detalhes técnicos: `README.md`; análise completa: `docs/analise_resultados.md`.

---

## 1. Em um parágrafo

O SeismicAI Pulse é um protótipo de startup que responde, toda semana, à pergunta: **em quais áreas do
Círculo de Fogo do Pacífico é mais provável haver um terremoto de magnitude ≥ 4,0 na próxima semana?**
Ele divide a região em 709 células de 2° × 2°, lê o catálogo público do USGS, resume as últimas 52
semanas de cada célula e de suas vizinhas, e uma rede neural (LSTM) estima a probabilidade de evento
na semana seguinte. Uma rede de Hopfield transforma essa probabilidade em um estado de alerta
(NORMAL / ATENÇÃO / ALARME). O resultado é publicado em um site (Firebase) com um mapa interativo. No
teste (2021–2025), a LSTM atinge ROC-AUC 0,841 e PR-AUC 0,571, superando o melhor baseline
estatístico (0,836 / 0,552), com probabilidades calibradas.

## 2. O problema e por que importa

- Terremotos não podem ser previstos (dia, lugar e magnitude exatos). Mas a sismicidade **não é
  aleatória**: concentra-se em regiões conhecidas e se agrupa no tempo — um evento grande é seguido de
  réplicas por semanas, e a atividade se espalha para áreas vizinhas.
- O que existe hoje: catálogos de eventos (o que já aconteceu) e mapas de risco de longo prazo
  (décadas). Falta uma visão **de curto prazo e por área**: "nesta semana, onde a atenção deve estar?".
- Público-alvo: seguradoras e resseguradoras (exposição de curto prazo), defesa civil e operadores de
  infraestrutura crítica (energia, portos, mineração) para priorizar inspeções e prontidão, e
  jornalismo/divulgação científica.
- O que o produto **não** é: sistema de alerta, previsão determinística, substituto de órgãos oficiais.
  Isso está escrito no site, no app e no artigo, e precisa ser dito na apresentação.

## 3. As decisões e por que foram tomadas

| Decisão | Por quê |
|---|---|
| Alvo M ≥ 4,0 (e não 5 ou 6) | É a faixa sentida pela população e com dano leve; há muito mais eventos, logo mais exemplos positivos (16 % das amostras) e um problema menos desbalanceado. |
| Círculo de Fogo, 12 caixas | Concentra 74 % dos M ≥ 4 do mundo; dá foco ao produto. |
| Células de 2° × 2° e semanas | Compromisso entre resolução e número de eventos por célula. Semana = segunda a domingo, UTC. |
| Período 2000–2025 | A rede sismográfica global já estava consolidada; antes de 2000 faltam eventos M 4 em áreas remotas. |
| Divisão temporal: treino 2001–2017, validação 2018–2020, teste 2021–2025 | Sem vazamento do futuro; 2000 serve só de histórico inicial. Tudo que o modelo "sabe" de uma célula (taxa histórica, padronização) vem só do treino. |
| Réplicas mantidas | Fazem parte da dinâmica real de curto prazo; o quanto do desempenho vem delas é medido na análise por estratos. |
| Classificação binária **probabilística** | O produto usa a probabilidade, não só "sim/não". Por isso a perda é a entropia cruzada sem ponderação: ponderar inflaria as probabilidades. |
| Métricas principais: PR-AUC e ROC-AUC + calibração | Com 16 % de positivos, acurácia engana (dizer sempre "não" acerta 84 %). PR-AUC mede o ranking dos positivos; a calibração mede se "30 %" acontece em 30 % das vezes. |
| Baselines antes das redes | Para saber quanto a rede acrescenta de verdade. A meta da Fase 1 era superar o baseline Combinado. |

## 4. Os dados

- **Fonte**: USGS ComCat, via FDSN Event Web Service (CSV). Catálogo global M ≥ 4,0 de 2000 a 2025:
  **358.672 eventos**; 73,8 % dentro das caixas do Círculo de Fogo.
- **Limpeza**: só `type = earthquake`; sem duplicatas de id; sem nulos de tempo/posição/profundidade/
  magnitude; M ≥ 4,0; profundidade negativa truncada em 0.
- **Grade**: 709 células ativas (≥ 10 eventos no período de treino, centro dentro das caixas);
  1.358 semanas; amostras = células × semanas: **629.592 treino / 111.313 validação / 184.340 teste**,
  com 14,5 % / 15,9 % / 16,6 % de positivos.
- **Entradas por semana (sequência de 52, para a LSTM)**: nº de eventos, magnitude máxima e média,
  log-energia liberada, profundidade média, nº de eventos M ≥ 5, nº M ≥ 6, semanas desde o último
  evento, nº de eventos e nº de M ≥ 5 nas 8 células vizinhas. Mais 5 atributos fixos da célula:
  latitude, cos/sin da longitude, taxa histórica de semanas com evento (treino), média de eventos/semana.
- **Entradas da MLP**: as mesmas quantidades agregadas em janelas de 1, 4, 12 e 52 semanas (45 features).
- **Saída**: 1 se houve ≥ 1 evento M ≥ 4 na célula na semana seguinte.
- **Diagnóstico de completude** (Fase 1): pela lei de Gutenberg–Richter, a razão entre eventos
  M 4,0–4,4 e M ≥ 4,5 deveria ser ~2 e constante; ela cresce ao longo do tempo (sobretudo na
  Indonésia: 0,58 em 2000–2005 → 1,23 em 2015+). Isso significa que a detecção melhorou, e explica a
  subida da taxa de positivos de 14,5 % para 16,6 %. É uma limitação declarada.

## 5. O pipeline (scripts em `src/`)

1. `download_usgs.py` — baixa o catálogo em blocos anuais, dividindo quando passa de 20.000 eventos.
2. `build_dataset.py` — limpeza, grade, agregação semanal, vizinhos, features, rótulo, splits
   (perfil `pulse`). Reproduz exatamente o dataset do notebook da Fase 1 (verificado).
3. `train_baselines.py` — persistência, climatologia, taxa recente, combinado, regressão logística.
4. `train_nn.py --model mlp | mlp_seq | lstm` — redes em PyTorch, early stopping pela PR-AUC de validação.
5. `train_rbm.py` — RBM não supervisionada + regressão logística / energia livre.
6. `train_hopfield.py` — rede de Hopfield como camada de saída (estados de alerta).
7. `make_report.py` e `analysis_strata.py` — tabelas, figuras, estratos e calibração.
8. `predict.py` — motor de inferência do produto (mesmo código do treino; diferença de 10⁻⁹).
9. `publish_firebase.py` — publica as previsões no Cloud Firestore; `web/` é o site.

## 6. Os modelos, em linguagem simples

| Modelo | O que é | Por que está no projeto | Resultado (teste) |
|---|---|---|---|
| Persistência | "se teve esta semana, vai ter na próxima" | O baseline mais ingênuo | 0,646 / 0,266 |
| Climatologia | taxa histórica da célula no treino | Mede o valor de saber *onde* | 0,830 / 0,528 |
| Taxa recente | fração das últimas 52 semanas com evento | Mede o valor de saber *quando* | 0,831 / 0,541 |
| Combinado | média das duas anteriores | Meta da Fase 1 | 0,836 / 0,552 |
| Regressão logística | modelo linear sobre as 45 features | Referência linear | 0,839 / 0,565 |
| **MLP** | rede densa (128 → 64 → 32) sobre as 45 features de janela | Rede neural sobre resumos | 0,841 / 0,569 |
| **MLP-seq** | rede densa sobre a sequência achatada (525 entradas) | Mesmas entradas da LSTM, sem recorrência | 0,839 / 0,566 |
| **LSTM** | rede recorrente lendo as 52 semanas + atributos fixos | A arquitetura prevista para dado temporal | **0,841 / 0,571** |
| **RBM** | Máquina de Boltzmann Restrita, treinada **sem rótulos** | Aprendizado não supervisionado de features | energia: 0,792 / 0,402; + reg. logística: 0,836 / 0,554 |
| **Hopfield** | memória associativa com 3 protótipos (NORMAL / ATENÇÃO / ALARME) | Camada de saída: estado de alerta com limiar de magnitude | ALARME: precision 2,4 × a taxa base; um limiar simples faz melhor |

(Pares = ROC-AUC / PR-AUC.)

**Como cada rede funciona, em duas frases:**

- **MLP (perceptron multicamadas)**: camadas de neurônios totalmente conectadas, com ReLU, batch
  normalization e dropout; aprende combinações não lineares das features. Treina em 68 s na CPU.
- **LSTM**: rede recorrente que lê a sequência semana a semana mantendo um estado interno ("memória");
  o estado final é concatenado aos atributos fixos da célula e passa por uma camada densa até a
  probabilidade. 24 mil parâmetros, 26 épocas, 12 min na CPU.
- **RBM**: duas camadas (visível = features em [0,1]; oculta = 64 unidades binárias) ligadas por uma
  matriz de pesos, sem conexões dentro de cada camada; treinada por divergência contrastiva para que
  as configurações reais tenham energia baixa. A energia livre de uma semana mede quão "típica" ela é.
- **Hopfield**: rede com uma única camada de neurônios ±1 totalmente conectados por pesos simétricos;
  os protótipos armazenados são atratores. Dado um estado inicial (os 26 bits da célula), a dinâmica
  assíncrona "desce a energia" até o protótipo mais próximo — é uma memória que completa padrões.

## 7. Os sete números para saber de cor

1. **709 células, 358.672 eventos, 2000–2025**; teste com 184.340 amostras e 16,6 % de positivos.
2. **Meta da Fase 1**: superar o combinado, ROC-AUC 0,836 / PR-AUC 0,552.
3. **LSTM: 0,841 / 0,571**, melhor em todas as métricas; MLP 0,841 / 0,569; reg. logística 0,839 / 0,565.
4. **Top 5 %**: das 5 % células mais prováveis de cada semana, **76 %** têm terremoto M ≥ 4 na semana seguinte.
5. **Calibração**: quando a LSTM diz 83 %, acontece em 84 %; a climatologia diria 60 %.
6. **Onde a rede ganha**: após um M ≥ 5 na célula, PR-AUC 0,682 → 0,753; células quietas, PR-AUC ×2,7.

7. **Mais variáveis não ajudam**: 22 extras (memória longa, ETAS, diárias) → +0,001; o teto está no catálogo.

E a frase que resume a ciência: **"onde" vale dez vezes mais que "quando"** (importância da taxa
histórica 0,105 contra 0,059 da contagem recente), e a recorrência da LSTM não acrescenta sobre
estatísticas simples da sequência — por isso MLP e LSTM empatam.

## 8. O produto

- **Site público** (Firebase Hosting + Cloud Firestore): `https://deep-learning-terremoto.web.app`.
  Mapa do Pacífico com as 709 células coloridas pela probabilidade da LSTM e contornadas pelo estado
  da Hopfield (vermelho = ALARME, laranja = ATENÇÃO); seletor de semana (ao vivo e 52 semanas do
  teste, com "o que aconteceu"); ranking; detalhe ao clicar; métricas; aviso de uso responsável.
- **Como a semana ao vivo é produzida**: `python src/publish_firebase.py --feed` baixa os eventos novos
  do USGS, calcula as features da última semana completa com o mesmo código do treino, aplica a LSTM e
  a Hopfield e grava um documento no Firestore (~100 KB: 709 células + eventos). O site lê esse
  documento. Rodar toda semana (ou agendar) mantém o produto vivo sem novo deploy.
- **App Streamlit** (`streamlit run app/app.py`): a mesma visão, para uso local e demonstração, com o
  botão de atualização pelo feed, detalhe por célula com 2 anos de histórico e os painéis de métricas.
- **Papel de cada rede no produto**: LSTM = probabilidade; Hopfield = estado de alerta (o alarme só
  dispara em padrões com M ≥ 5 recente); RBM = score de atipicidade (coluna opcional no app).

## 9. Limitações e uso responsável

Dizer sempre, nesta ordem: (1) finalidade acadêmica e experimental; (2) não é sistema de alerta nem
previsão determinística; (3) os ganhos sobre a taxa histórica são reais, mas pequenos — a maior parte
da informação é "onde"; (4) o catálogo tem completude variável no espaço e no tempo; (5) células de 2°
e semanas são grosseiras; (6) um terremoto grande sem precursores não tem padrão reconhecível (25 %
dos M ≥ 5 do teste chegam sem assinatura); (7) não há física de falhas no modelo.

## 10. Mapa do repositório

```
README.md                      definição do problema, modelos, como rodar
notebooks/01_fase1_observacoes.ipynb   Fase 1 do grupo (exploração, completude, baselines)
src/                           pipeline completo (seção 5)
app/app.py                     protótipo Streamlit
web/                           site (Firebase Hosting) + regras do Firestore
results/pulse/                 métricas, tabelas, figuras, predições do teste
models/pulse/                  lstm.pt, mlp.pt, mlp_seq.pt, rbm.pt, hopfield.json, logreg.pkl
docs/                          este resumo, análise, artigo, apresentação
```

Como rodar do zero (CPU basta; ~30 min): `pip install -r requirements.txt`, depois os scripts da
seção 5 na ordem. Para o site: `web/README.md`.

## 11. Roteiro da apresentação (12–15 minutos + demo)

A apresentação (`docs/apresentacao.html`, 19 slides) segue o percurso que o enunciado pede:
**Problema → Dados → Rede Neural → Resultados → Produto**. Sugestão de divisão por pessoa (ajustem):

| Bloco | Slides | Tempo | O que dizer |
|---|---|---|---|
| Abertura e problema | 1–3 | 2 min | A pergunta, o público, o que existe e o que falta; "não prevemos terremotos, estimamos probabilidade por área". |
| Dados e preparação | 4–7 | 3,5 min | USGS, 358 mil eventos, 709 células, semanas, sem vazamento; completude como limitação; baselines e a meta (0,836 / 0,552). |
| Redes neurais | 8–9 | 3 min | Cinco redes e o papel de cada uma; como foi o treino (early stopping na validação, sem ponderação para manter calibração). |
| Resultados | 10–14 | 4,5 min | Tabela: LSTM supera a meta; onde o ganho acontece (estratos); calibração; "onde ≫ quando"; LSTM ≈ MLP e por que isso é resultado; RBM sem rótulo; Hopfield legível mas inferior a um limiar — e seu papel. |
| Produto e demo | 15–16 | 3 min | Arquitetura (pipeline → Firestore → site), demo ao vivo: semana corrente, trocar para uma semana do teste e mostrar "o que aconteceu", clicar numa célula em ALARME. |
| Limitações e fechamento | 17–19 | 1 min | Uso responsável; próximos passos. |

**Roteiro da demo (2 minutos):** abrir o site → apontar o selo "Firestore" e a semana corrente →
mostrar os KPIs (ALARME / ATENÇÃO) → clicar na célula de maior probabilidade e ler o detalhe →
trocar o seletor para uma semana do teste (as de 2022 têm muitos eventos) → mostrar os círculos (o
que aconteceu) sobre as células quentes e o KPI "acertos no top 5 %" → voltar para a semana corrente.
Ter o app Streamlit aberto como plano B caso a internet falhe (ele funciona offline).

## 12. Perguntas prováveis e respostas

**"Vocês estão prevendo terremotos?"** Não. Estimamos a probabilidade de atividade M ≥ 4 por área na
próxima semana, a partir do padrão estatístico do catálogo — sobretudo a taxa histórica da área e as
sequências de réplicas em curso. Não dizemos dia, lugar exato nem magnitude.

**"Por que a LSTM não é muito melhor que a MLP?"** Porque a informação temporal útil está em
estatísticas simples da sequência (quantos eventos, há quanto tempo), que as janelas agregadas já
contêm. A recorrência não encontra estrutura além disso. É a resposta honesta à pergunta de pesquisa.

**"Então a rede neural serve para quê?"** Para combinar melhor "onde" e "quando": ela supera o melhor
baseline estatístico em todas as métricas e, principalmente, entrega probabilidades calibradas, que a
climatologia não entrega em sequências ativas (prevê 60 % onde ocorre 84 %).

**"Por que M ≥ 4 e não só os grandes?"** Volume de exemplos (16 % de positivos contra ~2 % para
M ≥ 5) e relevância: é a faixa sentida pela população. O limiar de magnitude mais alto entra na
Hopfield, no alarme (M ≥ 5).

**"Como garantem que não há vazamento?"** Divisão temporal pela semana do rótulo; taxa histórica,
células ativas e padronização calculadas só no treino; as features da semana t usam apenas dados até
o fim da semana t; o limiar de classificação é escolhido na validação, nunca no teste.

**"O que é a RBM e o que ela mostrou?"** Um modelo baseado em energia, treinado sem rótulos. Sua
energia livre sozinha dá ROC-AUC 0,79: o modelo descobre que semanas atípicas são semanas de risco.

**"E a Hopfield, vale a pena?"** Como classificador, não: um limiar na probabilidade calibrada faz
melhor no mesmo ponto de operação. Como camada de saída, sim: três estados discretos, protótipos
legíveis (o ALARME acende exatamente os bits de magnitude ≥ 5) e alarme que só dispara em padrões de
magnitude alta, como pedido.

**"Por que a precisão do alarme é baixa (5,6 %)?"** A taxa base de M ≥ 5 na semana seguinte é 2,3 %
por célula-semana; o alarme multiplica por 2,4. Em eventos raros, precisão alta exige abrir mão de
quase todo o recall. É uma escolha de ponto de operação, e o site mostra o estado com a probabilidade
ao lado para o usuário decidir.

**"Como o site se atualiza?"** `publish_firebase.py --feed` baixa os eventos novos do USGS, calcula
a semana corrente com o mesmo código do treino e grava no Firestore; o site lê o documento. Não há
re-treino: o modelo é o mesmo avaliado no teste.

**"Os picos da célula mais ativa crescem com os anos. Isso é real?"** Os picos são sequências de
réplicas de terremotos rasos (2023: dubleto M7,1 do Mar de Banda). O crescimento é efeito de três
pontos: o M7,0 de 1998, antes da janela, deu pico maior que 2008 e 2014. Em 5.914 sequências de todas
as células, o tamanho é explicado por magnitude e profundidade; a tendência temporal cai 45 % ao usar
a faixa bem detectada (melhora de instrumentação) e o resíduo fica como pergunta aberta. Está em
`results/pulse/aftershocks.md` e em `src/study_aftershocks.py`.

**"E terremotos maiores a intervalos maiores, como um ciclo?"** Testamos em 5.287 pares consecutivos
de principais: a espera não prevê a magnitude do próximo (ρ = −0,07), a magnitude não prevê a espera
(−0,03), intervalos e magnitudes não crescem ao longo das sequências, e os intervalos seguem um
processo de Poisson (CV 0,87). A máxima anual não sobe (ρ = −0,36, p = 0,07). Em regiões inteiras a
lei de Gutenberg–Richter não tem memória; ciclos aparecem, quando aparecem, em falhas individuais.
Está em `results/pulse/cycles.md` e em `src/study_cycles.py`.

**"Vocês não tentaram mais variáveis?"** Tentamos: 22 a mais (memória de 2 e 5 anos, anel de 5 × 5,
resolução diária, taxa tipo ETAS, proxy do valor-b). ROC-AUC +0,001, PR-AUC +0,001; as 22 sozinhas
recuperam 99 % do desempenho das 45 originais. O teto está na informação do catálogo, não nas variáveis
nem na arquitetura. Está em `results/pulse/feature_experiment.md`.

**"Qual seria o próximo passo?"** Grade de 1° e horizonte de 3 dias; completude por segmento; prever
o número esperado de eventos (onde a LSTM teria mais a dizer); comparar com um modelo ETAS, o padrão
da sismologia estatística para réplicas.
