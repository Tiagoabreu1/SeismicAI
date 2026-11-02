# SeismicAI Pulse — análise crítica dos resultados

Conjunto de teste: 2021–2025, 709 células do Círculo de Fogo × 260 semanas = 184.340 amostras,
16,6 % positivas (semana seguinte com pelo menos um terremoto M ≥ 4,0). Todas as métricas abaixo são
do teste; o limiar de classificação foi escolhido na validação (2018–2020), como manda a Fase 1.
Arquivos de origem: `results/pulse/` (tabelas `results_table.md`, `strata_table.md`,
`calibration_table.md`, `hopfield_table.md` e `metrics_*.json`).

## 1. A meta da Fase 1 foi superada

A Fase 1 fixou a meta: superar o baseline **Combinado** (média da taxa histórica da célula com a taxa
das últimas 52 semanas), que atinge ROC-AUC 0,836 e PR-AUC 0,552 no teste, com boa calibração.

| Modelo | ROC-AUC | PR-AUC | Brier | Precision top 5 % |
|---|---|---|---|---|
| Persistência | 0,646 | 0,266 | 0,1964 | 0,311 |
| Climatologia (taxa histórica) | 0,830 | 0,528 | 0,1065 | 0,705 |
| Taxa recente (52 sem.) | 0,831 | 0,541 | 0,1037 | 0,741 |
| Combinado (meta) | 0,836 | 0,552 | 0,1033 | 0,742 |
| Regressão logística | 0,839 | 0,565 | 0,1017 | 0,757 |
| RBM (energia livre, sem rótulos) | 0,792 | 0,402 | 0,1166 | 0,501 |
| RBM + regressão logística | 0,836 | 0,554 | 0,1027 | 0,748 |
| MLP (entradas da LSTM, achatadas) | 0,839 | 0,566 | 0,1018 | 0,761 |
| MLP (janelas agregadas) | 0,841 | 0,569 | 0,1014 | 0,760 |
| **LSTM** | **0,841** | **0,571** | **0,1012** | **0,761** |

A LSTM é o melhor modelo em todas as métricas: ROC-AUC 0,841 (+0,005 sobre a meta), PR-AUC 0,571
(+0,019), Brier 0,1012 (menor erro quadrático das probabilidades) e, no ponto de operação que mais
interessa ao produto, **76,1 % das 5 % células sinalizadas como mais prováveis em cada semana têm de
fato um terremoto M ≥ 4 na semana seguinte** (climatologia: 70,5 %). O desempenho é estável nos cinco
anos do teste (ROC-AUC entre 0,838 e 0,844; PR-AUC entre 0,560 e 0,580), sem degradação ao longo do
tempo apesar da deriva de completude do catálogo apontada na Fase 1.

Em termos absolutos os ganhos são modestos, e é importante dizer isso com clareza: o problema é
dominado por *onde* a sismicidade é crônica, informação que a taxa histórica já captura (ROC-AUC
0,830 sozinha). O que as redes acrescentam é a parcela temporal, e ela aparece concentrada em
situações específicas, como mostram as seções seguintes.

## 2. Onde o ganho acontece: nas sequências em curso

A tabela por estratos (`strata_table.md`) separa o teste por situação da célula na semana de entrada:

| Estrato | n | Positivos | PR-AUC climatologia | PR-AUC LSTM | Ganho |
|---|---|---|---|---|---|
| Todo o teste | 184.340 | 16,6 % | 0,528 | 0,571 | +0,043 |
| Sem M ≥ 4 na célula nas últimas 4 semanas | 112.522 | 7,2 % | 0,216 | 0,233 | +0,017 |
| Com M ≥ 4 na célula nas últimas 4 semanas | 71.818 | 31,4 % | 0,610 | 0,657 | +0,047 |
| Com M ≥ 4 na célula na última semana | 30.682 | 41,0 % | 0,691 | 0,743 | +0,052 |
| Com M ≥ 5 na célula nas últimas 4 semanas | 14.812 | 42,5 % | 0,682 | 0,753 | +0,071 |
| Com M ≥ 5 nos vizinhos nas últimas 4 semanas | 70.249 | 24,4 % | 0,593 | 0,637 | +0,044 |
| Células quietas (taxa histórica < 3 %/semana) | 45.760 | 1,7 % | 0,025 | 0,068 | ×2,7 |
| Células ativas (taxa histórica > 20 %/semana) | 46.020 | 43,3 % | 0,640 | 0,687 | +0,047 |

Três leituras:

1. **O ganho cresce com a atividade recente.** Após um M ≥ 5 na própria célula, a PR-AUC passa de
   0,682 para 0,753 e a ROC-AUC de 0,748 para 0,799. É a hipótese H1 (agrupamento temporal) confirmada
   no modelo: a rede reconhece sequências de réplicas e eleva a probabilidade enquanto elas duram.
2. **Os vizinhos contam (H2).** Quando há um M ≥ 5 nas células vizinhas, a LSTM ganha +0,044 de PR-AUC
   sobre a climatologia, e a variável `nviz` (eventos nos vizinhos) é a terceira mais importante da
   LSTM. O contágio espacial existe e é aproveitado, embora seja pequeno frente ao sinal da própria célula.
3. **Nas células historicamente quietas, a rede quase triplica a PR-AUC** (0,025 → 0,068) e eleva a
   ROC-AUC de 0,629 para 0,679. Em valor absoluto continua baixo — são células com 1,7 % de positivos —,
   mas é exatamente onde a taxa histórica não tem nada a dizer e a atividade recente é a única pista.

## 3. As probabilidades são confiáveis (calibração)

A tabela de calibração agrupa o teste pelas faixas de probabilidade da LSTM:

| Faixa prevista (LSTM) | n | LSTM prevê | Observado | Climatologia prevê |
|---|---|---|---|---|
| 0–2 % | 27.644 | 1,4 % | 1,0 % | 1,5 % |
| 2–5 % | 44.718 | 3,2 % | 3,7 % | 3,4 % |
| 5–10 % | 28.854 | 7,2 % | 8,3 % | 7,5 % |
| 10–20 % | 33.522 | 14,3 % | 15,1 % | 14,4 % |
| 20–30 % | 17.919 | 24,5 % | 25,4 % | 23,3 % |
| 30–50 % | 18.679 | 38,7 % | 40,4 % | 34,2 % |
| 50–70 % | 7.655 | 59,0 % | 61,6 % | 49,3 % |
| 70–100 % | 5.349 | 82,7 % | 83,7 % | 60,2 % |

Quando a LSTM diz 83 %, acontece em 84 % dos casos; quando diz 59 %, em 62 %. A climatologia, nas
mesmas células e semanas, prevê 60 % onde se observa 84 %: ela não enxerga a sequência em curso.
Esse é o critério de sucesso número 2 da Fase 1 (modelo bem calibrado) atendido com folga, e é o que
permite usar a probabilidade diretamente no produto, sem recalibração.

## 4. O que as redes aprenderam

A importância por permutação (queda de ROC-AUC ao embaralhar uma variável no teste) mostra, para a
LSTM: taxa histórica da célula +0,105; sequência de nº de eventos +0,059; sequência de eventos nos
vizinhos +0,004; semanas desde o último evento +0,003; as demais abaixo de +0,002. Para a MLP: taxa
histórica +0,105; depois `n_52w`, `n_12w`, `meanmag_52w`, `wktarget_52w`, `n_1w`, todas entre +0,002 e
+0,004.

Dois modelos muito diferentes chegam à mesma hierarquia: **"onde" (taxa histórica) vale cerca de dez
vezes mais que "quando" (atividade recente)**, e dentro do "quando" o que importa é a contagem de
eventos, não a magnitude nem a profundidade. Isso é coerente com a leitura da Fase 1 e com a
sismologia estatística: a taxa de fundo é a melhor previsora, e o excesso de curto prazo vem das
sequências de réplicas.

## 5. LSTM, MLP e MLP-seq: a recorrência não acrescenta — e isso é um resultado

Três redes receberam a mesma informação de formas diferentes: a LSTM lê a sequência bruta de 52
semanas; a MLP-seq lê a mesma sequência achatada (52 × 10 + 5 = 525 entradas); a MLP lê 45 resumos da
sequência (janelas de 1, 4, 12 e 52 semanas). Resultado: LSTM 0,841 / 0,571, MLP 0,841 / 0,569,
MLP-seq 0,839 / 0,566.

A diferença entre elas está dentro do ruído. A LSTM é a melhor por margem mínima, a MLP-seq a pior
(e a que mais sofreu com sobreajuste: 151 mil parâmetros, melhor época = 2). A interpretação: a
informação temporal útil está em estatísticas simples da sequência — quantos eventos, há quanto
tempo —, que as janelas agregadas já contêm. A recorrência da LSTM não encontra estrutura temporal
além dessa. Para o produto, isso é bom: a MLP treina em 68 s na CPU contra 12 min da LSTM, com
resultado equivalente; para a ciência, é a resposta honesta à pergunta de pesquisa — sim, a rede
supera os baselines, mas não porque "aprende a sequência", e sim porque combina melhor taxa de fundo
e atividade recente.

Outro ponto: a regressão logística (0,839 / 0,565) fica muito perto das redes. O que separa os modelos
é pequeno porque o problema, nesta formulação, tem pouca não-linearidade a explorar.

## 6. RBM: o que se aprende sem rótulos

A Máquina de Boltzmann Restrita foi treinada sem ver nenhum rótulo, por divergência contrastiva (10
épocas, 18 s; erro de reconstrução de 0,030 para 0,018). Sua **energia livre**, calibrada em
probabilidade por uma regressão logística de um único parâmetro, atinge ROC-AUC 0,792 e PR-AUC 0,402 —
acima da persistência (0,646) e não muito abaixo da climatologia (0,830). Ou seja, a RBM descobre
sozinha que "semana atípica" é "semana de risco": o padrão raro na distribuição das features é
justamente o das células ativas. Com uma regressão logística sobre as 64 unidades ocultas, chega a
0,836 / 0,554, empatando com o baseline combinado e um pouco abaixo da regressão logística sobre as
features originais (0,839 / 0,565): a compressão em 64 unidades binárias perde alguma informação.

## 7. Rede de Hopfield: a camada de saída, com honestidade

A rede de Hopfield recebe 26 bits (limiares sobre a probabilidade da LSTM, sobre a atividade da célula e
dos vizinhos e sobre a magnitude máxima recente) e relaxa até um de três protótipos: NORMAL, ATENÇÃO,
ALARME. O alarme é associado ao limiar de magnitude M ≥ 5,0, como pedido: a memória de ALARME foi
aprendida apenas com semanas de alarme *com assinatura* — evento M ≥ 5 recente na célula ou nos
vizinhos —, que são 75 % dos alarmes reais do teste (no Círculo de Fogo, com células de 2°, os M ≥ 5
se agrupam).

Os protótipos aprendidos são legíveis: NORMAL tem praticamente nada aceso; ATENÇÃO acende `p ≥ 0,3`,
atividade na célula e nos vizinhos e célula historicamente ativa; ALARME acende tudo isso **e** os bits
de magnitude (`maxmag_12w ≥ 5`, `maxmag_52w ≥ 5,5`, `wsl_big ≤ 8`). O estado é um diagnóstico que se
explica em uma frase.

Em desempenho, no teste: acurácia de 74,7 % nos três estados, 10,7 % de estados espúrios, alarme emitido
em 11,4 % das células-semana com precision 0,056 (2,4 × a taxa base de 2,3 %) e recall 0,279. A
comparação justa, uma regra `p ≥ 0,381` que emite a mesma fração de alarmes, faz melhor: precision
0,092 e recall 0,483. A conclusão deve ser dita como é: **uma memória binária de 26 bits não supera um
limiar bem escolhido sobre uma probabilidade calibrada, e não tinha como**. O papel da Hopfield no
produto é outro — transformar a probabilidade em estados discretos, interpretáveis e robustos a bits
faltantes, com um alarme que só dispara em padrões de magnitude alta — e é por isso que ela é a
camada de saída do mapa, não o classificador.

## 8. Mais variáveis ajudam? Não — e isso diz algo

Como os ganhos são pequenos, testamos se faltavam variáveis. Construímos 22 novas, com motivação
física, a partir do mesmo catálogo: memória longa (janelas de 104 e 260 semanas, tempo desde o último
M ≥ 6 e M ≥ 7), contexto espacial mais amplo (anel de 5 × 5 células e atividade do segmento),
resolução diária (eventos nos últimos 1 e 3 dias, dias desde o último evento), uma taxa tipo ETAS
(soma das contribuições dos eventos passados com decaimento de Omori, na célula e nos vizinhos) e
composição (proxy do valor-b, fração de eventos profundos, aceleração). Comparamos, nos mesmos splits,
regressão logística e gradient boosting — que extrai praticamente tudo o que uma tabela tem e, nas 45
variáveis base, reproduz o resultado das redes (0,841 / 0,569).

| Conjunto | Modelo | nº | ROC-AUC | PR-AUC | PR-AUC c/ M ≥ 5 recente | PR-AUC células quietas |
|---|---|---|---|---|---|---|
| base | gradient boosting | 45 | 0,8412 | 0,5694 | 0,750 | 0,066 |
| base + extras | gradient boosting | 67 | 0,8420 | 0,5707 | 0,752 | 0,073 |
| só extras | gradient boosting | 22 | 0,8389 | 0,5637 | 0,744 | 0,071 |

Vinte e duas variáveis a mais movem a ROC-AUC em +0,0008 e a PR-AUC em +0,0013 — dentro do ruído.
As 22 sozinhas recuperam 99 % do desempenho das 45: descrevem a mesma informação de outro jeito. Na
importância por permutação, a taxa histórica continua dominando (+0,063), seguida do nº de semanas
ativas nas últimas 52 (+0,008); a melhor variável nova, `n_104w`, vale +0,003. O único sinal
genuinamente novo é modesto e coerente com o resto: a memória longa (104 e 260 semanas) ajuda nas
células quietas (0,066 → 0,073, +10 % relativo), onde o passado distante é a única pista.

A conclusão é científica, não uma desculpa: **o teto desta formulação está na informação do catálogo,
não na escolha das variáveis nem na arquitetura**. O que falta para prever o restante — terremotos
grandes sem precursores — não está em um catálogo de eventos passados.

## 9. Os picos semanais são réplicas: produtividade, Omori e Båth

O gráfico da célula mais ativa (Mar de Banda, −7°, 129°) tem três picos — 16, 25 e 59 eventos em uma
semana, em 2008, 2014 e 2023 — que parecem crescer com os anos. São sequências de réplicas de
terremotos rasos: um M5,8 a 10 km (jan/2008), um M5,8 a 7 km (fev/2014) e o dubleto M7,1 + M6,7 a
6–10 km (8/nov/2023), com 39 réplicas M ≥ 4,5 em 28 dias. Não são outliers a descartar: são o
fenômeno que a hipótese H1 descreve e que o modelo aprende. A "tendência linear" é um efeito de três
pontos e da janela começar em 2000: o M7,0 raso de novembro de 1998, na mesma célula, produziu um pico
maior que os de 2008 e 2014, e a célula tem muitos terremotos profundos (100–220 km) que quase não
geram réplicas (o M7,3 de 2019, a 212 km, deu 5).

Para testar a intuição em escala, `src/study_aftershocks.py` tomou os 5.914 terremotos principais
M ≥ 5,5 (primeiro ou maior da sequência, sem evento maior nos 56 dias anteriores nem nos 28
seguintes) nas 627 células que os têm, contou as réplicas na célula nos 28 dias seguintes e ajustou
log₁₀(réplicas + 1) = a + α·(M − 5,5) + γ·[raso ≤ 70 km] + β·(ano − 2000)/10:

| Réplicas contadas | α (por unidade de M) | γ (raso) | β (por década) | t de β | R² |
|---|---|---|---|---|---|
| todas (M ≥ 4) | 0,47 ± 0,01 | +0,17 ± 0,01 | +0,068 ± 0,007 | 9,6 | 0,26 |
| faixa completa (M ≥ 4,5) | 0,42 ± 0,01 | +0,20 ± 0,01 | +0,038 ± 0,006 | 6,1 | 0,27 |

O tamanho de uma sequência é explicado por magnitude (≈ 2,8 × mais réplicas por unidade de magnitude
dentro da célula e dos 28 dias; o valor clássico, ≈ 10 ×, vale para a zona de réplicas inteira e sem
truncar o tempo) e por profundidade (rasos ≈ 1,5 × mais). Existe uma tendência temporal, mas **45 %
dela desaparece ao contar só a faixa bem detectada** — era melhora de instrumentação, a mesma que a
Fase 1 apontou na Indonésia. O resíduo, ≈ +9 % por década, é estatisticamente significativo e não
pode ser atribuído, com este catálogo, a incompletude residual ou a mudança real; fica registrado como
pergunta aberta. As leis clássicas fecham: Båth (M principal − maior réplica) dá 1,05 (esperado 1,2;
a diferença vem de réplicas grandes em células vizinhas) e Omori dá p entre 1,0 e 1,8 nas maiores
sequências (Tohoku, célula do epicentro: p = 1,21). Uma ressalva de leitura: nas sequências gigantes
a zona de réplicas excede uma célula de 2°, e o "principal" de uma célula vizinha ao epicentro pode
ser uma grande réplica — é o caso do M6,6 de 11/03/2011 em (37°, 143°), com 1.025 eventos em 28 dias.

**E um ciclo — terremotos maiores a intervalos maiores?** `src/study_cycles.py` testou a ideia nos
5.914 terremotos principais, em ordem cronológica dentro de cada célula (5.287 pares consecutivos):

| Hipótese | ρ de Spearman | leitura |
|---|---|---|
| espera maior desde o principal anterior → próximo maior? | −0,07 (dentro da célula: −0,04) | não; se algo, levemente menor |
| terremoto maior → espera maior até o próximo? | −0,03 | não |
| magnitudes consecutivas têm memória? | +0,04 | desprezível |
| intervalos crescem ao longo da sequência da célula? | +0,02 (p = 0,25) | não |
| magnitudes crescem ao longo da sequência? | −0,04 | não; levemente decrescentes |
| intervalo longo é seguido de intervalo mais longo? | −0,06 | não: é seguido de um mais curto |
| magnitude máxima anual sobe com os anos? | −0,36 (p = 0,07) | não; se algo, desce |

Todos os coeficientes são desprezíveis em magnitude (|ρ| < 0,1, embora alguns sejam "significativos"
pelo tamanho da amostra) e vários têm o sinal contrário ao da hipótese. Os intervalos entre
principais se distribuem como um processo de Poisson (coeficiente de variação mediano 0,87, quartis
0,75–0,99): o passado não diz quando vem o próximo. É o esperado para regiões inteiras — a lei de
Gutenberg–Richter não tem memória — e difere do que às vezes se observa em uma única falha com vários
ciclos registrados (modelo "previsível no tempo", Shimazaki & Nakata, 1980); nossas células de 2°
juntam várias falhas, e o ciclo de cada uma se dilui. Um achado lateral, no sentido oposto ao da
intuição de crescimento: a contagem anual de principais M ≥ 5,5 cai no período (ρ = −0,53, p = 0,006),
concentrada na Indonésia (−0,71). Pode ser mudança do tipo de magnitude preferido pelo USGS ao longo
dos anos (mb → Mw) ou variabilidade real; fica registrado como observação.

Para o modelo, a consequência é direta: os picos são o sinal, e a variável que melhor os descreve é a
contagem recente de eventos (segunda mais importante na LSTM) — exatamente o que a taxa tipo ETAS do
experimento de variáveis formaliza. E não há "ciclo" a modelar: a informação útil está no agrupamento
de curto prazo, não em intervalos longos.

## 10. Limitações

- **Completude do catálogo.** A detecção de eventos M 4,0–4,4 melhorou ao longo do período,
  sobretudo na Indonésia (razão M 4,0–4,4 / M ≥ 4,5 de 0,58 em 2000–2005 contra 1,23 em 2015+); a
  taxa de positivos sobe de 14,5 % (treino) para 16,6 % (teste). O modelo absorveu bem a deriva (a
  calibração no teste é boa), mas parte da "atividade" aprendida é capacidade de medição, não sismicidade.
- **Resolução.** Células de 2° (~220 km) e semanas são grosseiras. Um terremoto grande sem precursores
  no catálogo não tem padrão reconhecível: na Hopfield, 25 % dos M ≥ 5 chegam sem nenhuma assinatura.
- **Sem física.** Não há modelo de falhas, tensões ou mecanismos focais; só estatística do catálogo.
- **Catálogo heterogêneo.** Tipos de magnitude diferentes (mb, mww, ml) tratados como uma única escala.
- **Ganhos pequenos.** As redes superam a meta, mas por décimos de ponto. Quem usar o produto deve
  saber que a maior parte da informação é a taxa histórica da região.

## 11. Conclusões

1. A pergunta de pesquisa tem resposta afirmativa e quantificada: a LSTM supera o melhor baseline
   estatístico (ROC-AUC 0,841 vs 0,836; PR-AUC 0,571 vs 0,552), com probabilidades calibradas.
2. O ganho é temporal e localizado: aparece nas semanas após eventos grandes e nas células quietas que
   despertam, onde a taxa histórica é cega.
3. Nem a arquitetura nem mais variáveis mudam o resultado: LSTM, MLP e regressão logística ficam a
   0,002 de ROC-AUC umas das outras, e 22 variáveis extras acrescentam 0,001. O teto está no catálogo.
4. A RBM mostra que o sinal é descoberto até sem rótulos; a Hopfield entrega um estado de alerta
   legível, ao custo de precisão frente a um limiar simples.
5. O protótipo (site no Firebase e app Streamlit) usa exatamente o modelo avaliado: o motor de
   inferência reproduz as probabilidades do treino com diferença de 10⁻⁹.

## 12. O que faria a diferença numa próxima versão

Refinar a grade (1°) e o horizonte (3 dias) para capturar réplicas mais cedo; tratar a completude por
segmento (começar a Indonésia em 2006, como sugerido na Fase 1); incluir a magnitude do último evento
grande como entrada explícita da Hopfield com mais bits; prever também o número esperado de eventos
(regressão), que depende mais da sequência e daria à LSTM um papel maior; e comparar com um modelo
ETAS, o padrão da sismologia estatística para réplicas.
