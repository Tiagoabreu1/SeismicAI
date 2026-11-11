# Varredura de hiperparâmetros e sementes (perfil pulse, conjunto de teste)

## Variabilidade entre sementes (mesma configuração, sementes diferentes)

| Modelo | n | PR-AUC validação | ROC-AUC | PR-AUC | Brier | Precisão top 5 % |
|---|---|---|---|---|---|---|
| lstm | 5 | 0.5492 ± 0.0003 | 0.8413 ± 0.0001 | 0.5701 ± 0.0003 | 0.1012 ± 0.0001 | 0.762 ± 0.001 |
| mlp | 5 | 0.5475 ± 0.0003 | 0.8411 ± 0.0001 | 0.5689 ± 0.0003 | 0.1014 ± 0.0001 | 0.762 ± 0.002 |

LSTM contra MLP na PR-AUC de teste: diferença de +0.0012 (teste t de Welch: t = 6.3, p = 0.0002). LSTM de 0.5699 a 0.5705; MLP de 0.5685 a 0.5693. A LSTM é melhor de forma consistente entre sementes, e as faixas nem se sobrepõem: a diferença é pequena, mas não é ruído de treino.

## Varredura de arquitetura e treino

Cada linha é uma execução com uma semente, comparada com a configuração base do mesmo tipo de modelo (média das sementes) pela PR-AUC de validação; a MLP-seq é comparada com a base da LSTM.

| Execução | Argumentos alterados | Parâmetros | PR-AUC validação | Diferença para a base | ROC-AUC | PR-AUC | Brier | Top 5 % | Épocas (melhor) | Minutos |
|---|---|---|---|---|---|---|---|---|---|---|
| lstm_2layers_h128 | `--hidden 128 --layers 2` | 212,417 | 0.5494 | +0.0002 (dentro do ruído) | 0.8414 | 0.5706 | 0.1012 | 0.762 | 26 (21) | 8.6 |
| lstm_2layers | `--layers 2` | 57,281 | 0.5494 | +0.0002 (dentro do ruído) | 0.8412 | 0.5703 | 0.1012 | 0.763 | 23 (18) | 2.1 |
| lstm_lr3e-3 | `--lr 3e-3` | 24,001 | 0.5492 | -0.0000 (dentro do ruído) | 0.8415 | 0.5703 | 0.1012 | 0.761 | 22 (18) | 1.0 |
| lstm_drop0.4 | `--dropout 0.4` | 24,001 | 0.5492 | -0.0000 (dentro do ruído) | 0.8413 | 0.5702 | 0.1013 | 0.762 | 23 (21) | 1.1 |
| lstm_batch512 | `--batch-size 512` | 24,001 | 0.5492 | -0.0000 (dentro do ruído) | 0.8415 | 0.5704 | 0.1013 | 0.759 | 15 (10) | 0.8 |
| lstm_patience10_epochs60 | `--patience 10 --epochs 60` | 24,001 | 0.5492 | -0.0001 (dentro do ruído) | 0.8414 | 0.5705 | 0.1012 | 0.761 | 34 (24) | 1.5 |
| lstm_h256 | `--hidden 256` | 291,265 | 0.5492 | -0.0001 (dentro do ruído) | 0.8413 | 0.5699 | 0.1016 | 0.762 | 15 (15) | 3.1 |
| lstm_drop0 | `--dropout 0.0` | 24,001 | 0.5491 | -0.0001 (dentro do ruído) | 0.8413 | 0.5702 | 0.1013 | 0.762 | 17 (17) | 0.8 |
| lstm_h128 | `--hidden 128` | 80,321 | 0.5491 | -0.0002 (dentro do ruído) | 0.8414 | 0.5708 | 0.1012 | 0.763 | 22 (17) | 3.5 |
| lstm_lr3e-4 | `--lr 3e-4` | 24,001 | 0.5488 | -0.0004 (dentro do ruído) | 0.8410 | 0.5697 | 0.1012 | 0.762 | 23 (22) | 1.1 |
| lstm_h32 | `--hidden 32` | 8,129 | 0.5488 | -0.0005 (dentro do ruído) | 0.8414 | 0.5700 | 0.1012 | 0.763 | 22 (17) | 0.6 |
| mlp_512-256-128 | `--mlp-hidden 512,256,128` | 189,697 | 0.5478 | +0.0004 (dentro do ruído) | 0.8413 | 0.5693 | 0.1013 | 0.763 | 15 (10) | 0.2 |
| mlp_256-128-64 | `--mlp-hidden 256,128,64` | 53,889 | 0.5477 | +0.0003 (dentro do ruído) | 0.8412 | 0.5691 | 0.1013 | 0.762 | 16 (15) | 0.2 |
| mlp_64-32 | `--mlp-hidden 64,32` | 5,249 | 0.5473 | -0.0002 (dentro do ruído) | 0.8410 | 0.5687 | 0.1014 | 0.762 | 23 (18) | 0.2 |

## Experimentos dirigidos

Cada linha é uma execução com uma semente, comparada com a configuração base do mesmo tipo de modelo (média das sementes) pela PR-AUC de validação; a MLP-seq é comparada com a base da LSTM.

| Execução | Argumentos alterados | Parâmetros | PR-AUC validação | Diferença para a base | ROC-AUC | PR-AUC | Brier | Top 5 % | Épocas (melhor) | Minutos |
|---|---|---|---|---|---|---|---|---|---|---|
| lstm_seq104 | `--seq-len 104` | 24,001 | 0.5493 | +0.0001 (dentro do ruído) | 0.8415 | 0.5706 | 0.1013 | 0.762 | 21 (20) | 1.7 |
| lstm_posweight5.9 | `--pos-weight 5.9` | 24,001 | 0.5493 | +0.0000 (dentro do ruído) | 0.8415 | 0.5704 | 0.1013 | 0.761 | 23 (18) | 1.1 |
| lstm_posweight3 | `--pos-weight 3` | 24,001 | 0.5489 | -0.0003 (dentro do ruído) | 0.8413 | 0.5702 | 0.1012 | 0.763 | 15 (10) | 0.7 |
| lstm_seq26 | `--seq-len 26` | 24,001 | 0.5477 | -0.0016 (abaixo do ruído) | 0.8406 | 0.5689 | 0.1014 | 0.761 | 22 (21) | 0.6 |
| mlpseq_seq104 | `--model mlp_seq --seq-len 104` | 284,929 | 0.5442 | -0.0051 (abaixo do ruído) | 0.8389 | 0.5655 | 0.1018 | 0.756 | 6 (2) | 0.1 |

## Resumo

Fora do ruído na PR-AUC de validação: lstm_seq26 (pior, -0.0016), mlpseq_seq104 (pior, -0.0051).
Brier fora do ruído (positivo significa calibração pior): lstm_h256 (+0.0003), mlpseq_seq104 (+0.0005).

Como ler: a escolha de configuração se faz pela PR-AUC de validação, nunca pela de teste; as colunas de teste estão aqui para mostrar que a ordem se mantém. "Dentro do ruído" significa que a execução cai no intervalo de 95 % esperado para uma nova semente da configuração base (lstm 3.04 desvios (n = 5), mlp 3.04 desvios (n = 5)). Em "Épocas (melhor)", o segundo número é a época de maior PR-AUC de validação. A perda ponderada (pos_weight) vem acompanhada, no train_nn.py, de uma correção do logit que devolve a calibração; por isso o esperado é que ela não mude nem a PR-AUC nem o Brier. Sem essa correção, as probabilidades sairiam infladas.
