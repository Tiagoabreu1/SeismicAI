# SeismicAI Pulse, mapa das entregas finais

Disciplina de Redes Neurais e Deep Learning, 2026, Alice De Muylder, Valmir Cerutti e Tiago Abreu.
Site do protótipo: https://deep-learning-terremoto.web.app

| # | Entrega pedida | Onde está neste repositório |
|---|---|---|
| 1 | Código-fonte do projeto | `src/` (pipeline, modelos, análises), `app/` (protótipo Streamlit), `web/` (site), `notebooks/01_fase1_observacoes.ipynb` (Fase 1) |
| 2 | Dataset utilizado ou instruções para sua obtenção | `data/README.md`: comando de download (`src/download_usgs.py`), fonte (USGS ComCat), período, filtros e contagens esperadas; `src/build_dataset.py` reconstrói o dataset de modelagem em segundos |
| 3 | Modelo de Rede Neural desenvolvido | `models/pulse/`: `lstm.pt` (modelo principal), `mlp.pt`, `mlp_seq.pt`, `rbm.pt`, `hopfield.json` (camada de saída), `logreg.pkl`; código em `src/train_nn.py`, `src/train_rbm.py`, `src/train_hopfield.py` |
| 4 | Resultados, métricas e gráficos | `results/pulse/`: `results_table.md` (tabela comparativa), `strata_table.md`, `calibration_table.md`, `hopfield_table.md`, `sweep.md` (sementes e varredura de hiperparâmetros), `feature_experiment.md`, `aftershocks.md`, `cycles.md`, `metrics_*.json`, figuras `fig_*.png`; leitura em `docs/analise_resultados.md` |
| 5 | Protótipo funcional | Site público no Firebase (Hosting + Firestore): https://deep-learning-terremoto.web.app, código em `web/`, publicação em `src/publish_firebase.py`; app local `streamlit run app/app.py` |
| 6 | Apresentação da startup | `docs/apresentacao.html` (21 slides; abre em qualquer navegador) |
| 7 | Artigo de divulgação da startup | `docs/artigo.docx` (também `artigo.pdf` e `artigo.md`) |

Documentos de apoio: `README.md` (definição do problema, modelos, como rodar), `docs/analise_resultados.md`
(análise crítica dos resultados) e `web/README.md` (deploy no Firebase).

Reprodução do zero (CPU basta, ~30 min): `pip install -r requirements.txt` e os comandos da seção 3 do `README.md`.

> Finalidade acadêmica e experimental. O SeismicAI Pulse não é um sistema de alerta nem faz previsão
> determinística de terremotos; não substitui as instituições responsáveis pelo monitoramento sísmico.
