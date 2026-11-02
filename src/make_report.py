#!/usr/bin/env python3
"""
SeismicAI - consolida os resultados dos modelos em uma tabela e figuras (para relatório e artigo).

Lê results/metrics_*.json, results/preds_*.parquet, results/history_*.csv e results/importance_*.csv
e gera, em results/:
  results_table.md / results_table.csv   comparação dos modelos no teste
  fig_roc.png, fig_pr.png                curvas ROC e precision-recall (teste)
  fig_calibration.png                    diagrama de confiabilidade (calibração das probabilidades)
  fig_auc_by_year.png                    ROC-AUC e PR-AUC por ano do teste
  fig_training.png                       curvas de treino das redes (perda e PR-AUC de validação)
  fig_importance_{mlp,lstm}.png          importância por permutação
  fig_map_week.png                       mapa de probabilidades de uma semana do teste vs. eventos reais
(A RBM gera fig_rbm_weights.png e a rede de Hopfield gera hopfield_table.md e fig_hopfield_*.png.)

Uso:
    python src/make_report.py                    # perfil pulse -> results/pulse
    python src/make_report.py --profile global
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Console do Windows (cp1252) não imprime "≥" e afins: força UTF-8 na saída (sem efeito em Linux/macOS)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ORDER = ["persistencia", "climatologia", "taxa_recente", "combinado", "logreg", "rbm_energy", "rbm_logreg", "mlp_seq", "mlp", "lstm"]
LABEL = {"persistencia": "Persistência", "climatologia": "Climatologia (taxa histórica)", "taxa_recente": "Taxa recente (52 sem.)",
         "combinado": "Combinado", "logreg": "Regressão logística", "rbm_energy": "RBM (energia livre, não supervisionada)",
         "rbm_logreg": "RBM + regressão logística", "mlp_seq": "MLP (entradas da LSTM)", "mlp": "MLP (janelas)", "lstm": "LSTM"}
COLOR = {"persistencia": "#9e9e9e", "climatologia": "#f0a202", "taxa_recente": "#e76f51", "combinado": "#8d5a97",
         "logreg": "#2a9d8f", "rbm_energy": "#b5b5b5", "rbm_logreg": "#7a9e7e", "mlp_seq": "#6c8ebf", "mlp": "#1d3557", "lstm": "#c1121f"}
CURVE_MODELS = {"climatologia", "combinado", "logreg", "rbm_logreg", "mlp_seq", "mlp", "lstm"}   # curvas: só modelos com score contínuo
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(results: Path):
    metrics, preds, hist, imp = {}, {}, {}, {}
    for name in ORDER:
        f = results / f"metrics_{name}.json"
        if not f.exists():
            continue
        metrics[name] = json.loads(f.read_text(encoding="utf-8"))
        preds[name] = pd.read_parquet(results / f"preds_{name}.parquet")
        if (results / f"history_{name}.csv").exists():
            hist[name] = pd.read_csv(results / f"history_{name}.csv")
        if (results / f"importance_{name}.csv").exists():
            imp[name] = pd.read_csv(results / f"importance_{name}.csv")
    return metrics, preds, hist, imp


def table(metrics: dict, results: Path) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        t = m["test"]
        rows.append({"Modelo": LABEL[name], "ROC-AUC": t["roc_auc"], "PR-AUC": t["pr_auc"], "Brier": t["brier"],
                     "Log loss": t["log_loss"], "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"],
                     "Precision top 5%": t["precision_top5"], "Recall top 5%": t["recall_top5"],
                     "Limiar (val.)": m["threshold_from_val"]})
    df = pd.DataFrame(rows)
    df.to_csv(results / "results_table.csv", index=False)
    fmt = df.copy()
    for c in fmt.columns[1:]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.4f}" if c in ("Brier", "Log loss", "Limiar (val.)") else f"{v:.3f}")
    md = "| " + " | ".join(fmt.columns) + " |\n|" + "---|" * len(fmt.columns) + "\n"
    md += "\n".join("| " + " | ".join(map(str, r)) + " |" for r in fmt.to_numpy())
    (results / "results_table.md").write_text(md + "\n", encoding="utf-8")
    return df


def fig_curves(preds: dict, results: Path) -> None:
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(5, 4.2))
        for name, df in preds.items():
            if name not in CURVE_MODELS:
                continue
            if kind == "roc":
                x, y, _ = roc_curve(df["y"], df["p"])
            else:
                y, x, _ = precision_recall_curve(df["y"], df["p"])
            ax.plot(x, y, color=COLOR[name], label=LABEL[name], lw=1.6)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", title="Curva ROC (teste)")
        else:
            base = preds[next(iter(preds))]["y"].mean()
            ax.axhline(base, color="k", ls=":", lw=0.8, label=f"aleatório ({base:.3f})")
            ax.set(xlabel="Recall", ylabel="Precision", title="Curva precision-recall (teste)", ylim=(0, 1))
        ax.legend(frameon=False)
        fig.savefig(results / f"fig_{kind}.png")
        plt.close(fig)


def fig_calibration(preds: dict, results: Path) -> None:
    bins = [0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 1.0]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, df in preds.items():
        if name not in CURVE_MODELS:
            continue
        grp = df.groupby(pd.cut(df["p"], bins, include_lowest=True), observed=True)
        mean_p, frac, n = grp["p"].mean(), grp["y"].mean(), grp.size()
        keep = n >= 30                              # faixas com poucas linhas são ruidosas
        ax.plot(mean_p[keep], frac[keep], "o-", ms=3, color=COLOR[name], label=LABEL[name], lw=1.4)
    ax.plot([0, 0.8], [0, 0.8], "k:", lw=0.8, label="calibração perfeita")
    ax.set(xlabel="Probabilidade prevista (média da faixa)", ylabel="Fração observada de positivos",
           title="Calibração das probabilidades (teste)", xlim=(0, 0.8), ylim=(0, 0.8))
    ax.legend(frameon=False)
    fig.savefig(results / "fig_calibration.png")
    plt.close(fig)


def fig_by_year(metrics: dict, results: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    names = [n for n in metrics if n in CURVE_MODELS]
    years = sorted({int(y) for n in names for y in metrics[n]["test_by_year"]})
    w = 0.8 / len(names)
    for k, metric in enumerate(("roc_auc", "pr_auc")):
        ax = axes[k]
        for i, n in enumerate(names):
            vals = [metrics[n]["test_by_year"].get(str(y), metrics[n]["test_by_year"].get(y, {})).get(metric, np.nan)
                    for y in years]
            ax.bar(np.arange(len(years)) + (i - len(names) / 2 + 0.5) * w, vals, w, color=COLOR[n], label=LABEL[n])
        ax.set_xticks(range(len(years)), years)
        ax.set_title({"roc_auc": "ROC-AUC por ano (teste)", "pr_auc": "PR-AUC por ano (teste)"}[metric])
        if metric == "roc_auc":
            ax.set_ylim(0.5, 1.0)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(results / "fig_auc_by_year.png")
    plt.close(fig)


def fig_training(hist: dict, results: Path) -> None:
    if not hist:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, h in hist.items():
        axes[0].plot(h["epoch"], h["train_loss"], "o-", ms=3, color=COLOR[name], label=LABEL[name])
        axes[1].plot(h["epoch"], h["val_pr_auc"], "o-", ms=3, color=COLOR[name], label=LABEL[name])
    axes[0].set(xlabel="Época", ylabel="Perda de treino (BCE)", title="Perda de treino")
    axes[1].set(xlabel="Época", ylabel="PR-AUC (validação)", title="PR-AUC de validação por época")
    axes[1].legend(frameon=False)
    fig.savefig(results / "fig_training.png")
    plt.close(fig)


def fig_importance(imp: dict, results: Path, top: int = 15) -> None:
    for name, df in imp.items():
        df = df.head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5.5, 0.28 * len(df) + 1))
        ax.barh(df["feature"], df["roc_auc_drop"], color=COLOR[name])
        ax.set(xlabel="Queda de ROC-AUC ao embaralhar a variável", title=f"Importância por permutação — {LABEL[name]}")
        fig.savefig(results / f"fig_importance_{name}.png")
        plt.close(fig)


def fig_map(preds: dict, cells: pd.DataFrame, events: pd.DataFrame, results: Path, model: str | None,
            label_mag: float = 5.0) -> None:
    """Mapa de uma semana do teste: probabilidade por célula vs. eventos-alvo que ocorreram na semana seguinte."""
    name = model or next(n for n in ("lstm", "mlp", "logreg") if n in preds)
    df = preds[name]
    # escolhe a semana do teste com mais positivos (mais ilustrativa)
    wk = df.groupby("week")["y"].sum().idxmax()
    sel = df[df["week"] == wk].merge(cells[["cell", "lat_c", "lon_c"]], on="cell")
    start = pd.Timestamp(sel["label_week_start"].iloc[0])
    ev = events[(events["time"] >= start.tz_localize("UTC")) & (events["time"] < start.tz_localize("UTC") + pd.Timedelta(days=7))
                & (events["mag"] >= label_mag)]
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(sel["lon_c"], sel["lat_c"], c=sel["p"], cmap="YlOrRd", s=28, marker="s", vmin=0, vmax=max(0.3, sel["p"].max()),
                    edgecolors="none", alpha=0.95)
    ax.scatter(ev["longitude"], ev["latitude"], s=10 + 6 * (ev["mag"] - label_mag) ** 2, facecolors="none",
               edgecolors="#1d3557", lw=0.8, label=f"eventos M ≥ {label_mag:g} na semana seguinte ({len(ev)})")
    pad = 4
    ax.set(xlim=(max(-180, cells["lon_c"].min() - pad), min(180, cells["lon_c"].max() + pad)),
           ylim=(max(-90, cells["lat_c"].min() - pad), min(90, cells["lat_c"].max() + pad)),
           xlabel="Longitude", ylabel="Latitude",
           title=f"{LABEL[name]} — probabilidade de M ≥ {label_mag:g} na semana de {start:%d/%m/%Y}")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", frameon=False)
    fig.colorbar(sc, ax=ax, shrink=0.75, label="probabilidade prevista")
    fig.savefig(results / "fig_map_week.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--results", default=None, help="padrão: results/{perfil}")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--map-model", default=None)
    args = ap.parse_args()
    results = Path(args.results or f"results/{args.profile}")
    proc = Path(args.processed)
    summary = json.loads((proc / f"summary_{args.profile}.json").read_text(encoding="utf-8"))
    metrics, preds, hist, imp = load(results)
    if not metrics:
        raise SystemExit("nenhum results/metrics_*.json encontrado")
    df = table(metrics, results)
    fig_curves(preds, results)
    fig_calibration(preds, results)
    fig_by_year(metrics, results)
    fig_training(hist, results)
    fig_importance(imp, results)
    fig_map(preds, pd.read_csv(proc / f"cells_{args.profile}.csv"),
            pd.read_parquet(proc / f"events_clean_{args.profile}.parquet"), results, args.map_model,
            label_mag=float(summary["label_mag"]))
    print((results / "results_table.md").read_text(encoding="utf-8"))
    print("figuras salvas em", results)


if __name__ == "__main__":
    main()
