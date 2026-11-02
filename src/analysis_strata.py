#!/usr/bin/env python3
"""
SeismicAI - análise crítica dos resultados: onde os modelos ganham da climatologia?

Avalia cada modelo em estratos do conjunto de teste (semanas após sequências sísmicas, células
quietas vs. ativas) e compara a calibração das probabilidades por faixa. Gera:
  results/strata_table.md     ROC-AUC e PR-AUC por estrato e modelo
  results/calibration_table.md  probabilidade prevista vs. observada por faixa (modelo escolhido)

Uso:
    python src/analysis_strata.py --results results --data data/processed/dataset_c3.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Console do Windows (cp1252) não imprime "≥" e afins: força UTF-8 na saída (sem efeito em Linux/macOS)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = ["climatologia", "combinado", "logreg", "rbm_logreg", "mlp_seq", "mlp", "lstm"]
LABEL = {"climatologia": "Climatologia", "combinado": "Combinado", "logreg": "Reg. logística",
         "rbm_logreg": "RBM+logreg", "mlp_seq": "MLP (seq.)", "mlp": "MLP", "lstm": "LSTM"}


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.get(c, "{}").format(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--results", default=None, help="padrão: results/{perfil}")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--data", default=None, help="dataset parquet (padrão: o do perfil)")
    ap.add_argument("--calib-model", default=None, help="modelo da tabela de calibração (padrão: lstm ou mlp)")
    args = ap.parse_args()
    results = Path(args.results or f"results/{args.profile}")
    import json
    summary = json.loads((Path(args.processed) / f"summary_{args.profile}.json").read_text(encoding="utf-8"))
    target, big = float(summary["label_mag"]), float(summary["big_mag"])

    ds = pd.read_parquet(args.data or Path(args.processed) / f"dataset_{args.profile}.parquet")
    te = ds[ds["split"] == "test"].reset_index(drop=True)
    y = te["y"].to_numpy()
    preds = {}
    for m in MODELS:
        f = results / f"preds_{m}.parquet"
        if f.exists():
            p = pd.read_parquet(f)
            assert len(p) == len(te) and (p["cell"].to_numpy() == te["cell"].to_numpy()).all()
            preds[m] = p["p"].to_numpy()

    tg = "ntarget" if "ntarget_4w" in te.columns else "n"       # no perfil pulse o alvo é qualquer evento
    q_lo, q_hi = te["rate_target_train"].quantile([0.25, 0.75])
    strata = {
        "Todo o teste": np.ones(len(te), bool),
        f"Sem M ≥ {target:g} na célula nas últimas 4 semanas": (te[f"{tg}_4w"] == 0).to_numpy(),
        f"Com M ≥ {target:g} na célula nas últimas 4 semanas": (te[f"{tg}_4w"] > 0).to_numpy(),
        f"Com M ≥ {target:g} na célula na última semana": (te[f"{tg}_1w"] > 0).to_numpy(),
        f"Com M ≥ {big:g} na célula nas últimas 4 semanas": (te["nbig_4w"] > 0).to_numpy(),
        f"Com M ≥ {big:g} nos vizinhos nas últimas 4 semanas": (te["nvizbig_4w"] > 0).to_numpy(),
        f"Células quietas (taxa histórica abaixo de {q_lo:.1%}/semana)": (te["rate_target_train"] < q_lo).to_numpy(),
        f"Células ativas (taxa histórica acima de {q_hi:.1%}/semana)": (te["rate_target_train"] > q_hi).to_numpy(),
    }
    rows = []
    for name, m in strata.items():
        row = {"Estrato": name, "n": int(m.sum()), "Taxa de positivos": float(y[m].mean())}
        for k, p in preds.items():
            row[f"ROC-AUC {LABEL[k]}"] = roc_auc_score(y[m], p[m])
            row[f"PR-AUC {LABEL[k]}"] = average_precision_score(y[m], p[m])
        rows.append(row)
    df = pd.DataFrame(rows)
    fmt = {c: "{:.3f}" for c in df.columns if c not in ("Estrato", "n")}
    fmt["n"] = "{:,}"
    (results / "strata_table.md").write_text(md_table(df, fmt), encoding="utf-8")
    df.to_csv(results / "strata_table.csv", index=False)
    print(md_table(df, fmt))

    model = args.calib_model or next(m for m in ("lstm", "mlp", "logreg") if m in preds)
    bins = [0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]
    cut = pd.cut(preds[model], bins, include_lowest=True)
    cal = (pd.DataFrame({"faixa": cut, "p": preds[model], "y": y, "clim": preds["climatologia"]})
           .groupby("faixa", observed=True)
           .agg(n=("y", "size"), prevista=("p", "mean"), observada=("y", "mean"), climatologia=("clim", "mean"))
           .reset_index())
    cal.columns = ["Faixa de probabilidade prevista", "n", f"Prevista ({LABEL[model]})", "Observada", "Prevista (climatologia)"]
    cal["Faixa de probabilidade prevista"] = cal["Faixa de probabilidade prevista"].astype(str)
    fmt = {c: "{:.3f}" for c in cal.columns[2:]}
    fmt["n"] = "{:,}"
    (results / "calibration_table.md").write_text(md_table(cal, fmt), encoding="utf-8")
    print(md_table(cal, fmt))


if __name__ == "__main__":
    main()
