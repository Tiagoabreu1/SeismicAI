#!/usr/bin/env python3
"""
SeismicAI - modelos de referência (baselines), contra os quais as redes neurais são comparadas.

- persistência  : p = 1 se houve evento-alvo na célula na última semana, senão 0;
- climatologia  : p = fração histórica de semanas com evento-alvo na célula (só período de treino);
- taxa recente  : p = fração das últimas 52 semanas com evento-alvo na célula;
- combinado     : média da climatologia com a taxa recente (o melhor baseline do notebook da Fase 1);
- regressão logística : modelo linear sobre todas as features tabulares padronizadas.

Uso:
    python src/train_baselines.py                   # perfil pulse
    python src/train_baselines.py --profile global
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from metrics import save_preds, summarize
from seismic_data import Scaler, load_tabular, paths, save_json, split_xy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--data", default=None, help="dataset parquet (padrão: o do perfil)")
    ap.add_argument("--outdir", default=None, help="padrão: results/{perfil}")
    ap.add_argument("--models-dir", default=None, help="padrão: models/{perfil}")
    args = ap.parse_args()
    outdir = Path(args.outdir or f"results/{args.profile}")
    mdir = Path(args.models_dir or f"models/{args.profile}")
    outdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    ds, feature_cols, _ = load_tabular(args.data or paths(args.profile)["dataset"])
    X, y, ids = split_xy(ds, feature_cols)
    col = {c: i for i, c in enumerate(feature_cols)}
    target_1w = col.get("ntarget_1w", col["n_1w"])            # no perfil pulse o alvo é qualquer evento

    # 1) persistência
    p_val = (X["val"][:, target_1w] > 0).astype(np.float32)
    p_test = (X["test"][:, target_1w] > 0).astype(np.float32)
    report(outdir, "persistencia", ids, p_val, p_test)

    # 2) climatologia (taxa histórica da célula no treino: já é uma probabilidade)
    clim = {s: X[s][:, col["rate_target_train"]] for s in X}
    report(outdir, "climatologia", ids, clim["val"], clim["test"])

    # 3) taxa recente (fração das últimas 52 semanas com evento-alvo)
    recent = {s: X[s][:, col["wktarget_52w"]] / 52.0 for s in X}
    report(outdir, "taxa_recente", ids, recent["val"], recent["test"])

    # 4) combinado
    report(outdir, "combinado", ids, 0.5 * (clim["val"] + recent["val"]), 0.5 * (clim["test"] + recent["test"]))

    # 5) regressão logística
    t0 = time.time()
    scaler = Scaler(feature_cols).fit(X["train"])
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(scaler.transform(X["train"]), y["train"])
    p_val = clf.predict_proba(scaler.transform(X["val"]))[:, 1]
    p_test = clf.predict_proba(scaler.transform(X["test"]))[:, 1]
    report(outdir, "logreg", ids, p_val, p_test, extra={"train_seconds": round(time.time() - t0, 1)})
    with open(mdir / "logreg.pkl", "wb") as fh:
        pickle.dump({"model": clf, "scaler": scaler.to_dict(), "features": feature_cols}, fh)
    coef = sorted(zip(feature_cols, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    save_json({"intercept": float(clf.intercept_[0]), "coef": {k: float(v) for k, v in coef}},
              outdir / "coef_logreg.json")


def report(outdir: Path, name: str, ids, p_val, p_test, extra: dict | None = None) -> None:
    res = summarize(name, ids["val"], p_val, ids["test"], p_test)
    res.update(extra or {})
    save_json(res, outdir / f"metrics_{name}.json")
    save_preds(ids["test"], p_test, outdir / f"preds_{name}.parquet")
    save_preds(ids["val"], p_val, outdir / f"preds_{name}_val.parquet")
    t = res["test"]
    print(f"{name:<13} teste: ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | Brier {t['brier']:.4f} | "
          f"P {t['precision']:.3f} R {t['recall']:.3f} F1 {t['f1']:.3f} | top5%: P {t['precision_top5']:.3f} R {t['recall_top5']:.3f}")


if __name__ == "__main__":
    main()
