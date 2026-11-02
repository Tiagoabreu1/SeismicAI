"""
SeismicAI - métricas de avaliação compartilhadas por baselines e redes neurais.

Para um problema de classificação binária probabilística e desbalanceado, reportamos:
- ranking: ROC-AUC e PR-AUC (average precision);
- calibração: Brier score e log loss;
- classificação em um limiar escolhido na VALIDAÇÃO (máximo F1): precision, recall, F1, accuracy;
- operação do produto: precision e recall ao sinalizar os 5% de (célula, semana) com maior
  probabilidade ("top 5%"), que é como um monitor priorizaria atenção.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)

TOP_FRAC = 0.05


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float, top_frac: float = TOP_FRAC) -> dict:
    y = np.asarray(y).astype(int)
    p_raw = np.asarray(p, dtype=np.float64)
    pred = (p_raw >= threshold).astype(int)
    p = np.clip(p_raw, 1e-7, 1 - 1e-7)   # só para Brier / log loss
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    k = max(int(round(top_frac * len(p))), 1)
    top = np.argsort(-p)[:k]
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": float((pred == y).mean()),
        "precision_top5": float(y[top].mean()),
        "recall_top5": float(y[top].sum() / max(y.sum(), 1)),
        "mean_pred": float(p.mean()),
    }


def per_year(ids: pd.DataFrame, p: np.ndarray) -> dict:
    """ROC-AUC e PR-AUC por ano da semana do rótulo."""
    years = pd.to_datetime(ids["label_week_start"]).dt.year.to_numpy()
    out = {}
    for yr in np.unique(years):
        m = years == yr
        yy = ids["y"].to_numpy()[m]
        if yy.sum() == 0:
            continue
        out[int(yr)] = {"n": int(m.sum()), "positives": int(yy.sum()),
                        "roc_auc": float(roc_auc_score(yy, p[m])),
                        "pr_auc": float(average_precision_score(yy, p[m]))}
    return out


def summarize(name: str, ids_val, p_val, ids_test, p_test) -> dict:
    """Escolhe o limiar na validação e avalia validação e teste."""
    thr = best_f1_threshold(ids_val["y"].to_numpy(), p_val)
    return {
        "model": name,
        "threshold_from_val": thr,
        "val": evaluate(ids_val["y"].to_numpy(), p_val, thr),
        "test": evaluate(ids_test["y"].to_numpy(), p_test, thr),
        "test_by_year": per_year(ids_test, p_test),
    }


def save_preds(ids: pd.DataFrame, p: np.ndarray, path) -> None:
    out = ids.copy()
    out["p"] = p.astype(np.float32)
    out.to_parquet(path, index=False)
