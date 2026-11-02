"""
SeismicAI - utilitários de dados compartilhados pelos scripts de modelagem.

- resolve os caminhos dos arquivos de um perfil (pulse / global);
- carrega o dataset tabular (uma linha por célula x semana) e separa features / rótulo / split;
- padroniza as features (log1p nas contagens + z-score), com parâmetros ajustados só no treino;
- carrega o painel semanal como array 3-D [célula, semana, variável] para a LSTM.
"""
from __future__ import annotations

import json
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

ID_COLS = ["cell", "week", "y", "split", "week_start", "label_week_start"]
STATIC_CANDIDATES = ["lat_c", "cos_lon", "sin_lon", "rate_target_train", "rate_any_train", "mean_n_train"]
PANEL_ID_COLS = ["cell", "week", "week_start", "is_train"]
SEQ_LEN = 52
COUNT_PREFIXES = ("n_", "ntarget", "nbig", "n6", "nviz", "wktarget", "mean_n")


def paths(profile: str, processed: str | Path = "data/processed") -> dict[str, Path]:
    p = Path(processed)
    return {"dataset": p / f"dataset_{profile}.parquet", "panel": p / f"panel_{profile}.parquet",
            "cells": p / f"cells_{profile}.csv", "events": p / f"events_clean_{profile}.parquet",
            "summary": p / f"summary_{profile}.json"}


def is_count(col: str) -> bool:
    return col == "n" or col.startswith(COUNT_PREFIXES)


class Scaler:
    """log1p nas colunas de contagem e z-score em todas; ajustado apenas no treino."""

    def __init__(self, cols: list[str]):
        self.cols = list(cols)
        self.log_mask = np.array([is_count(c) for c in self.cols])
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _pre(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=True)
        X[..., self.log_mask] = np.log1p(np.maximum(X[..., self.log_mask], 0.0))
        return X

    def fit(self, X: np.ndarray) -> "Scaler":
        Z = self._pre(X).reshape(-1, len(self.cols))
        self.mean = Z.mean(axis=0)
        self.std = Z.std(axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((self._pre(X) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"cols": self.cols, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        s = cls(d["cols"])
        s.mean, s.std = np.array(d["mean"], np.float32), np.array(d["std"], np.float32)
        return s


def load_tabular(path: str | Path):
    """Dataset tabular -> (DataFrame, lista de features, lista de features estáticas da célula)."""
    ds = pd.read_parquet(path)
    feature_cols = [c for c in ds.columns if c not in ID_COLS]
    static_cols = [c for c in STATIC_CANDIDATES if c in ds.columns]
    return ds, feature_cols, static_cols


def split_xy(ds: pd.DataFrame, feature_cols: list[str]):
    """Devolve dicionários X[split], y[split] (numpy) e a tabela de ids por split."""
    X, y, ids = {}, {}, {}
    for s in ("train", "val", "test"):
        part = ds[ds["split"] == s]
        X[s] = part[feature_cols].to_numpy(np.float32)
        y[s] = part["y"].to_numpy(np.int8)
        ids[s] = part[["cell", "week", "label_week_start", "y"]].reset_index(drop=True)
    return X, y, ids


def load_panel(path: str | Path):
    """Painel semanal -> (P[célula, semana, variável], ids das células na ordem das linhas,
    nomes das variáveis, máscara das semanas de treino)."""
    panel = pd.read_parquet(path)
    seq_cols = [c for c in panel.columns if c not in PANEL_ID_COLS]
    cells = np.sort(panel["cell"].unique())
    n_weeks = int(panel["week"].max()) + 1
    panel = panel.set_index(["cell", "week"]).sort_index()
    P = panel[seq_cols].to_numpy(np.float32).reshape(len(cells), n_weeks, len(seq_cols))
    train_mask = panel["is_train"].to_numpy().reshape(len(cells), n_weeks)[0]
    return P, cells, seq_cols, train_mask


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=float)
