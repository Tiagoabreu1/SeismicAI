#!/usr/bin/env python3
"""
SeismicAI - construção do dataset de modelagem (célula espacial x semana), orientada a perfis.

Perfis
  pulse   (padrão) SeismicAI Pulse, a definição da Fase 1 do grupo: probabilidade de >= 1 evento
          M >= 4,0 na semana seguinte, células de 2° x 2° no Círculo de Fogo, 2000-2025,
          treino <= 2017 / validação 2018-2020 / teste 2021-2025, células com >= 10 eventos no treino.
  global  formulação alternativa: M >= 5,0, células de 3° no mundo todo, 1990 até hoje,
          treino <= 2017 / validação 2018-2021 / teste 2022+, células com >= 20 eventos no treino.

Entrada : catálogo bruto gerado por download_usgs.py (o arquivo *_min.csv basta).
Saída   : data/processed/
            dataset_{perfil}.parquet       uma linha por (célula, semana): features, rótulo e split
            panel_{perfil}.parquet         agregados semanais por célula (sequências da LSTM)
            cells_{perfil}.csv             metadados das células ativas
            events_clean_{perfil}.parquet  catálogo limpo (com célula e semana)
            summary_{perfil}.json          parâmetros e contagens para o relatório

Uso:
    python src/build_dataset.py                       # perfil pulse
    python src/build_dataset.py --profile global

Definições (iguais ao notebook da Fase 1):
    - semana = bloco de 7 dias a partir da segunda-feira igual ou anterior ao início do período (UTC);
    - amostra (célula, semana t) usa as semanas t-51..t como entrada; rótulo = semana t+1;
    - splits pela semana do rótulo; células ativas definidas só com o período de treino;
    - features de vizinhança: soma das 8 células vizinhas na grade global (longitude dá a volta).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Console do Windows (cp1252) não imprime "≥" e afins: força UTF-8 na saída (sem efeito em Linux/macOS)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

WEEK = pd.Timedelta(days=7)
WINDOWS = (1, 4, 12, 52)   # janelas de agregação, em semanas
SEQ_LEN = 52               # semanas de histórico por amostra (warm-up = SEQ_LEN - 1)

# Círculo de Fogo: caixas (lat_min, lat_max, lon_min, lon_max), como no notebook da Fase 1
RING_BOXES = {
    "Andes (Chile–Colômbia)":          (-60, 15, -85, -65),
    "América Central e México":        (5, 33, -118, -77),
    "Costa oeste EUA/Canadá":          (33, 60, -135, -114),
    "Alasca e Aleutas (leste)":        (50, 66, -180, -135),
    "Aleutas (oeste)":                 (48, 60, 160, 180),
    "Kamchatka e Kurilas":             (40, 62, 140, 165),
    "Japão":                           (24, 46, 125, 150),
    "Izu–Bonin e Marianas":            (10, 35, 135, 150),
    "Taiwan e Filipinas":              (4, 26, 115, 130),
    "Indonésia e Papua-Nova Guiné":    (-12, 6, 95, 155),
    "Melanésia, Tonga, Kermadec, NZ":  (-50, -4, 150, 180),
    "Tonga/Samoa (oeste da linha)":    (-30, -10, -180, -170),
}

PROFILES = {
    "pulse": dict(
        description="P(>= 1 evento M >= 4,0 na célula 2°x2° do Círculo de Fogo na semana seguinte)",
        input="data/raw/usgs_m40_min.csv",
        min_mag=4.0, label_mag=4.0, big_mag=5.0, cell=2.0,
        start="2000-01-01", end="2026-01-01",            # end é exclusivo
        train_last_year=2017, val_last_year=2020,
        min_events_train=10, since_cap=260, region="ring_of_fire",
    ),
    "global": dict(
        description="P(>= 1 evento M >= 5,0 na célula 3°x3°, mundo todo, na semana seguinte)",
        input="data/raw/usgs_m45_min.csv",
        min_mag=4.5, label_mag=5.0, big_mag=6.0, cell=3.0,
        start="1990-01-01", end=None,                     # None = até a última semana completa dos dados
        train_last_year=2017, val_last_year=2021,
        min_events_train=20, since_cap=104, region=None,
    ),
}


# --------------------------------------------------------------------------- limpeza
def clean_events(df: pd.DataFrame, cfg: dict, end: str | None = "profile") -> tuple[pd.DataFrame, dict]:
    """Limpeza do catálogo bruto (DataFrame já carregado). end="profile" usa o fim do perfil; None não corta o fim."""
    stats = {"eventos_brutos": int(len(df))}
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    df = df[df["type"].eq("earthquake")]
    stats["apos_filtro_tipo_earthquake"] = int(len(df))
    df = df.dropna(subset=["time", "latitude", "longitude", "depth", "mag"])
    stats["apos_remover_nulos"] = int(len(df))
    df = df.drop_duplicates(subset="id", keep="last")
    stats["apos_remover_duplicados"] = int(len(df))

    start = pd.Timestamp(cfg["start"], tz="UTC")
    keep = (df["mag"] >= cfg["min_mag"]) & (df["time"] >= start)
    end = cfg["end"] if end == "profile" else end
    if end:
        keep &= df["time"] < pd.Timestamp(end, tz="UTC")
    df = df[keep].copy()
    stats["apos_filtro_magnitude_periodo"] = int(len(df))

    df["depth"] = df["depth"].clip(lower=0.0)
    df["longitude"] = ((df["longitude"] + 180.0) % 360.0) - 180.0
    return df.sort_values("time").reset_index(drop=True), stats


def load_and_clean(path: str | Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    return clean_events(pd.read_csv(path, low_memory=False), cfg)


def week_origin(cfg: dict) -> pd.Timestamp:
    """Segunda-feira igual ou anterior ao início do período: semana 0."""
    start = pd.Timestamp(cfg["start"], tz="UTC")
    return start - pd.Timedelta(days=start.dayofweek)


# --------------------------------------------------------------------------- grade e região
def grid_shape(cell: float) -> tuple[int, int]:
    return int(round(180 / cell)), int(round(360 / cell))


def assign_grid(df: pd.DataFrame, cell: float, origin: pd.Timestamp) -> pd.DataFrame:
    nrows, ncols = grid_shape(cell)
    row = np.clip(np.floor((df["latitude"].to_numpy() + 90.0) / cell).astype(int), 0, nrows - 1)
    col = np.clip(np.floor((df["longitude"].to_numpy() + 180.0) / cell).astype(int), 0, ncols - 1)
    df = df.copy()
    df["cell"] = row * ncols + col
    df["week"] = ((df["time"] - origin) // WEEK).astype(int)
    return df


def cell_centers(cells: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    _, ncols = grid_shape(cell)
    return -90.0 + (cells // ncols + 0.5) * cell, -180.0 + (cells % ncols + 0.5) * cell


def segment_of(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Nome da caixa do Círculo de Fogo que contém cada ponto (primeira que casa), ou None."""
    seg = np.full(len(lat), None, dtype=object)
    for name, (la0, la1, lo0, lo1) in RING_BOXES.items():
        m = (seg == None) & (lat >= la0) & (lat < la1) & (lon >= lo0) & (lon < lo1)  # noqa: E711
        seg[m] = name
    return seg


# --------------------------------------------------------------------------- utilitários
def rolling_sum(A: np.ndarray, w: int) -> np.ndarray:
    """Soma das últimas w colunas (inclusive a atual), por linha."""
    cs = np.cumsum(A, axis=1)
    out = cs.copy()
    out[:, w:] = cs[:, w:] - cs[:, :-w]
    return out


def rolling_max(A: np.ndarray, w: int) -> np.ndarray:
    return pd.DataFrame(A.T).rolling(w, min_periods=1).max().to_numpy().T


def weeks_since(flag: np.ndarray, cap: int) -> np.ndarray:
    """Semanas desde a última coluna com flag > 0 (0 = nesta semana); teto = cap."""
    n_weeks = flag.shape[1]
    idx = np.where(flag > 0, np.arange(n_weeks)[None, :], -10**9)
    last = np.maximum.accumulate(idx, axis=1)
    return np.minimum(np.arange(n_weeks)[None, :] - last, cap)


def neighbors_sum(G: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
    """Soma das 8 células vizinhas em uma grade [nrows*ncols, semanas]; a longitude dá a volta em ±180°."""
    G = G.reshape(nrows, ncols, -1)
    S = np.zeros_like(G)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            sh = np.roll(G, dj, axis=1)
            if di == 1:
                sh = np.concatenate([np.zeros_like(sh[:1]), sh[:-1]], axis=0)
            elif di == -1:
                sh = np.concatenate([sh[1:], np.zeros_like(sh[:1])], axis=0)
            S += sh
    return S.reshape(nrows * ncols, -1)


# --------------------------------------------------------------------------- construção
def weekly_arrays(df: pd.DataFrame, cfg: dict, active: np.ndarray, n_weeks: int) -> dict[str, np.ndarray]:
    """Agregados semanais [célula ativa, semana] para as semanas 0..n_weeks-1, incluindo a vizinhança
    calculada na grade global (todas as células). df já deve ter as colunas cell e week."""
    min_mag, label_mag, big_mag = cfg["min_mag"], cfg["label_mag"], cfg["big_mag"]
    nrows, ncols = grid_shape(cfg["cell"])
    df = df[df["week"] < n_weeks].copy()
    df["is_big"] = (df["mag"] >= big_mag).astype(int)

    # grade global semanal (todas as células), para as features de vizinhança
    g_all = df.groupby(["cell", "week"]).agg(n=("mag", "size"), nbig=("is_big", "sum"))
    N_all = np.zeros((nrows * ncols, n_weeks), np.float32)
    NBIG_all = np.zeros_like(N_all)
    ci = g_all.index.get_level_values("cell").to_numpy()
    wi = g_all.index.get_level_values("week").to_numpy()
    N_all[ci, wi] = g_all["n"].to_numpy()
    NBIG_all[ci, wi] = g_all["nbig"].to_numpy()
    out = {"NVIZ": neighbors_sum(N_all, nrows, ncols)[active].astype(np.float64),
           "NVIZBIG": neighbors_sum(NBIG_all, nrows, ncols)[active].astype(np.float64)}
    del N_all, NBIG_all

    # agregados semanais das células ativas
    df = df[df["cell"].isin(active)].copy()
    df["energy"] = 10.0 ** (1.5 * (df["mag"] - min_mag))          # energia relativa ao evento M = min_mag
    df["is_target"] = (df["mag"] >= label_mag).astype(int)
    df["is_m6"] = (df["mag"] >= 6.0).astype(int)
    wk = df.groupby(["cell", "week"]).agg(
        n=("mag", "size"), max_mag=("mag", "max"), sum_mag=("mag", "sum"),
        energy=("energy", "sum"), sum_depth=("depth", "sum"),
        ntarget=("is_target", "sum"), nbig=("is_big", "sum"), n6=("is_m6", "sum"),
    )

    def grid(col: str) -> np.ndarray:
        return (wk[col].unstack("week")
                .reindex(index=active, columns=np.arange(n_weeks))
                .fillna(0.0)                      # semanas sem eventos -> 0
                .to_numpy(dtype=np.float64))

    for name, col in (("N", "n"), ("MAXM", "max_mag"), ("SUMM", "sum_mag"), ("E", "energy"), ("SUMD", "sum_depth"),
                      ("NT", "ntarget"), ("NBIG", "nbig"), ("N6", "n6")):
        out[name] = grid(col)
    return out


def window_features(A: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    """Features tabulares [célula, semana] a partir dos agregados semanais (semana t inclusa, só passado)."""
    distinct_target = cfg["label_mag"] > cfg["min_mag"]       # senão ntarget == n
    distinct_m6 = cfg["big_mag"] != 6.0                        # senão n6 == nbig
    feats: dict[str, np.ndarray] = {}
    for w in WINDOWS:
        n_w = rolling_sum(A["N"], w)
        feats[f"n_{w}w"] = n_w
        feats[f"maxmag_{w}w"] = rolling_max(A["MAXM"], w)
        with np.errstate(divide="ignore", invalid="ignore"):
            feats[f"meanmag_{w}w"] = np.where(n_w > 0, rolling_sum(A["SUMM"], w) / n_w, 0.0)
            feats[f"meandepth_{w}w"] = np.where(n_w > 0, rolling_sum(A["SUMD"], w) / n_w, 0.0)
        feats[f"logE_{w}w"] = np.log10(1.0 + rolling_sum(A["E"], w))
        if distinct_target:
            feats[f"ntarget_{w}w"] = rolling_sum(A["NT"], w)
        feats[f"nbig_{w}w"] = rolling_sum(A["NBIG"], w)
        if distinct_m6:
            feats[f"n6_{w}w"] = rolling_sum(A["N6"], w)
        feats[f"nviz_{w}w"] = rolling_sum(A["NVIZ"], w)
        feats[f"nvizbig_{w}w"] = rolling_sum(A["NVIZBIG"], w)
    cap = cfg["since_cap"]
    feats["wsl_any"] = weeks_since(A["N"], cap)
    if distinct_target:
        feats["wsl_target"] = weeks_since(A["NT"], cap)
    feats["wsl_big"] = weeks_since(A["NBIG"], cap)
    feats["wktarget_52w"] = rolling_sum((A["NT"] > 0).astype(np.float64), 52)   # semanas com evento-alvo nas últimas 52
    return feats


def panel_columns(A: dict[str, np.ndarray], feats: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    """Variáveis semanais [célula, semana] da sequência da LSTM."""
    distinct_target = cfg["label_mag"] > cfg["min_mag"]
    distinct_m6 = cfg["big_mag"] != 6.0
    N = A["N"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = {"n": N, "max_mag": A["MAXM"], "mean_mag": np.where(N > 0, A["SUMM"] / N, 0.0),
                "log_energy": np.log10(1.0 + A["E"]), "mean_depth": np.where(N > 0, A["SUMD"] / N, 0.0)}
    if distinct_target:
        cols["ntarget"] = A["NT"]
    cols["nbig"] = A["NBIG"]
    if distinct_m6:
        cols["n6"] = A["N6"]
    cols["since_target"] = feats["wsl_target"] if distinct_target else feats["wsl_any"]
    cols["nviz"] = A["NVIZ"]
    cols["nvizbig"] = A["NVIZBIG"]
    return cols


def build(df: pd.DataFrame, cfg: dict, outdir: Path, stats: dict, tag: str) -> dict:
    t0 = time.time()
    cell = cfg["cell"]
    _, ncols = grid_shape(cell)
    origin = week_origin(cfg)
    df = assign_grid(df, cell, origin)
    df.to_parquet(outdir / f"events_clean_{tag}.parquet", index=False)

    data_end = pd.Timestamp(cfg["end"], tz="UTC") if cfg["end"] else df["time"].max()
    n_weeks = int((data_end - origin) // WEEK)                      # semanas 0..n_weeks-1 completas
    week_start = origin + pd.to_timedelta(np.arange(n_weeks) * 7, unit="D")
    week_year = np.asarray(week_start.year)
    df = df[df["week"] < n_weeks]

    # células ativas: contagem só no treino; depois recorte pela região (centro da célula)
    train_weeks = week_year <= cfg["train_last_year"]
    in_train = train_weeks[df["week"].to_numpy()]
    counts_train = df.loc[in_train].groupby("cell").size()
    active = np.sort(counts_train[counts_train >= cfg["min_events_train"]].index.to_numpy())
    lat_c, lon_c = cell_centers(active, cell)
    if cfg["region"] == "ring_of_fire":
        keep = segment_of(lat_c, lon_c) != None  # noqa: E711
        active, lat_c, lon_c = active[keep], lat_c[keep], lon_c[keep]
    n_c = len(active)
    n_cells_with_events = int(df["cell"].nunique())

    A = weekly_arrays(df, cfg, active, n_weeks)
    N, NT = A["N"], A["NT"]
    feats = window_features(A, cfg)
    distinct_target = cfg["label_mag"] > cfg["min_mag"]

    # climatologia da célula, só com o período de treino
    rate_target = (NT[:, train_weeks] > 0).mean(axis=1)
    rate_any = (N[:, train_weeks] > 0).mean(axis=1)
    mean_n = N[:, train_weeks].mean(axis=1)

    # rótulo e tabela longa
    warmup = SEQ_LEN - 1
    t = np.arange(warmup, n_weeks - 1)                 # semanas com 52 semanas de histórico e rótulo disponível
    n_t = len(t)
    Y = (NT[:, t + 1] > 0).astype(np.int8)
    label_year = week_year[t + 1]
    split = np.where(label_year <= cfg["train_last_year"], "train",
                     np.where(label_year <= cfg["val_last_year"], "val", "test"))

    out: dict[str, np.ndarray] = {
        "cell": np.repeat(active, n_t),
        "week": np.tile(t, n_c),
        "lat_c": np.repeat(lat_c, n_t).astype(np.float32),
        "lon_c": np.repeat(lon_c, n_t).astype(np.float32),
        "cos_lon": np.repeat(np.cos(np.radians(lon_c)), n_t).astype(np.float32),
        "sin_lon": np.repeat(np.sin(np.radians(lon_c)), n_t).astype(np.float32),
        "rate_target_train": np.repeat(rate_target, n_t).astype(np.float32),
    }
    if distinct_target:
        out["rate_any_train"] = np.repeat(rate_any, n_t).astype(np.float32)
    out["mean_n_train"] = np.repeat(mean_n, n_t).astype(np.float32)
    for name, arr in feats.items():
        out[name] = arr[:, t].ravel().astype(np.float32)
    out["y"] = Y.ravel()
    out["split"] = np.tile(split, n_c)
    ds = pd.DataFrame(out)
    ws = week_start.tz_localize(None)
    ds["week_start"] = ws[ds["week"].to_numpy()]
    ds["label_week_start"] = ds["week_start"] + WEEK
    ds.to_parquet(outdir / f"dataset_{tag}.parquet", index=False)

    # painel semanal (todas as semanas, inclusive warm-up): base das sequências da LSTM
    panel_cols = panel_columns(A, feats, cfg)
    panel = pd.DataFrame({"cell": np.repeat(active, n_weeks), "week": np.tile(np.arange(n_weeks), n_c)})
    for k, arr in panel_cols.items():
        panel[k] = arr.ravel().astype(np.float32)
    panel["week_start"] = ws[panel["week"].to_numpy()]
    panel["is_train"] = np.tile(train_weeks, n_c)
    panel.to_parquet(outdir / f"panel_{tag}.parquet", index=False)

    cells = pd.DataFrame({
        "cell": active, "row": active // ncols, "col": active % ncols, "lat_c": lat_c, "lon_c": lon_c,
        "segment": segment_of(lat_c, lon_c),
        "n_events_train": counts_train.reindex(active).to_numpy(),
        "rate_target_train": rate_target, "rate_any_train": rate_any, "mean_n_train": mean_n,
    })
    cells.to_csv(outdir / f"cells_{tag}.csv", index=False)

    feature_cols = [c for c in ds.columns if c not in ("cell", "week", "y", "split", "week_start", "label_week_start")]
    summary = {
        "profile": tag, **cfg,
        "week_origin": str(origin.date()),
        "data_end": str(data_end), "n_weeks_complete": n_weeks,
        "limpeza": stats,
        "n_cells_with_events": n_cells_with_events, "n_cells_active": int(n_c),
        "cells_per_segment": cells["segment"].value_counts(dropna=False).to_dict() if cfg["region"] else None,
        "n_features": len(feature_cols), "features": feature_cols,
        "panel_features": list(panel_cols),
        "rows": ds["split"].value_counts().to_dict(),
        "positives": ds.groupby("split")["y"].sum().astype(int).to_dict(),
        "positive_rate": ds.groupby("split")["y"].mean().round(4).to_dict(),
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(outdir / f"summary_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(PROFILES), default="pulse")
    ap.add_argument("--input", default=None, help="catálogo bruto (padrão: o do perfil)")
    ap.add_argument("--cell", type=float, default=None, help="sobrescreve o lado da célula, em graus")
    ap.add_argument("--min-events", type=int, default=None, help="sobrescreve o mínimo de eventos no treino")
    ap.add_argument("--tag", default=None, help="sufixo dos arquivos de saída (padrão: nome do perfil)")
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    cfg = dict(PROFILES[args.profile])
    if args.cell:
        cfg["cell"] = args.cell
    if args.min_events:
        cfg["min_events_train"] = args.min_events
    tag = args.tag or args.profile
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, stats = load_and_clean(args.input or cfg["input"], cfg)
    s = build(df, cfg, outdir, stats, tag)

    print(f"\nPerfil {tag}: {cfg['description']}")
    print(f"  célula {cfg['cell']:g}° | M >= {cfg['min_mag']} | alvo M >= {cfg['label_mag']} | "
          f"grande M >= {cfg['big_mag']} | min {cfg['min_events_train']} eventos no treino")
    print(f"  semanas           : {s['n_weeks_complete']} completas a partir de {s['week_origin']} (até {s['data_end'][:10]})")
    print(f"  limpeza           : {s['limpeza']}")
    print(f"  células ativas    : {s['n_cells_active']} de {s['n_cells_with_events']} com eventos")
    if s["cells_per_segment"]:
        print("  por segmento      : " + ", ".join(f"{k} {v}" for k, v in s["cells_per_segment"].items()))
    print(f"  features          : {s['n_features']} tabulares | sequência: {s['panel_features']}")
    for sp in ("train", "val", "test"):
        print(f"  {sp:<6}: {s['rows'].get(sp, 0):>9,} linhas  {s['positives'].get(sp, 0):>7,} positivos  "
              f"taxa {s['positive_rate'].get(sp, 0):.4f}")
    print(f"  tempo             : {s['build_seconds']} s")


if __name__ == "__main__":
    main()
