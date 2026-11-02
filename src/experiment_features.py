#!/usr/bin/env python3
"""
SeismicAI - experimento: mais variáveis ajudam?

Constrói uma segunda camada de variáveis por (célula, semana), com motivação física, a partir do
catálogo limpo, e compara três conjuntos de entradas nos mesmos splits temporais:
  base         as 45 variáveis usadas no projeto
  base+extras  as 45 + as novas
  só extras    só as novas
com dois modelos rápidos: regressão logística (linear) e gradient boosting (HistGradientBoosting,
que extrai praticamente tudo o que uma tabela tem; nas redes do projeto deu o mesmo resultado).

Variáveis novas
  memória longa    : n_104w, n_260w, maxmag_104w, maxmag_260w, wsl_m6, wsl_m7
  espaço amplo     : nviz2_1w, nviz2_4w, nviz2big_4w (anel 5x5 = 24 células); nseg_1w, nseg_4w (segmento)
  resolução diária : n_1d, n_3d, maxmag_3d, days_since_any, days_since_big
  taxa tipo ETAS   : etas_cell, etas_viz  -> sum_i 10^(alpha (m_i - m0)) (dt_i + c)^(-p), eventos dos últimos 365 dias
  composição       : frac45_52w (proxy do valor-b), frac_deep_52w (profundidade > 70 km), accel_4_12, ratio_4_52

Uso:
    python src/experiment_features.py                 # perfil pulse -> results/pulse/feature_experiment.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import WEEK, grid_shape, rolling_max, rolling_sum, week_origin, weeks_since  # noqa: E402
from seismic_data import load_tabular, paths  # noqa: E402

ALPHA, P_OMORI, C_OMORI, ETAS_DAYS = 1.0, 1.1, 0.02, 365.0
LOG_COLS_PREFIX = ("n_", "nviz", "nseg", "n_1d", "n_3d", "etas", "days_since")


def ring_sum(G: np.ndarray, nrows: int, ncols: int, r: int) -> np.ndarray:
    """Soma das células no quadrado (2r+1)x(2r+1) em volta, sem o centro; longitude dá a volta."""
    G = G.reshape(nrows, ncols, -1)
    S = np.zeros_like(G)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di == 0 and dj == 0:
                continue
            sh = np.roll(G, dj, axis=1)
            if di > 0:
                sh = np.concatenate([np.zeros_like(sh[:di]), sh[:-di]], axis=0)
            elif di < 0:
                sh = np.concatenate([sh[-di:], np.zeros_like(sh[:-di])], axis=0)
            S += sh
    return S.reshape(nrows * ncols, -1)


def daily_and_etas(ev: pd.DataFrame, active: np.ndarray, week_end_s: np.ndarray, big_mag: float, m0: float,
                   neighbor_cells: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Features em resolução diária e taxa tipo ETAS, avaliadas no fim de cada semana."""
    n_c, n_w = len(active), len(week_end_s)
    out = {k: np.zeros((n_c, n_w), np.float32) for k in ("n_1d", "n_3d", "maxmag_3d", "etas_cell", "etas_viz")}
    out["days_since_any"] = np.full((n_c, n_w), 3650.0, np.float32)
    out["days_since_big"] = np.full((n_c, n_w), 3650.0, np.float32)
    by_cell = {c: g for c, g in ev.groupby("cell")}
    t0 = time.time()

    def etas(ts: np.ndarray, mags: np.ndarray) -> np.ndarray:
        """Taxa no fim de cada semana: eventos dos últimos ETAS_DAYS dias, peso 10^(alpha(m-m0)) (dt + c)^-p."""
        rate = np.zeros(n_w, np.float64)
        if len(ts) == 0:
            return rate
        w = 10.0 ** (ALPHA * (mags - m0))
        hi = np.searchsorted(ts, week_end_s, side="right")
        lo = np.searchsorted(ts, week_end_s - ETAS_DAYS * 86400.0, side="left")
        for j in range(n_w):
            if hi[j] > lo[j]:
                dt = (week_end_s[j] - ts[lo[j]:hi[j]]) / 86400.0
                rate[j] = np.sum(w[lo[j]:hi[j]] * (dt + C_OMORI) ** (-P_OMORI))
        return rate

    for i, c in enumerate(active):
        g = by_cell.get(int(c))
        if g is not None:
            ts = g["time"].values.astype("datetime64[s]").astype(np.int64).astype(np.float64)
            mags = g["mag"].to_numpy(np.float64)
            order = np.argsort(ts)
            ts, mags = ts[order], mags[order]
            hi = np.searchsorted(ts, week_end_s, side="right")
            lo3 = np.searchsorted(ts, week_end_s - 3 * 86400.0, side="left")
            lo1 = np.searchsorted(ts, week_end_s - 1 * 86400.0, side="left")
            out["n_3d"][i] = hi - lo3
            out["n_1d"][i] = hi - lo1
            for j in np.flatnonzero(hi > lo3):
                out["maxmag_3d"][i, j] = mags[lo3[j]:hi[j]].max()
            last = np.where(hi > 0, ts[np.maximum(hi - 1, 0)], -1e18)
            out["days_since_any"][i] = np.minimum((week_end_s - last) / 86400.0, 3650.0)
            big = mags >= big_mag
            if big.any():
                tb = ts[big]
                hb = np.searchsorted(tb, week_end_s, side="right")
                lastb = np.where(hb > 0, tb[np.maximum(hb - 1, 0)], -1e18)
                out["days_since_big"][i] = np.minimum((week_end_s - lastb) / 86400.0, 3650.0)
            out["etas_cell"][i] = etas(ts, mags)
        nb = [by_cell[int(k)] for k in neighbor_cells[int(c)] if int(k) in by_cell]
        if nb:
            gg = pd.concat(nb)
            ts = gg["time"].values.astype("datetime64[s]").astype(np.int64).astype(np.float64)
            mags = gg["mag"].to_numpy(np.float64)
            order = np.argsort(ts)
            out["etas_viz"][i] = etas(ts[order], mags[order])
        if i % 100 == 0:
            print(f"  diárias/ETAS: {i}/{n_c} células ({time.time() - t0:.0f}s)", flush=True)
    return out


def neighbors_of(cell: int, nrows: int, ncols: int) -> np.ndarray:
    r, c = cell // ncols, cell % ncols
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and 0 <= r + dr < nrows:
                out.append((r + dr) * ncols + (c + dc) % ncols)
    return np.array(out)


def build_extras(profile: str) -> tuple[pd.DataFrame, list[str], dict]:
    files = paths(profile)
    cfg = json.loads(files["summary"].read_text(encoding="utf-8"))
    ds, feature_cols, _ = load_tabular(files["dataset"])
    cells = pd.read_csv(files["cells"])
    ev = pd.read_parquet(files["events"])
    active = cells["cell"].to_numpy()
    assert np.array_equal(np.sort(active), active)
    cell_deg, big_mag, m0 = float(cfg["cell"]), float(cfg["big_mag"]), float(cfg["min_mag"])
    nrows, ncols = grid_shape(cell_deg)
    origin = week_origin(cfg)
    n_weeks = int(ds["week"].max()) + 2
    ev = ev[ev["week"] < n_weeks].copy()
    t = np.arange(51, n_weeks - 1)
    n_c, n_t = len(active), len(t)
    assert len(ds) == n_c * n_t and np.array_equal(ds["cell"].to_numpy()[::n_t], active), "ordem do dataset diferente da esperada"

    # ---- grade global semanal (todas as células)
    def full(mask=None, col=None, agg="size"):
        d = ev if mask is None else ev[mask]
        g = d.groupby(["cell", "week"]).size() if col is None else d.groupby(["cell", "week"])[col].agg(agg)
        A = np.zeros((nrows * ncols, n_weeks), np.float32)
        A[g.index.get_level_values(0).to_numpy(), g.index.get_level_values(1).to_numpy()] = g.to_numpy()
        return A

    N_all, NBIG_all = full(), full(ev["mag"] >= big_mag)
    N, NBIG = N_all[active].astype(np.float64), NBIG_all[active].astype(np.float64)
    MAXM = full(col="mag", agg="max")[active].astype(np.float64)
    N6 = full(ev["mag"] >= 6.0)[active].astype(np.float64)
    N7 = full(ev["mag"] >= 7.0)[active].astype(np.float64)
    N45 = full(ev["mag"] >= m0 + 0.5)[active].astype(np.float64)
    NDEEP = full(ev["depth"] > 70.0)[active].astype(np.float64)
    VIZ2 = ring_sum(N_all, nrows, ncols, 2)[active].astype(np.float64) - ring_sum(N_all, nrows, ncols, 1)[active].astype(np.float64)
    VIZ2BIG = ring_sum(NBIG_all, nrows, ncols, 2)[active].astype(np.float64) - ring_sum(NBIG_all, nrows, ncols, 1)[active].astype(np.float64)

    # ---- atividade do segmento (soma sobre as células ativas do mesmo segmento, sem a própria)
    seg = cells["segment"].fillna("fora").to_numpy()
    SEG = np.zeros_like(N)
    for s in np.unique(seg):
        m = seg == s
        SEG[m] = N[m].sum(axis=0, keepdims=True) - N[m]

    X: dict[str, np.ndarray] = {}
    for w in (104, 260):
        X[f"n_{w}w"] = rolling_sum(N, w)
        X[f"maxmag_{w}w"] = rolling_max(MAXM, w)
    X["wsl_m6"] = weeks_since(N6, 260)
    X["wsl_m7"] = weeks_since(N7, 520)
    X["nviz2_1w"] = VIZ2
    X["nviz2_4w"] = rolling_sum(VIZ2, 4)
    X["nviz2big_4w"] = rolling_sum(VIZ2BIG, 4)
    X["nseg_1w"] = SEG
    X["nseg_4w"] = rolling_sum(SEG, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        n52 = rolling_sum(N, 52)
        X["frac45_52w"] = np.where(n52 > 0, rolling_sum(N45, 52) / n52, 0.0)
        X["frac_deep_52w"] = np.where(n52 > 0, rolling_sum(NDEEP, 52) / n52, 0.0)
    n4, n12 = rolling_sum(N, 4), rolling_sum(N, 12)
    X["accel_4_12"] = n4 - n12 / 3.0
    X["ratio_4_52"] = n4 / (n52 / 13.0 + 1.0)

    week_end_s = ((origin + pd.to_timedelta((t + 1) * 7, unit="D")).values.astype("datetime64[s]").astype(np.int64)).astype(np.float64)
    nb_map = {int(c): neighbors_of(int(c), nrows, ncols) for c in active}
    print("construindo features diárias e ETAS (leva alguns minutos)…", flush=True)
    daily = daily_and_etas(ev, active, week_end_s, big_mag, m0, nb_map)

    extras = pd.DataFrame({k: A[:, t].ravel().astype(np.float32) for k, A in X.items()})
    for k, A in daily.items():
        extras[k] = A.ravel().astype(np.float32)
    extra_cols = list(extras.columns)
    extras["cell"], extras["week"], extras["y"], extras["split"] = ds["cell"].to_numpy(), ds["week"].to_numpy(), ds["y"].to_numpy(), ds["split"].to_numpy()
    base = ds[feature_cols].reset_index(drop=True)
    return pd.concat([base, extras.reset_index(drop=True)], axis=1), extra_cols, {"feature_cols": feature_cols, "n_cells": n_c, "weeks": n_t}


def prep(Xtr: np.ndarray, Xo: list[np.ndarray], cols: list[str]):
    """log1p nas contagens/taxas + z-score ajustado no treino."""
    log = np.array([c.startswith(LOG_COLS_PREFIX) for c in cols])
    def pre(X):
        X = X.astype(np.float64, copy=True)
        X[:, log] = np.log1p(np.maximum(X[:, log], 0))
        return X
    Ztr = pre(Xtr)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    return (Ztr - mu) / sd, [(pre(X) - mu) / sd for X in Xo]


def evaluate(y, p, strata: dict[str, np.ndarray]) -> dict:
    out = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}
    for name, m in strata.items():
        out[f"pr_auc|{name}"] = average_precision_score(y[m], p[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = Path(args.outdir or f"results/{args.profile}")
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, extra_cols, info = build_extras(args.profile)
    base_cols = info["feature_cols"]
    df.to_parquet(outdir / "feature_experiment_dataset.parquet", index=False)
    print(f"features: base {len(base_cols)} | extras {len(extra_cols)} | linhas {len(df):,} ({time.time() - t0:.0f}s)")

    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    y_tr, y_te = tr["y"].to_numpy(), te["y"].to_numpy()
    q25 = te["rate_target_train"].quantile(0.25)
    strata = {"com M≥big recente (4 sem.)": (te["nbig_4w"] > 0).to_numpy(), "células quietas": (te["rate_target_train"] < q25).to_numpy()}
    sets = {"base": base_cols, "base + extras": base_cols + extra_cols, "só extras": extra_cols}
    rows, hgb_models = [], {}
    for sname, cols in sets.items():
        Xtr, [Xte] = prep(tr[cols].to_numpy(), [te[cols].to_numpy()], cols)
        t1 = time.time()
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y_tr)
        r = evaluate(y_te, lr.predict_proba(Xte)[:, 1], strata)
        rows.append({"conjunto": sname, "modelo": "regressão logística", "n_features": len(cols), **r, "segundos": round(time.time() - t1)})
        t1 = time.time()
        hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06, max_leaf_nodes=31, early_stopping=True,
                                             validation_fraction=0.1, random_state=0).fit(tr[cols].to_numpy(), y_tr)
        r = evaluate(y_te, hgb.predict_proba(te[cols].to_numpy())[:, 1], strata)
        rows.append({"conjunto": sname, "modelo": "gradient boosting", "n_features": len(cols), **r, "segundos": round(time.time() - t1)})
        hgb_models[sname] = hgb
        print(f"{sname:<14} LR  ROC {rows[-2]['roc_auc']:.4f} PR {rows[-2]['pr_auc']:.4f} | HGB ROC {rows[-1]['roc_auc']:.4f} PR {rows[-1]['pr_auc']:.4f}", flush=True)

    # importância por permutação das extras no HGB (amostra do teste)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(te), min(60000, len(te)), replace=False)
    cols = base_cols + extra_cols
    Xs, ys = te[cols].to_numpy()[idx], y_te[idx]
    hgb = hgb_models["base + extras"]
    base_auc = roc_auc_score(ys, hgb.predict_proba(Xs)[:, 1])
    imp = []
    for j, c in enumerate(cols):
        Xp = Xs.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp.append((c, base_auc - roc_auc_score(ys, hgb.predict_proba(Xp)[:, 1]), c in extra_cols))
    imp = sorted(imp, key=lambda x: -x[1])

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "feature_experiment.csv", index=False)
    md = ["# Experimento: mais variáveis ajudam? (perfil %s, teste)\n" % args.profile,
          f"{len(base_cols)} variáveis base + {len(extra_cols)} novas. Mesmos splits temporais; gradient boosting com early stopping.\n",
          "| Conjunto | Modelo | nº features | ROC-AUC | PR-AUC | Brier | PR-AUC c/ M≥big recente | PR-AUC células quietas |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['conjunto']} | {r['modelo']} | {r['n_features']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['brier']:.4f} | "
                  f"{r['pr_auc|com M≥big recente (4 sem.)']:.3f} | {r['pr_auc|células quietas']:.3f} |")
    md += ["\n**Importância por permutação (gradient boosting, base + extras; queda de ROC-AUC)** — as 15 maiores; (*) = variável nova\n",
           "| Variável | Queda de ROC-AUC |", "|---|---|"]
    md += [f"| {c}{' (*)' if e else ''} | {v:+.4f} |" for c, v, e in imp[:15]]
    md += ["\nNovas variáveis entre as 15 mais importantes: " + (", ".join(c for c, v, e in imp[:15] if e) or "nenhuma") + "."]
    (outdir / "feature_experiment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\ntotal {time.time() - t0:.0f}s | {outdir / 'feature_experiment.md'}")


if __name__ == "__main__":
    main()
