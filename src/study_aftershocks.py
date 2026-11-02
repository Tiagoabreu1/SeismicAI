#!/usr/bin/env python3
"""
SeismicAI - estudo: o que explica os picos semanais? (réplicas, produtividade, Omori, Båth)

Pergunta: nos gráficos de eventos por semana, os picos crescem com os anos? São outliers?
Hipótese testada: os picos são sequências de réplicas, e seu tamanho é explicado pela magnitude e
pela profundidade do terremoto principal (lei de produtividade de Utsu), não pelo calendário — com
um efeito residual de melhora na detecção de eventos pequenos.

Para cada terremoto principal (M >= --min-mainshock, sem evento maior ou igual na mesma célula nos
56 dias anteriores nem nos 28 dias seguintes — exclui réplicas e precursores), em todas as células ativas:
  N28      = eventos na célula nos 28 dias seguintes (todas as magnitudes do catálogo)
  N28c     = idem, só na faixa completa (M >= min_mag + 0,5)
  réplica máxima, profundidade, ano
e ajusta por mínimos quadrados:
  log10(N + 1) = a + alpha·(M − Mref) + gamma·[raso: prof. <= 70 km] + beta·(ano − 2000)/10
Reporta alpha (esperado ~0,8–1,0), gamma (rasos produzem mais réplicas), beta (tendência temporal: se
some ao usar a faixa completa, era detecção), a lei de Båth (M principal − maior réplica ≈ 1,2) e a
lei de Omori modificada (p ≈ 1) ajustada por máxima verossimilhança nas maiores sequências.

Uso:
    python src/study_aftershocks.py                       # perfil pulse, principais M >= 5,5
    python src/study_aftershocks.py --profile global --min-mainshock 6.0
Saídas: results/{perfil}/aftershocks.md, aftershocks_mainshocks.csv, fig_aftershocks_productivity.png,
        fig_aftershocks_cell.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seismic_data import paths  # noqa: E402

SHALLOW_KM = 70.0
WINDOW_DAYS = 28


def mainshocks(ev: pd.DataFrame, min_mag: float, quiet_days: int = 56) -> pd.DataFrame:
    rows = []
    for cell, g in ev.groupby("cell"):
        g = g.sort_values("time")
        t = g["time"].dt.tz_convert(None).to_numpy().astype("datetime64[s]")   # numpy datetime64, sem fuso
        m = g["mag"].to_numpy()
        for i in np.flatnonzero(m >= min_mag):
            before = (t >= t[i] - np.timedelta64(quiet_days, "D")) & (t < t[i]) & (m >= m[i])
            if before.any():
                continue                       # é réplica de um maior
            after = (t > t[i]) & (t <= t[i] + np.timedelta64(WINDOW_DAYS, "D"))
            if after.any() and m[after].max() >= m[i]:
                continue                       # é precursor de um maior (ex.: M7,3 dois dias antes de Tohoku)
            rows.append({"cell": cell, "time": g["time"].iloc[i], "mag": m[i], "depth": g["depth"].iloc[i],
                         "lat": g["latitude"].iloc[i], "lon": g["longitude"].iloc[i],
                         "n28": int(after.sum()), "n28c": int((after & (m >= min_mag_c)).sum()),
                         "max_aftershock": float(m[after].max()) if after.any() else np.nan,
                         "aftershock_days": ((t[after] - t[i]) / np.timedelta64(1, "s") / 86400.0).tolist()})
    return pd.DataFrame(rows)


def fit_productivity(df: pd.DataFrame, col: str, mref: float) -> dict:
    y = np.log10(df[col].to_numpy() + 1.0)
    X = np.column_stack([np.ones(len(df)), df["mag"].to_numpy() - mref, (df["depth"].to_numpy() <= SHALLOW_KM).astype(float),
                         (df["time"].dt.year.to_numpy() + df["time"].dt.dayofyear.to_numpy() / 365.25 - 2000.0) / 10.0])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(((y - pred) ** 2).sum())
    r2 = 1.0 - rss / float(((y - y.mean()) ** 2).sum())
    # erros-padrão clássicos
    n, k = X.shape
    sigma2 = rss / max(n - k, 1)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"a": beta[0], "alpha": beta[1], "gamma_raso": beta[2], "beta_decada": beta[3],
            "se_alpha": se[1], "se_gamma": se[2], "se_beta": se[3], "t_beta": beta[3] / se[3] if se[3] > 0 else np.nan,
            "r2": r2, "n": n}


def fit_omori(days: np.ndarray, T: float = WINDOW_DAYS) -> dict:
    """Lei de Omori modificada lambda(t) = K (t + c)^-p por máxima verossimilhança (processo de Poisson não homogêneo)."""
    days = np.asarray(days, dtype=float)
    days = days[(days > 0) & (days <= T)]
    if len(days) < 15:
        return {}

    def nll(theta):
        logK, logc, p = theta
        K, c = np.exp(logK), np.exp(logc)
        integral = K * ((T + c) ** (1 - p) - c ** (1 - p)) / (1 - p) if abs(p - 1) > 1e-6 else K * np.log((T + c) / c)
        return -(len(days) * logK - p * np.log(days + c).sum() - integral)

    best = None
    for p0 in (0.8, 1.0, 1.2):
        r = minimize(nll, x0=[np.log(len(days) / 5.0), np.log(0.05), p0], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    logK, logc, p = best.x
    return {"K": float(np.exp(logK)), "c_dias": float(np.exp(logc)), "p": float(p), "n": int(len(days))}


def main() -> None:
    global min_mag_c
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--min-mainshock", type=float, default=None, help="padrão: big_mag + 0,5 (pulse: 5,5)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    files = paths(args.profile)
    cfg = json.loads(files["summary"].read_text(encoding="utf-8"))
    outdir = Path(args.outdir or f"results/{args.profile}")
    outdir.mkdir(parents=True, exist_ok=True)
    min_mag = float(cfg["min_mag"])
    min_mag_c = min_mag + 0.5
    mmin = args.min_mainshock or float(cfg["big_mag"]) + 0.5
    mref = mmin

    cells = pd.read_csv(files["cells"])
    ev = pd.read_parquet(files["events"])
    ev = ev[ev["cell"].isin(cells["cell"])].copy()
    ev["time"] = pd.to_datetime(ev["time"], utc=True)
    ms = mainshocks(ev, mmin)
    ms = ms.merge(cells[["cell", "segment", "lat_c", "lon_c"]], on="cell", how="left")
    ms.drop(columns=["aftershock_days"]).to_csv(outdir / "aftershocks_mainshocks.csv", index=False)
    print(f"{len(ms)} terremotos principais M >= {mmin:g} em {ms['cell'].nunique()} células")

    fit_all = fit_productivity(ms, "n28", mref)
    fit_c = fit_productivity(ms, "n28c", mref)
    bath = (ms["mag"] - ms["max_aftershock"]).dropna()
    top = ms.sort_values("n28", ascending=False).head(6)
    omori = {f"{r.time:%Y-%m-%d} M{r.mag:.1f} ({r.segment}, {r.depth:.0f} km)": fit_omori(r.aftershock_days)
             for r in top.itertuples()}

    # ---------------- figuras
    fig, ax = plt.subplots(figsize=(7.5, 5))
    yrs = ms["time"].dt.year
    for shallow, mk, lab in ((True, "o", f"raso (≤ {SHALLOW_KM:.0f} km)"), (False, "^", f"profundo (> {SHALLOW_KM:.0f} km)")):
        d = ms[(ms["depth"] <= SHALLOW_KM) == shallow]
        sc = ax.scatter(d["mag"] + np.random.default_rng(0).uniform(-0.04, 0.04, len(d)), np.log10(d["n28"] + 1), c=yrs[d.index],
                        cmap="viridis", marker=mk, s=28, alpha=.75, label=lab, vmin=yrs.min(), vmax=yrs.max())
    mm = np.linspace(ms["mag"].min(), ms["mag"].max(), 50)
    mid_year = (ms["time"].dt.year.mean() - 2000) / 10
    for shallow, ls in ((1.0, "-"), (0.0, "--")):
        ax.plot(mm, fit_all["a"] + fit_all["alpha"] * (mm - mref) + fit_all["gamma_raso"] * shallow + fit_all["beta_decada"] * mid_year, "k" + ls, lw=1.2,
                label=f"ajuste {'raso' if shallow else 'profundo'}: α = {fit_all['alpha']:.2f}")
    ax.set(xlabel="magnitude do terremoto principal", ylabel=f"log10(réplicas M ≥ {min_mag:g} em 28 dias + 1)",
           title=f"Produtividade de réplicas — {len(ms)} sequências, {ms['cell'].nunique()} células")
    fig.colorbar(sc, ax=ax, label="ano")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(outdir / "fig_aftershocks_productivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    busiest = ev.groupby("cell").size().idxmax()
    g = ev[ev["cell"] == busiest].set_index("time").sort_index()
    weekly = g["mag"].resample("W-MON", label="left", closed="left").size()
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.bar(weekly.index, weekly.to_numpy(), width=7, color="#9fb3c8")
    mb = ms[ms["cell"] == busiest].sort_values("n28", ascending=False)
    ymax = float(weekly.max()) * 1.25
    for k, r in enumerate(mb.itertuples()):
        col = "#c1121f" if r.depth <= SHALLOW_KM else "#1d3557"
        ax.axvline(r.time, color=col, lw=0.6, alpha=.45)
        if k < 8:                               # rótulo só nas 8 sequências mais produtivas da célula
            ax.annotate(f"M{r.mag:.1f} · {r.depth:.0f} km · {r.n28} rép.", (r.time, ymax * (0.95 - 0.08 * (k % 4))), ha="center", fontsize=7, color=col)
    ax.set_ylim(0, ymax)
    info = cells.set_index("cell").loc[busiest]
    ax.set(title=f"Célula mais ativa ({info['lat_c']:.0f}°, {info['lon_c']:.0f}°, {info['segment']}): eventos M ≥ {min_mag:g} por semana e terremotos principais M ≥ {mmin:g} "
                 f"(vermelho = raso, azul = profundo)", ylabel="eventos / semana")
    fig.savefig(outdir / "fig_aftershocks_cell.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- relatório
    md = [f"# Picos semanais, réplicas e produtividade (perfil {args.profile})\n",
          f"{len(ms)} terremotos principais M ≥ {mmin:g} (primeiro ou maior da sequência) em {ms['cell'].nunique()} células ativas; "
          f"réplicas contadas na mesma célula nos {WINDOW_DAYS} dias seguintes.\n",
          "## Regressão: log10(réplicas + 1) = a + α·(M − Mref) + γ·[raso ≤ 70 km] + β·(ano − 2000)/10\n",
          "| Contagem | α (por unidade de M) | γ (raso) | β (por década) | t de β | R² | n |", "|---|---|---|---|---|---|---|"]
    for name, f in ((f"todas as réplicas M ≥ {min_mag:g}", fit_all), (f"faixa completa M ≥ {min_mag_c:g}", fit_c)):
        md.append(f"| {name} | {f['alpha']:.2f} ± {f['se_alpha']:.2f} | {f['gamma_raso']:+.2f} ± {f['se_gamma']:.2f} | {f['beta_decada']:+.3f} ± {f['se_beta']:.3f} | {f['t_beta']:.1f} | {f['r2']:.2f} | {f['n']} |")
    md += ["\nLeitura: α ≈ 1 significa ~10× mais réplicas por unidade de magnitude (lei de produtividade); γ > 0 significa que "
           "terremotos rasos produzem mais réplicas que os profundos; β é a tendência temporal, descontadas magnitude e profundidade — "
           "se cai ao usar só a faixa completa, a parte que sumiu era melhora de detecção.",
           f"\n**Lei de Båth**: M principal − maior réplica = {bath.mean():.2f} em média (mediana {bath.median():.2f}, n = {len(bath)}); esperado ≈ 1,2.\n",
           "## Lei de Omori modificada nas maiores sequências (máxima verossimilhança, 28 dias)\n",
           "| Sequência | réplicas | p | c (dias) | K |", "|---|---|---|---|---|"]
    for k, v in omori.items():
        md.append(f"| {k} | {v.get('n', '—')} | {v.get('p', float('nan')):.2f} | {v.get('c_dias', float('nan')):.3f} | {v.get('K', float('nan')):.1f} |" if v else f"| {k} | poucas réplicas | — | — | — |")
    md += ["\nLeitura: p ≈ 1 é o decaimento clássico de Omori (a taxa cai como 1/t); c é o atraso aparente, inflado pela incompletude nas primeiras horas.\n",
           "## As 10 maiores sequências do catálogo\n", "| Data | Célula / segmento | M | prof. (km) | réplicas 28 d | maior réplica |", "|---|---|---|---|---|---|"]
    for r in ms.sort_values("n28", ascending=False).head(10).itertuples():
        md.append(f"| {r.time:%Y-%m-%d} | {r.lat_c:.0f}°, {r.lon_c:.0f}° · {r.segment} | {r.mag:.1f} | {r.depth:.0f} | {r.n28} | {r.max_aftershock:.1f} |")
    (outdir / "aftershocks.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
