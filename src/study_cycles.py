#!/usr/bin/env python3
"""
SeismicAI - estudo: existe "ciclo sísmico" nas células? Intervalos maiores trazem terremotos maiores?

Hipóteses testadas, sobre os terremotos PRINCIPAIS de cada célula (réplicas e precursores excluídos,
como em study_aftershocks.py), em ordem cronológica:
  H-tempo    (previsível no tempo)         : quanto maior a espera desde o principal anterior, maior o próximo
                                             -> correlação de Spearman entre log(intervalo anterior) e magnitude
  H-desliz.  (previsível no deslizamento)  : quanto maior o terremoto, maior a espera até o próximo
                                             -> correlação entre magnitude e log(intervalo seguinte)
  H-memória                                : magnitudes consecutivas se parecem? intervalos consecutivos crescem?
                                             -> correlação M_i x M_{i-1} e log(dt_i) x log(dt_{i-1})
  H-tendência                              : a magnitude máxima anual sobe ao longo dos anos?
                                             -> Spearman (ano, máx. anual) por segmento e no total; Mann-Kendall simples
Também compara os intervalos com um processo de Poisson (coeficiente de variação ≈ 1 = sem ciclo).

Uso:
    python src/study_cycles.py                        # perfil pulse, principais M >= 5,5
Saídas: results/{perfil}/cycles.md, fig_cycles.png
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seismic_data import paths  # noqa: E402
import study_aftershocks as sa  # noqa: E402


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--min-mainshock", type=float, default=None, help="padrão: big_mag + 0,5")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    files = paths(args.profile)
    cfg = json.loads(files["summary"].read_text(encoding="utf-8"))
    outdir = Path(args.outdir or f"results/{args.profile}")
    outdir.mkdir(parents=True, exist_ok=True)
    mmin = args.min_mainshock or float(cfg["big_mag"]) + 0.5
    sa.min_mag_c = float(cfg["min_mag"]) + 0.5

    cells = pd.read_csv(files["cells"])
    ev = pd.read_parquet(files["events"])
    ev = ev[ev["cell"].isin(cells["cell"])].copy()
    ev["time"] = pd.to_datetime(ev["time"], utc=True)
    ms = sa.mainshocks(ev, mmin).drop(columns=["aftershock_days"]).merge(cells[["cell", "segment"]], on="cell", how="left")
    ms = ms.sort_values(["cell", "time"]).reset_index(drop=True)
    g = ms.groupby("cell")
    ms["dt_before"] = g["time"].diff().dt.total_seconds() / 86400.0                 # dias desde o principal anterior
    ms["dt_after"] = -g["time"].diff(-1).dt.total_seconds() / 86400.0               # dias até o próximo
    ms["mag_prev"] = g["mag"].shift(1)
    ms["dt_prev"] = g["dt_before"].shift(1)
    print(f"{len(ms)} principais M >= {mmin:g} em {ms['cell'].nunique()} células; pares consecutivos: {int(ms['dt_before'].notna().sum())}")

    # versão "dentro da célula": subtrai a média da célula, para separar o ritmo próprio de cada célula de uma tendência
    ms["ldt"] = np.log10(ms["dt_before"])
    ms["ldt_c"] = ms["ldt"] - ms.groupby("cell")["ldt"].transform("mean")
    ms["ldt_prev_c"] = ms.groupby("cell")["ldt_c"].shift(1)
    ms["mag_c"] = ms["mag"] - ms.groupby("cell")["mag"].transform("mean")
    ms["seq_idx"] = g.cumcount()
    tests = {
        "H-tempo: log(intervalo anterior) × magnitude": spearman(np.log10(ms["dt_before"].to_numpy()), ms["mag"].to_numpy()),
        "H-tempo dentro da célula (ambos centrados na média da célula)": spearman(ms["ldt_c"].to_numpy(), ms["mag_c"].to_numpy()),
        "H-desliz.: magnitude × log(intervalo seguinte)": spearman(ms["mag"].to_numpy(), np.log10(ms["dt_after"].to_numpy())),
        "H-memória: magnitude × magnitude anterior": spearman(ms["mag"].to_numpy(), ms["mag_prev"].to_numpy()),
        "H-memória: log(intervalo) × log(intervalo anterior)": spearman(np.log10(ms["dt_before"].to_numpy()), np.log10(ms["dt_prev"].to_numpy())),
        "H-memória dentro da célula: intervalo × intervalo anterior (centrados)": spearman(ms["ldt_c"].to_numpy(), ms["ldt_prev_c"].to_numpy()),
        "Intervalos crescem ao longo da sequência? (ordem na célula × intervalo centrado)": spearman(ms["seq_idx"].to_numpy(dtype=float), ms["ldt_c"].to_numpy()),
        "Magnitudes crescem ao longo da sequência? (ordem na célula × magnitude centrada)": spearman(ms["seq_idx"].to_numpy(dtype=float), ms["mag_c"].to_numpy()),
    }
    # mesma coisa só em células com muitos principais (onde um ciclo teria chance de aparecer)
    busy = ms.groupby("cell")["mag"].transform("size") >= 20
    tests_busy = {k.replace(":", " (células com ≥ 20 principais):"): v for k, v in {
        "H-tempo": spearman(np.log10(ms.loc[busy, "dt_before"].to_numpy()), ms.loc[busy, "mag"].to_numpy()),
        "H-desliz.": spearman(ms.loc[busy, "mag"].to_numpy(), np.log10(ms.loc[busy, "dt_after"].to_numpy())),
    }.items()}

    # intervalos vs Poisson: coeficiente de variação por célula (Poisson -> CV ≈ 1; ciclo regular -> CV << 1; agrupado -> CV > 1)
    cv = ms.dropna(subset=["dt_before"]).groupby("cell")["dt_before"].agg(lambda s: s.std() / s.mean() if len(s) >= 10 else np.nan).dropna()

    # tendência da magnitude máxima anual
    ms["year"] = ms["time"].dt.year
    years = np.arange(int(ms["year"].min()), int(ms["year"].max()) + 1)
    rows_tr = []
    for seg, d in [("Todo o Círculo de Fogo", ms)] + [(s, d) for s, d in ms.groupby("segment")]:
        amax = d.groupby("year")["mag"].max().reindex(years)
        r, p, n = spearman(years.astype(float), amax.to_numpy(dtype=float))
        cnt = d.groupby("year").size().reindex(years, fill_value=0)
        rc, pc, _ = spearman(years.astype(float), cnt.to_numpy(dtype=float))
        rows_tr.append({"segmento": seg, "n_principais": len(d), "rho_maxanual_ano": r, "p_max": p, "rho_contagem_ano": rc, "p_cont": pc})
    tr = pd.DataFrame(rows_tr)

    # ---------------- figura
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    d = ms.dropna(subset=["dt_before"])
    axes[0].scatter(d["dt_before"], d["mag"] + np.random.default_rng(0).uniform(-0.04, 0.04, len(d)), s=6, alpha=.3, color="#1d3557")
    axes[0].set_xscale("log")
    axes[0].set(xlabel="dias desde o principal anterior na célula", ylabel="magnitude do principal",
                title=f"Espera × magnitude (ρ = {tests['H-tempo: log(intervalo anterior) × magnitude'][0]:+.3f})")
    axes[1].hist(np.clip(cv, 0, 4), bins=30, color="#fd8d3c")
    axes[1].axvline(1, color="k", ls="--", lw=1, label="Poisson (CV = 1)")
    axes[1].set(xlabel="coef. de variação dos intervalos (por célula)", ylabel="células", title="Intervalos: regular (<1), Poisson (1), agrupado (>1)")
    axes[1].legend(frameon=False)
    amax = ms.groupby("year")["mag"].max().reindex(years)
    axes[2].plot(years, amax, "o-", color="#c1121f", ms=4)
    axes[2].set(xlabel="ano", ylabel="magnitude máxima no ano", title=f"Máx. anual, Círculo de Fogo (ρ = {tr.iloc[0]['rho_maxanual_ano']:+.2f}, p = {tr.iloc[0]['p_max']:.2f})")
    fig.tight_layout()
    fig.savefig(outdir / "fig_cycles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- relatório
    md = [f"# Ciclo sísmico? Intervalos, magnitudes e tendências (perfil {args.profile})\n",
          f"{len(ms)} terremotos principais M ≥ {mmin:g} (réplicas e precursores excluídos) em {ms['cell'].nunique()} células; "
          f"{int(ms['dt_before'].notna().sum())} pares consecutivos na mesma célula.\n",
          "## Correlações de Spearman (ρ entre −1 e 1; p < 0,05 = significativo; |ρ| < 0,1 = desprezível mesmo se significativo)\n",
          "| Hipótese | ρ | p | n |", "|---|---|---|---|"]
    for k, (r, p, n) in {**tests, **tests_busy}.items():
        md.append(f"| {k} | {r:+.3f} | {p:.3g} | {n} |")
    md += [f"\n## Intervalos entre principais vs. processo de Poisson\n",
           f"Coeficiente de variação por célula (≥ 10 principais): mediana {cv.median():.2f}, quartis {cv.quantile(.25):.2f}–{cv.quantile(.75):.2f} em {len(cv)} células. "
           "CV ≈ 1 é um processo sem memória (Poisson); CV < 1 indicaria regularidade (ciclo); CV > 1, agrupamento residual.\n",
           "## Tendência da magnitude máxima anual e da contagem anual de principais\n",
           "| Segmento | principais | ρ (máx. anual × ano) | p | ρ (contagem × ano) | p |", "|---|---|---|---|---|---|"]
    for r in tr.itertuples():
        md.append(f"| {r.segmento} | {r.n_principais} | {r.rho_maxanual_ano:+.2f} | {r.p_max:.2f} | {r.rho_contagem_ano:+.2f} | {r.p_cont:.3f} |")
    md += ["\nLeitura: ρ positivo e p < 0,05 na coluna da máxima anual indicaria magnitudes máximas crescendo com os anos; "
           "na coluna da contagem, mais principais por ano (que pode ser detecção/revisão de magnitudes, não só sismicidade)."]
    (outdir / "cycles.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
