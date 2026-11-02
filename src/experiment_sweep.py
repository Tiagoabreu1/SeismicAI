#!/usr/bin/env python3
"""
SeismicAI - varredura de hiperparâmetros, sementes e experimentos dirigidos (para rodar na GPU).

Responde a "treinar mais / de outro jeito melhora?" com números:
  1. sementes   : a mesma configuração com 5 sementes -> média ± desvio (barras de erro para a tabela);
  2. varredura  : variações de tamanho, camadas, dropout, taxa de aprendizado, lote e paciência da LSTM;
  3. dirigidos  : sequência de 104 semanas (memória longa), perda ponderada (pos_weight, como o notebook da
                  Fase 1 sugeriu) e a MLP com as mesmas sementes.
Cada execução é um `train_nn.py` com seu próprio diretório em results/{perfil}/sweep/<nome>/; ao final,
tudo é consolidado em results/{perfil}/sweep.md.

Uso:
    python src/experiment_sweep.py                  # ~30 min em GPU (LSTM ~1 min por execução)
    python src/experiment_sweep.py --only seeds     # ou: sweep, targeted
    python src/experiment_sweep.py --dry-run        # só lista os comandos
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

BASE_LSTM = ["--model", "lstm", "--hidden", "64", "--layers", "1", "--dropout", "0.2", "--lr", "1e-3", "--batch-size", "2048", "--patience", "5"]
BASE_MLP = ["--model", "mlp"]

RUNS: list[tuple[str, str, list[str]]] = []     # (grupo, nome, args)
for seed in (1, 2, 3, 4, 5):
    RUNS.append(("seeds", f"lstm_seed{seed}", BASE_LSTM + ["--seed", str(seed)]))
    RUNS.append(("seeds", f"mlp_seed{seed}", BASE_MLP + ["--seed", str(seed)]))


def _merge(base: list[str], extra: list[str]) -> list[str]:
    """Sobrescreve opções do base com as de extra (mesma chave)."""
    out = list(base)
    for i in range(0, len(extra), 2):
        k, v = extra[i], extra[i + 1]
        if k in out:
            j = out.index(k)
            out[j + 1] = v
        else:
            out += [k, v]
    return out


for name, extra in [
    ("lstm_h32", ["--hidden", "32"]), ("lstm_h128", ["--hidden", "128"]), ("lstm_h256", ["--hidden", "256"]),
    ("lstm_2layers", ["--layers", "2"]), ("lstm_2layers_h128", ["--hidden", "128", "--layers", "2"]),
    ("lstm_drop0", ["--dropout", "0.0"]), ("lstm_drop0.4", ["--dropout", "0.4"]),
    ("lstm_lr3e-4", ["--lr", "3e-4"]), ("lstm_lr3e-3", ["--lr", "3e-3"]),
    ("lstm_batch512", ["--batch-size", "512"]), ("lstm_patience10_epochs60", ["--patience", "10", "--epochs", "60"]),
]:
    RUNS.append(("sweep", name, _merge(BASE_LSTM, extra)))
for name, extra in [
    ("lstm_seq104", ["--seq-len", "104"]), ("lstm_seq26", ["--seq-len", "26"]),
    ("lstm_posweight3", ["--pos-weight", "3"]), ("lstm_posweight5.9", ["--pos-weight", "5.9"]),
    ("mlpseq_seq104", ["--model", "mlp_seq", "--seq-len", "104"]),
]:
    RUNS.append(("targeted", name, _merge(BASE_LSTM, extra)))


def run_one(profile: str, group: str, name: str, args: list[str], dry: bool) -> dict | None:
    outdir = ROOT / "results" / profile / "sweep" / name
    mdir = ROOT / "models" / profile / "sweep" / name
    model = args[args.index("--model") + 1]
    cmd = [PY, str(ROOT / "src" / "train_nn.py"), "--profile", profile, "--outdir", str(outdir), "--models-dir", str(mdir)] + args
    if dry:
        print(" ".join(cmd))
        return None
    f = outdir / f"metrics_{model}.json"
    if f.exists():
        print(f"[{group}] {name}: já existe, pulando")
    else:
        print(f"[{group}] {name}: " + " ".join(args), flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:])
            return {"grupo": group, "nome": name, "erro": True}
        print(f"    {time.time() - t0:.0f}s | " + [l for l in r.stdout.splitlines() if "teste:" in l][-1][:120], flush=True)
    m = json.loads(f.read_text(encoding="utf-8"))
    t, v = m["test"], m["val"]
    return {"grupo": group, "nome": name, "modelo": model, "args": " ".join(a for a in args if a not in ("--model", model)),
            "val_pr_auc": v["pr_auc"], "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"], "brier": t["brier"],
            "precision_top5": t["precision_top5"], "epocas": m.get("epochs_run"), "melhor_epoca": m.get("best_epoch"),
            "segundos": m.get("total_seconds"), "params": m.get("n_params")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--only", choices=["seeds", "sweep", "targeted"], default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="teste rápido da mecânica: subamostra o treino e roda 1 época (resultados não valem)")
    ap.add_argument("--limit", type=int, default=None, help="executa só as N primeiras execuções")
    args = ap.parse_args()
    rows = []
    runs = [r for r in RUNS if not args.only or r[0] == args.only][: args.limit]
    for group, name, a in runs:
        if args.quick:
            a = a + ["--max-train", "150000", "--max-val", "30000", "--epochs", "1"]
        r = run_one(args.profile, group, name, a, args.dry_run)
        if r:
            rows.append(r)
    if args.dry_run or not rows:
        return
    df = pd.DataFrame(rows)
    df = df[~df.get("erro", pd.Series(False, index=df.index)).fillna(False).astype(bool)] if "erro" in df else df
    outdir = ROOT / "results" / args.profile
    if args.quick:
        print("\n(modo --quick: números não valem; só testa a mecânica)")
    df.to_csv(outdir / "sweep.csv", index=False)

    md = [f"# Varredura de hiperparâmetros e sementes (perfil {args.profile}, teste)\n"]
    seeds = df[df["grupo"] == "seeds"]
    if len(seeds):
        md += ["## Variabilidade entre sementes (mesma configuração, 5 sementes)\n", "| Modelo | ROC-AUC | PR-AUC | Brier | Precisão top 5 % |", "|---|---|---|---|---|"]
        for model, d in seeds.groupby("modelo"):
            cell = lambda c: f"{d[c].mean():.4f} ± {d[c].std():.4f}"
            md.append(f"| {model} (n = {len(d)}) | {cell('roc_auc')} | {cell('pr_auc')} | {cell('brier')} | {cell('precision_top5')} |")
        if {"lstm", "mlp"} <= set(seeds["modelo"]):
            a, b = seeds[seeds["modelo"] == "lstm"]["pr_auc"], seeds[seeds["modelo"] == "mlp"]["pr_auc"]
            md.append(f"\nDiferença LSTM − MLP em PR-AUC: {a.mean() - b.mean():+.4f} (desvios: {a.std():.4f} e {b.std():.4f}). "
                      "Se a diferença é menor que ~2 desvios, os modelos empatam dentro do ruído.")
    for group, title in (("sweep", "Varredura da LSTM (uma semente cada; comparar com lstm_seed1)"), ("targeted", "Experimentos dirigidos")):
        d = df[df["grupo"] == group]
        if not len(d):
            continue
        md += [f"\n## {title}\n", "| Execução | Argumentos | parâmetros | PR-AUC val. | ROC-AUC | PR-AUC | Brier | top 5 % | épocas (melhor) | s |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in d.sort_values("pr_auc", ascending=False).itertuples():
            md.append(f"| {r.nome} | `{r.args}` | {r.params:,} | {r.val_pr_auc:.4f} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | {r.brier:.4f} | {r.precision_top5:.3f} | {r.epocas} ({r.melhor_epoca}) | {r.segundos:.0f} |")
    md += ["\nLeitura: escolha de configuração deve ser feita pela PR-AUC de **validação** (a coluna do teste está aqui só para "
           "mostrar que a ordem se mantém). `pos_weight` costuma deixar a PR-AUC igual e piorar o Brier: as probabilidades saem infladas."]
    (outdir / "sweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
