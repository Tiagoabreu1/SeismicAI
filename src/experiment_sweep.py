#!/usr/bin/env python3
"""
SeismicAI - varredura de hiperparâmetros, sementes e experimentos dirigidos.

Responde a "treinar mais, ou de outro jeito, melhora?" com números:
  1. sementes  : a mesma configuração com 5 sementes (LSTM e MLP), para ter média e desvio;
  2. varredura : mais e menos neurônios, mais camadas, dropout, taxa de aprendizado, lote e paciência;
  3. dirigidos : sequência de 104 e de 26 semanas, perda ponderada (pos_weight) e MLP-seq com 104 semanas.

Cada execução é um train_nn.py com pasta própria em results/{perfil}/sweep/<nome>/. No fim, tudo o que
existe em disco é consolidado em results/{perfil}/sweep.md e sweep.csv.
Pode ser interrompido e retomado: execuções concluídas são puladas.

Uso:
    python src/experiment_sweep.py --quick --limit 2    # teste da mecânica: 2 execuções curtas, em pasta separada
    python src/experiment_sweep.py                      # varredura completa
    python src/experiment_sweep.py --only seeds         # ou: sweep, targeted
    python src/experiment_sweep.py --runs lstm_h128,lstm_h256
    python src/experiment_sweep.py --dry-run            # só lista os comandos
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# O console do Windows não imprime alguns símbolos quando a saída é redirecionada; força UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Configuração base de cada modelo: a mesma usada no treino oficial
BASE_LSTM = ["--model", "lstm", "--hidden", "64", "--layers", "1", "--dropout", "0.2", "--lr", "1e-3",
             "--batch-size", "2048", "--patience", "5"]
BASE_MLP = ["--model", "mlp", "--mlp-hidden", "128,64,32"]


def _merge(base: list[str], extra: list[str]) -> list[str]:
    """Sobrescreve opções do base com as de extra (mesma chave) e acrescenta as novas."""
    out = list(base)
    for i in range(0, len(extra), 2):
        k, v = extra[i], extra[i + 1]
        if k in out:
            out[out.index(k) + 1] = v
        else:
            out += [k, v]
    return out


# Lista de execuções: (grupo, nome, argumentos do train_nn.py)
RUNS: list[tuple[str, str, list[str]]] = []
for seed in (1, 2, 3, 4, 5):
    RUNS.append(("seeds", f"lstm_seed{seed}", BASE_LSTM + ["--seed", str(seed)]))
    RUNS.append(("seeds", f"mlp_seed{seed}", BASE_MLP + ["--seed", str(seed)]))
for name, extra in [
    ("lstm_h32", ["--hidden", "32"]), ("lstm_h128", ["--hidden", "128"]), ("lstm_h256", ["--hidden", "256"]),
    ("lstm_2layers", ["--layers", "2"]), ("lstm_2layers_h128", ["--hidden", "128", "--layers", "2"]),
    ("lstm_drop0", ["--dropout", "0.0"]), ("lstm_drop0.4", ["--dropout", "0.4"]),
    ("lstm_lr3e-4", ["--lr", "3e-4"]), ("lstm_lr3e-3", ["--lr", "3e-3"]),
    ("lstm_batch512", ["--batch-size", "512"]), ("lstm_patience10_epochs60", ["--patience", "10", "--epochs", "60"]),
]:
    RUNS.append(("sweep", name, _merge(BASE_LSTM, extra)))
for name, hidden in [("mlp_64-32", "64,32"), ("mlp_256-128-64", "256,128,64"), ("mlp_512-256-128", "512,256,128")]:
    RUNS.append(("sweep", name, _merge(BASE_MLP, ["--mlp-hidden", hidden])))
for name, extra in [
    ("lstm_seq104", ["--seq-len", "104"]), ("lstm_seq26", ["--seq-len", "26"]),
    ("lstm_posweight3", ["--pos-weight", "3"]), ("lstm_posweight5.9", ["--pos-weight", "5.9"]),
    ("mlpseq_seq104", ["--model", "mlp_seq", "--seq-len", "104"]),
]:
    RUNS.append(("targeted", name, _merge(BASE_LSTM, extra)))

QUICK_ARGS = ["--max-train", "150000", "--max-val", "30000", "--epochs", "1"]


def model_of(args: list[str]) -> str:
    return args[args.index("--model") + 1]


def changed_args(args: list[str]) -> str:
    """Opções que diferem da configuração base do mesmo tipo de modelo, no formato '--opcao valor'."""
    base = BASE_MLP if model_of(args) == "mlp" else BASE_LSTM
    b = {base[i]: base[i + 1] for i in range(0, len(base), 2)}
    diff = []
    for i in range(0, len(args), 2):
        if b.get(args[i]) != args[i + 1]:
            diff += [args[i], args[i + 1]]
    return " ".join(diff) or "(configuração base)"


def load_valid(f: Path, quick: bool) -> dict | None:
    """Lê as métricas de uma execução. Na varredura de verdade, rejeita resultados de teste rápido
    (treino subamostrado), para que eles nunca se misturem à tabela."""
    try:
        m = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    a = m.get("args", {})
    if not quick and (a.get("max_train") or a.get("max_val")):
        return None
    return m if "test" in m and "val" in m else None


def keep_awake(on: bool) -> None:
    """No Windows, pede ao sistema para não suspender enquanto a varredura roda."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))
    except Exception:
        pass


def run_one(profile: str, sub: str, group: str, name: str, args: list[str], quick: bool, tag: str) -> str:
    """Executa um treino, se ainda não houver resultado válido. Devolve 'feito', 'pulado' ou 'falhou'."""
    outdir = ROOT / "results" / profile / sub / name
    mdir = ROOT / "models" / profile / sub / name
    f = outdir / f"metrics_{model_of(args)}.json"
    if f.exists():
        if load_valid(f, quick) is not None:
            print(f"{tag} {name}: já concluído, pulando", flush=True)
            return "pulado"
        print(f"{tag} {name}: resultado antigo de teste rápido encontrado; refazendo", flush=True)
        shutil.rmtree(outdir, ignore_errors=True)
    cmd = [PY, str(ROOT / "src" / "train_nn.py"), "--profile", profile, "--outdir", str(outdir),
           "--models-dir", str(mdir)] + args
    print(f"{tag} {name}: " + " ".join(args[2:]), flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, env=env)
    minutes = (time.time() - t0) / 60
    # O critério de sucesso é o resultado gravado nesta execução, não o código de saída: no Windows, treinos com
    # GPU podem terminar com código diferente de zero ao encerrar o processo, depois de tudo já estar salvo.
    fresh = f.exists() and f.stat().st_mtime >= t0 - 2
    m = load_valid(f, quick) if fresh else None
    if m is None:
        print(f"    FALHOU após {minutes:.1f} min (código de saída {r.returncode}). Fim da saída do treino:", flush=True)
        print("    " + "\n    ".join((r.stdout[-800:] + "\n" + r.stderr[-1500:]).strip().splitlines()[-18:]), flush=True)
        return "falhou"
    t = m["test"]
    print(f"    {minutes:.1f} min, {m.get('device', '?')}, {m.get('epochs_run', '?')} épocas | "
          f"validação PR-AUC {m['val']['pr_auc']:.4f} | teste ROC-AUC {t['roc_auc']:.4f} PR-AUC {t['pr_auc']:.4f}", flush=True)
    if r.returncode != 0:
        saved = (mdir / f"{model_of(args)}.pt").exists()
        print(f"    aviso: o processo terminou com código {r.returncode} depois de gravar os resultados"
              + ("; métricas e modelo estão salvos e valem." if saved else "; as métricas valem, mas o arquivo do modelo não foi salvo."), flush=True)
    return "feito"


def collect(profile: str, sub: str, quick: bool) -> pd.DataFrame:
    """Reúne tudo o que já existe em disco para a lista de execuções (não só o que rodou agora)."""
    rows = []
    for group, name, args in RUNS:
        model = model_of(args)
        m = load_valid(ROOT / "results" / profile / sub / name / f"metrics_{model}.json", quick)
        if m is None:
            continue
        t, v = m["test"], m["val"]
        rows.append({"grupo": group, "nome": name, "modelo": model, "args": " ".join(args[2:]), "alterado": changed_args(args),
                     "val_pr_auc": v["pr_auc"], "val_roc_auc": v["roc_auc"], "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"],
                     "brier": t["brier"], "precision_top5": t["precision_top5"], "epocas": int(m.get("epochs_run", 0)),
                     "melhor_epoca": int(m.get("best_epoch", 0)), "segundos": float(m.get("total_seconds", 0)),
                     "params": int(m.get("n_params", 0)), "device": m.get("device", "")})
    return pd.DataFrame(rows)


def noise_factor(n: int) -> float:
    """Meia-largura, em desvios amostrais, do intervalo de 95 % para UMA nova execução da mesma configuração,
    quando o desvio foi estimado com n sementes: t(0,975; n-1) * raiz(1 + 1/n). Com 5 sementes, cerca de 3."""
    if n < 2:
        return float("nan")
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, n - 1) * (1 + 1 / n) ** 0.5)
    except ImportError:
        return {2: 15.56, 3: 4.97, 4: 3.56, 5: 3.04, 6: 2.78}.get(n, 2.5)


def report(df: pd.DataFrame, profile: str, quick: bool) -> str:
    """Monta a tabela consolidada em Markdown."""
    md = [f"# Varredura de hiperparâmetros e sementes (perfil {profile}, conjunto de teste)\n"]
    if quick:
        md.append("**Teste rápido: treino subamostrado e 1 época. Os números abaixo não valem; servem só para conferir a mecânica.**\n")
    ref: dict[str, dict] = {}
    seeds = df[df["grupo"] == "seeds"]
    if len(seeds):
        md += ["## Variabilidade entre sementes (mesma configuração, sementes diferentes)\n",
               "| Modelo | n | PR-AUC validação | ROC-AUC | PR-AUC | Brier | Precisão top 5 % |", "|---|---|---|---|---|---|---|"]
        for model, d in seeds.groupby("modelo"):
            def cell(c: str, nd: int = 4) -> str:
                return f"{d[c].mean():.{nd}f} ± {d[c].std():.{nd}f}" if len(d) > 1 else f"{d[c].mean():.{nd}f}"
            k = noise_factor(len(d))
            ref[model] = {"n": len(d), "k": k,
                          "val": (float(d["val_pr_auc"].mean()), float(d["val_pr_auc"].std()) if len(d) > 1 else float("nan")),
                          "brier": (float(d["brier"].mean()), float(d["brier"].std()) if len(d) > 1 else float("nan"))}
            md.append(f"| {model} | {len(d)} | {cell('val_pr_auc')} | {cell('roc_auc')} | {cell('pr_auc')} | {cell('brier')} | {cell('precision_top5', 3)} |")
        if {"lstm", "mlp"} <= set(ref):
            a, b = seeds[seeds["modelo"] == "lstm"]["pr_auc"], seeds[seeds["modelo"] == "mlp"]["pr_auc"]
            if len(a) > 1 and len(b) > 1:
                separated = a.min() > b.max() or b.min() > a.max()
                try:
                    from scipy import stats
                    tstat, pval = stats.ttest_ind(a, b, equal_var=False)
                    test = f" (teste t de Welch: t = {tstat:.1f}, p = {pval:.4f})"
                    significant = pval < 0.05
                except ImportError:
                    test, significant = "", separated
                better = "LSTM" if a.mean() > b.mean() else "MLP"
                md.append(f"\nLSTM contra MLP na PR-AUC de teste: diferença de {a.mean() - b.mean():+.4f}{test}. "
                          f"LSTM de {a.min():.4f} a {a.max():.4f}; MLP de {b.min():.4f} a {b.max():.4f}. "
                          + (f"A {better} é melhor de forma consistente entre sementes" + (", e as faixas nem se sobrepõem" if separated else "")
                             + ": a diferença é pequena, mas não é ruído de treino." if significant
                             else "A diferença não se distingue do ruído entre sementes."))
    outside, brier_out = [], []
    for group, title in (("sweep", "Varredura de arquitetura e treino"), ("targeted", "Experimentos dirigidos")):
        d = df[df["grupo"] == group]
        if not len(d):
            continue
        md += [f"\n## {title}\n",
               "Cada linha é uma execução com uma semente, comparada com a configuração base do mesmo tipo de modelo "
               "(média das sementes) pela PR-AUC de validação; a MLP-seq é comparada com a base da LSTM.\n",
               "| Execução | Argumentos alterados | Parâmetros | PR-AUC validação | Diferença para a base | ROC-AUC | PR-AUC | Brier | Top 5 % | Épocas (melhor) | Minutos |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in d.sort_values("val_pr_auc", ascending=False).itertuples():
            base = ref.get("lstm" if r.modelo == "mlp_seq" else r.modelo)
            comp = "sem base"
            if base and base["val"][1] == base["val"][1]:
                diff, band = r.val_pr_auc - base["val"][0], base["k"] * base["val"][1]
                verdict = "acima do ruído" if diff > band else "abaixo do ruído" if diff < -band else "dentro do ruído"
                comp = f"{diff:+.4f} ({verdict})"
                if verdict != "dentro do ruído":
                    outside.append(f"{r.nome} ({'melhor' if diff > 0 else 'pior'}, {diff:+.4f})")
                bdiff = r.brier - base["brier"][0]
                if abs(bdiff) > base["k"] * base["brier"][1]:
                    brier_out.append(f"{r.nome} ({bdiff:+.4f})")
            md.append(f"| {r.nome} | `{r.alterado}` | {int(r.params):,} | {r.val_pr_auc:.4f} | {comp} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | "
                      f"{r.brier:.4f} | {r.precision_top5:.3f} | {int(r.epocas)} ({int(r.melhor_epoca)}) | {r.segundos / 60:.1f} |")
    if ref and (df["grupo"] != "seeds").any():
        md += ["\n## Resumo\n",
               "Fora do ruído na PR-AUC de validação: " + (", ".join(outside) if outside else "nenhuma execução") + ".",
               "Brier fora do ruído (positivo significa calibração pior): " + (", ".join(brier_out) if brier_out else "nenhuma execução") + "."]
    k_txt = ", ".join(f"{m} {v['k']:.2f} desvios (n = {v['n']})" for m, v in ref.items() if v["k"] == v["k"])
    md += ["\nComo ler: a escolha de configuração se faz pela PR-AUC de validação, nunca pela de teste; as colunas de teste "
           "estão aqui para mostrar que a ordem se mantém. \"Dentro do ruído\" significa que a execução cai no intervalo de 95 % "
           "esperado para uma nova semente da configuração base" + (f" ({k_txt})" if k_txt else "") + ". "
           "Em \"Épocas (melhor)\", o segundo número é a época de maior PR-AUC de validação. "
           "A perda ponderada (pos_weight) vem acompanhada, no train_nn.py, de uma correção do logit que devolve a calibração; "
           "por isso o esperado é que ela não mude nem a PR-AUC nem o Brier. Sem essa correção, as probabilidades sairiam infladas."]
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--only", choices=["seeds", "sweep", "targeted"], default=None, help="roda só um grupo")
    ap.add_argument("--runs", default=None, help="roda só as execuções com estes nomes, separados por vírgula")
    ap.add_argument("--limit", type=int, default=None, help="roda só as N primeiras execuções selecionadas")
    ap.add_argument("--quick", action="store_true",
                    help="teste da mecânica: treino subamostrado e 1 época, em pasta separada (os números não valem)")
    ap.add_argument("--dry-run", action="store_true", help="só lista os comandos")
    args = ap.parse_args()

    # Seleção das execuções
    runs = [r for r in RUNS if not args.only or r[0] == args.only]
    if args.runs:
        wanted = [n.strip() for n in args.runs.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in {r[1] for r in RUNS}]
        if unknown:
            raise SystemExit(f"execuções desconhecidas: {unknown}. Nomes válidos: {[r[1] for r in RUNS]}")
        runs = [r for r in runs if r[1] in wanted]
    runs = runs[: args.limit]
    sub = "sweep/_quick" if args.quick else "sweep"

    if args.dry_run:
        for group, name, a in runs:
            print(f"[{group}] {name}: python src/train_nn.py --profile {args.profile} " + " ".join(a + (QUICK_ARGS if args.quick else [])))
        print(f"{len(runs)} execuções")
        return

    # Conferências antes de começar: dados prontos e dispositivo de treino
    proc = ROOT / "data" / "processed"
    missing = [p.name for p in (proc / f"dataset_{args.profile}.parquet", proc / f"panel_{args.profile}.parquet") if not p.exists()]
    if missing:
        raise SystemExit(f"faltam os dados {missing} em data/processed. Rode antes:\n"
                         "    python src/download_usgs.py\n    python src/build_dataset.py")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU encontrada: {torch.cuda.get_device_name(0)}")
        else:
            print("ATENÇÃO: o PyTorch não encontrou GPU com CUDA; os treinos vão rodar na CPU e a varredura completa "
                  "leva várias horas. Se este computador tem placa NVIDIA, reinstale o torch com CUDA antes de continuar.")
    except ImportError:
        raise SystemExit("PyTorch não está instalado neste ambiente (pip install torch).")

    if args.quick:                                   # o teste rápido sempre recomeça do zero, em pasta própria
        shutil.rmtree(ROOT / "results" / args.profile / sub, ignore_errors=True)
        shutil.rmtree(ROOT / "models" / args.profile / sub, ignore_errors=True)

    print(f"{len(runs)} execuções selecionadas, perfil {args.profile}" + (" (teste rápido)" if args.quick else ""))
    keep_awake(True)
    t_start, counts = time.time(), {"feito": 0, "pulado": 0, "falhou": 0}
    try:
        for i, (group, name, a) in enumerate(runs, 1):
            tag = f"[{time.strftime('%H:%M')}] [{i}/{len(runs)}] [{group}]"
            counts[run_one(args.profile, sub, group, name, a + (QUICK_ARGS if args.quick else []), args.quick, tag)] += 1
    except KeyboardInterrupt:
        print("\ninterrompido pelo usuário; consolidando o que já terminou")
    finally:
        keep_awake(False)

    # Consolida tudo o que existe em disco, inclusive o que foi rodado em outra sessão
    df = collect(args.profile, sub, args.quick)
    outdir = ROOT / "results" / args.profile / (sub if args.quick else "")
    stem = "sweep_quick" if args.quick else "sweep"
    if len(df):
        df.to_csv(outdir / f"{stem}.csv", index=False)
        text = report(df, args.profile, args.quick)
        (outdir / f"{stem}.md").write_text(text, encoding="utf-8")
        print("\n" + text)
        print(f"tabela salva em {outdir / (stem + '.md')}")
    print(f"\nconcluídas agora: {counts['feito']} | já existiam: {counts['pulado']} | falharam: {counts['falhou']} | "
          f"na tabela: {len(df)} de {len(RUNS)} | tempo desta sessão: {(time.time() - t_start) / 60:.0f} min")
    if args.quick and counts["falhou"] == 0:
        print("Teste rápido concluído sem falhas. Rode agora sem --quick para a varredura de verdade.")
    if counts["falhou"]:
        print("Houve falhas. Corrija a causa e rode o mesmo comando de novo: só o que faltou será refeito.")


if __name__ == "__main__":
    main()
