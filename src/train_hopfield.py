#!/usr/bin/env python3
"""
SeismicAI - Rede de Hopfield como CAMADA DE SAÍDA: memória associativa que decide o estado de alerta.

Ideia (do professor): associar um limiar de MAGNITUDE ao alarme, para que ele só dispare quando o
padrão corresponder a terremotos de magnitude mais alta, e não a qualquer atividade.

Como funciona
  1. O estado de uma célula na semana t vira um vetor bipolar s ∈ {-1,+1}^N: cada bit é um limiar
     sobre (a) a probabilidade do modelo (LSTM/MLP), (b) a atividade recente da célula e dos vizinhos
     e (c) a MAGNITUDE máxima recente (limiar de alarme).
  2. A rede armazena protótipos de três estados, aprendidos na VALIDAÇÃO (onde existem
     probabilidades do modelo), a partir do que aconteceu na semana seguinte:
        NORMAL   nenhum evento;   ATENÇÃO   evento abaixo do limiar de alarme;   ALARME   evento M >= limiar.
     Protótipo = sinal do centroide da classe (k-means com k protótipos por classe; repetidos ficam
     com a classe mais branda). A memória de ALARME é aprendida só nas semanas de alarme COM
     ASSINATURA — a célula ou um vizinho já teve um evento M >= limiar nas últimas 12 semanas —,
     porque esse é o único tipo de alarme que tem um padrão para ser memorizado: um terremoto grande
     "do nada" não se distingue, bit a bit, de uma semana de atenção. É aqui que o limiar de
     magnitude entra: ele define o que é uma assinatura de alarme.
  3. Pesos pela regra de projeção (pseudo-inversa), W = Ξ (ΞᵀΞ)⁻¹ Ξᵀ — separa padrões correlacionados
     melhor que a regra de Hebb (--rule hebb, com diagonal zero). Com a projeção a diagonal é mantida,
     o que garante que cada protótipo é um ponto fixo exato da dinâmica.
  4. Dinâmica assíncrona: s_i <- sign(Σ_j W_ij s_j), neurônio a neurônio em ordem aleatória, até
     convergir. O atrator alcançado é a saída da rede (estado de alerta da célula). Se a dinâmica
     para em um estado espúrio (não é nenhum protótipo), vale o protótipo mais próximo, mas o ALARME
     só é emitido quando a rede converge exatamente para um protótipo de alarme (--loose-alarm
     desliga essa exigência).

Avaliação no TESTE: matriz de confusão, precision/recall/F1 por estado, taxa de estados espúrios, e a
comparação do ALARME com uma regra simples (limiar na probabilidade do modelo escolhido na validação),
no teste inteiro e no subconjunto de alarmes com assinatura.

Uso:
    python src/train_hopfield.py --model lstm                 # perfil pulse, alarme = M >= 5,0
    python src/train_hopfield.py --model mlp --alarm-mag 5.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support

from seismic_data import load_panel, load_tabular, paths, save_json

STATES = ["NORMAL", "ATENÇÃO", "ALARME"]


def bit_spec(alarm: float) -> list[tuple[str, object, str]]:
    """(feature, limiar, operador). Limiar 'qX' = quantil X na validação. 'p' = probabilidade do modelo."""
    return [
        ("p", 0.15, "ge"), ("p", 0.30, "ge"), ("p", 0.50, "ge"), ("p", 0.70, "ge"),
        ("n_1w", 1, "ge"), ("n_4w", 2, "ge"), ("n_4w", 5, "ge"), ("n_12w", 10, "ge"), ("n_52w", 25, "ge"),
        ("maxmag_1w", alarm - 0.5, "ge"), ("maxmag_4w", alarm, "ge"), ("maxmag_12w", alarm, "ge"),
        ("maxmag_52w", alarm + 0.5, "ge"), ("maxmag_52w", alarm + 1.0, "ge"),
        ("nbig_4w", 1, "ge"), ("nbig_52w", 2, "ge"), ("wsl_big", 2, "le"), ("wsl_big", 8, "le"),
        ("nviz_1w", 1, "ge"), ("nviz_4w", 5, "ge"), ("nvizbig_4w", 1, "ge"),
        ("rate_target_train", "q0.5", "ge"), ("rate_target_train", "q0.8", "ge"),
        ("wktarget_52w", 13, "ge"), ("wktarget_52w", 26, "ge"), ("logE_4w", "q0.9", "ge"),
    ]


def resolve_thresholds(spec, ref: pd.DataFrame) -> list[dict]:
    out = []
    for feat, thr, op in spec:
        if isinstance(thr, str) and thr.startswith("q"):
            thr = float(ref[feat].quantile(float(thr[1:])))
        out.append({"feature": feat, "threshold": float(thr), "op": op,
                    "name": f"{feat} {'≥' if op == 'ge' else '≤'} {float(thr):g}"})
    return out


def to_bipolar(df: pd.DataFrame, bits: list[dict]) -> np.ndarray:
    cols = []
    for b in bits:
        v = df[b["feature"]].to_numpy(dtype=np.float64)
        cols.append(v >= b["threshold"] if b["op"] == "ge" else v <= b["threshold"])
    return np.where(np.column_stack(cols), 1, -1).astype(np.int8)


class Hopfield:
    def __init__(self, prototypes: np.ndarray, rule: str = "projection"):
        self.P = prototypes.astype(np.float64)                  # [m, N]
        n = self.P.shape[1]
        if rule == "projection":                                # projetor no espaço dos protótipos: W ξ = ξ
            self.W = self.P.T @ np.linalg.pinv(self.P @ self.P.T) @ self.P
        else:                                                   # Hebb, sem autoconexões
            self.W = self.P.T @ self.P / n
            np.fill_diagonal(self.W, 0.0)
        self.rule = rule

    def energy(self, S: np.ndarray) -> np.ndarray:
        return -0.5 * np.einsum("bi,ij,bj->b", S, self.W, S)

    def recall(self, S0: np.ndarray, max_sweeps: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Atualização assíncrona (um neurônio por vez, mesma ordem para todas as amostras do lote)."""
        rng = np.random.default_rng(seed)
        S = S0.astype(np.float64).copy()
        sweeps = np.zeros(len(S), dtype=np.int16)
        for sweep in range(1, max_sweeps + 1):
            changed = np.zeros(len(S), dtype=bool)
            for i in rng.permutation(S.shape[1]):
                h = S @ self.W[:, i]
                new = np.where(h > 0, 1.0, np.where(h < 0, -1.0, S[:, i]))
                changed |= new != S[:, i]
                S[:, i] = new
            sweeps[changed] = sweep
            if not changed.any():
                break
        return S.astype(np.int8), sweeps

    def nearest(self, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Índice do protótipo mais próximo (distância de Hamming) e se o estado é espúrio."""
        dist = (S[:, None, :] != self.P[None, :, :]).sum(axis=2)
        idx = dist.argmin(axis=1)
        return idx, dist.min(axis=1) > 0


def classify(net: "Hopfield", proto_cls: np.ndarray, S: np.ndarray, loose_alarm: bool = False, seed: int = 0):
    """Relaxa os estados S até o atrator e devolve (classe prevista, estado espúrio?, varreduras, estado final).
    Sem loose_alarm, ALARME só é emitido por convergência exata a um protótipo de alarme."""
    S_final, sweeps = net.recall(S, seed=seed)
    idx, spurious = net.nearest(S_final)
    pred = proto_cls[idx].copy()
    if not loose_alarm:
        demote = spurious & (pred == 2)
        if demote.any():                                         # protótipo não-alarme mais próximo
            dist = (S_final[demote][:, None, :] != net.P[None, :, :]).sum(axis=2).astype(float)
            dist[:, proto_cls == 2] = np.inf
            pred[demote] = proto_cls[dist.argmin(axis=1)]
    return pred, spurious, sweeps, S_final


def class_of(next_maxmag: np.ndarray, alarm: float) -> np.ndarray:
    return np.where(next_maxmag <= 0, 0, np.where(next_maxmag >= alarm, 2, 1))


def learn_prototypes(S: np.ndarray, cls: np.ndarray, signature: np.ndarray, k: int, seed: int):
    """k protótipos por classe (sinal do centroide do k-means); repetidos ficam com a classe mais branda.
    A classe ALARME usa só as semanas com assinatura (evento >= limiar recente na célula ou nos vizinhos)."""
    protos, labels = [], []
    for c in range(3):
        Sc = S[(cls == c) & (signature if c == 2 else True)].astype(np.float64)
        if len(Sc) == 0:
            continue
        kk = min(k, len(Sc))
        km = KMeans(n_clusters=kk, n_init=5, random_state=seed).fit(Sc)
        for center in km.cluster_centers_:
            p = np.where(center >= 0, 1, -1).astype(np.int8)
            if not any(np.array_equal(p, q) for q in protos):
                protos.append(p)
                labels.append(c)
    return np.array(protos), np.array(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--model", default="lstm", help="modelo cujas probabilidades entram no estado (lstm, mlp, ...)")
    ap.add_argument("--alarm-mag", type=float, default=None, help="limiar de magnitude do alarme (padrão: big_mag do perfil)")
    ap.add_argument("--k", type=int, default=1, help="protótipos por estado")
    ap.add_argument("--rule", choices=["projection", "hebb"], default="projection")
    ap.add_argument("--loose-alarm", action="store_true", help="emite ALARME também a partir de estados espúrios")
    ap.add_argument("--results", default=None, help="padrão: results/{perfil}")
    ap.add_argument("--models-dir", default=None, help="padrão: models/{perfil}")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = Path(args.results or f"results/{args.profile}")
    mdir = Path(args.models_dir or f"models/{args.profile}")
    mdir.mkdir(parents=True, exist_ok=True)
    files = paths(args.profile, args.processed)
    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    alarm = args.alarm_mag or float(summary["big_mag"])

    # ---------------- dados: features + probabilidade do modelo (validação e teste) + magnitude da semana seguinte
    ds, _, _ = load_tabular(files["dataset"])
    P, cells, seq_cols, _ = load_panel(files["panel"])
    mm = seq_cols.index("max_mag")
    parts = {}
    for split, fname in (("val", f"preds_{args.model}_val.parquet"), ("test", f"preds_{args.model}.parquet")):
        pr = pd.read_parquet(results / fname)
        part = ds[ds["split"] == split].reset_index(drop=True)
        assert len(pr) == len(part) and (pr["cell"].to_numpy() == part["cell"].to_numpy()).all(), fname
        part["p"] = pr["p"].to_numpy()
        rows = np.searchsorted(cells, part["cell"].to_numpy())
        part["next_maxmag"] = P[rows, part["week"].to_numpy() + 1, mm]
        parts[split] = part
    del ds

    bits = resolve_thresholds(bit_spec(alarm), parts["val"])
    S = {s: to_bipolar(parts[s], bits) for s in parts}
    cls = {s: class_of(parts[s]["next_maxmag"].to_numpy(), alarm) for s in parts}
    signature = {s: ((parts[s]["maxmag_12w"] >= alarm) | (parts[s]["nvizbig_4w"] >= 1)).to_numpy() for s in parts}
    print(f"perfil {args.profile} | modelo {args.model} | alarme: M ≥ {alarm:g} | {len(bits)} bits | "
          f"classes na validação: " + ", ".join(f"{STATES[c]} {np.mean(cls['val'] == c):.1%}" for c in range(3))
          + f" | alarmes com assinatura: {signature['val'][cls['val'] == 2].mean():.0%}")

    # ---------------- memória: protótipos (validação) e pesos
    protos, proto_cls = learn_prototypes(S["val"], cls["val"], signature["val"], args.k, args.seed)
    net = Hopfield(protos, args.rule)
    fixed = [bool((net.recall(protos[i:i + 1], seed=args.seed)[0][0] == protos[i]).all()) for i in range(len(protos))]
    print(f"protótipos armazenados: {len(protos)} (" + ", ".join(f"{STATES[c]}×{int((proto_cls == c).sum())}" for c in range(3))
          + f") | pontos fixos: {sum(fixed)}/{len(fixed)}")

    # ---------------- recordação
    out = {}
    for split in ("val", "test"):
        pred, spurious, sweeps, S_final = classify(net, proto_cls, S[split], args.loose_alarm, args.seed)
        out[split] = {"pred": pred, "spurious": spurious, "sweeps": sweeps,
                      "energy_in": net.energy(S[split].astype(np.float64)), "energy_out": net.energy(S_final.astype(np.float64))}
    pred, true = out["test"]["pred"], cls["test"]
    cm = pd.crosstab(pd.Series([STATES[c] for c in true], name="real"), pd.Series([STATES[c] for c in pred], name="Hopfield"))
    cm = cm.reindex(index=STATES, columns=STATES, fill_value=0)
    pr, rc, f1, sup = precision_recall_fscore_support(true, pred, labels=[0, 1, 2], zero_division=0)

    # alarme (saída binária) vs. regra simples: p >= tau, com tau escolhido na validação para a classe ALARME
    y_alarm = {s: (cls[s] == 2).astype(int) for s in cls}
    taus = np.quantile(parts["val"]["p"], np.linspace(0.5, 0.999, 200))
    f1s = [precision_recall_fscore_support(y_alarm["val"], (parts["val"]["p"] >= t).astype(int), average="binary", zero_division=0)[2] for t in taus]
    tau = float(taus[int(np.argmax(f1s))])
    rule = (parts["test"]["p"] >= tau).astype(int)
    hop_alarm = (pred == 2).astype(int)
    # regra com a MESMA fração de alarmes da Hopfield (limiar fixado na validação): comparação no mesmo ponto de operação
    hop_rate_val = float((out["val"]["pred"] == 2).mean())
    tau_m = float(np.quantile(parts["val"]["p"], 1.0 - hop_rate_val)) if hop_rate_val > 0 else 1.0
    rule_m = (parts["test"]["p"] >= tau_m).astype(int)
    a_h = precision_recall_fscore_support(y_alarm["test"], hop_alarm, average="binary", zero_division=0)
    a_r = precision_recall_fscore_support(y_alarm["test"], rule, average="binary", zero_division=0)
    a_m = precision_recall_fscore_support(y_alarm["test"], rule_m, average="binary", zero_division=0)
    sg = signature["test"]
    a_hs = precision_recall_fscore_support(y_alarm["test"][sg], hop_alarm[sg], average="binary", zero_division=0)
    a_rs = precision_recall_fscore_support(y_alarm["test"][sg], rule[sg], average="binary", zero_division=0)
    a_ms = precision_recall_fscore_support(y_alarm["test"][sg], rule_m[sg], average="binary", zero_division=0)

    res = {
        "profile": args.profile, "model": args.model, "alarm_mag": alarm, "rule": args.rule, "k": args.k,
        "n_bits": len(bits), "bits": bits, "n_prototypes": int(len(protos)),
        "prototypes": [{"state": STATES[c], "pattern": p.tolist()} for p, c in zip(protos, proto_cls)],
        "class_rate_test": {STATES[c]: float(np.mean(true == c)) for c in range(3)},
        "confusion_test": cm.to_dict(),
        "per_state_test": {STATES[c]: {"precision": float(pr[c]), "recall": float(rc[c]), "f1": float(f1[c]), "support": int(sup[c])} for c in range(3)},
        "accuracy_test": float((pred == true).mean()),
        "spurious_rate_test": float(out["test"]["spurious"].mean()),
        "mean_sweeps_test": float(out["test"]["sweeps"].mean()),
        "prototypes_fixed_points": fixed,
        "signature_share_of_alarms_test": float(sg[y_alarm["test"] == 1].mean()),
        "alarm_test": {"hopfield": {"precision": float(a_h[0]), "recall": float(a_h[1]), "f1": float(a_h[2]), "alarm_rate": float(hop_alarm.mean())},
                       f"regra p>={tau:.3f} (máx. F1 na validação)": {"precision": float(a_r[0]), "recall": float(a_r[1]), "f1": float(a_r[2]), "alarm_rate": float(rule.mean())},
                       f"regra p>={tau_m:.3f} (mesma fração de alarmes)": {"precision": float(a_m[0]), "recall": float(a_m[1]), "f1": float(a_m[2]), "alarm_rate": float(rule_m.mean())},
                       "base_rate": float(y_alarm["test"].mean())},
        "alarm_test_with_signature": {"n": int(sg.sum()), "base_rate": float(y_alarm["test"][sg].mean()),
                                      "hopfield": {"precision": float(a_hs[0]), "recall": float(a_hs[1]), "f1": float(a_hs[2])},
                                      f"regra p>={tau:.3f} (máx. F1 na validação)": {"precision": float(a_rs[0]), "recall": float(a_rs[1]), "f1": float(a_rs[2])},
                                      f"regra p>={tau_m:.3f} (mesma fração de alarmes)": {"precision": float(a_ms[0]), "recall": float(a_ms[1]), "f1": float(a_ms[2])}},
    }
    save_json(res, results / "hopfield_results.json")
    save_json({"profile": args.profile, "model": args.model, "alarm_mag": alarm, "states": STATES, "bits": bits,
               "prototypes": protos.tolist(), "prototype_states": proto_cls.tolist(), "W": net.W.tolist(), "rule": args.rule},
              mdir / "hopfield.json")
    preds = parts["test"][["cell", "week", "label_week_start", "y", "p", "next_maxmag"]].copy()
    preds["true_state"] = [STATES[c] for c in true]
    preds["hopfield_state"] = [STATES[c] for c in pred]
    preds["spurious"] = out["test"]["spurious"]
    preds.to_parquet(results / "preds_hopfield.parquet", index=False)

    # ---------------- tabela e figuras
    md = [f"### Rede de Hopfield — saída de alerta (teste, alarme = M ≥ {alarm:g}, modelo {args.model})\n",
          f"Protótipos: {len(protos)} ({', '.join(f'{STATES[c]}×{int((proto_cls == c).sum())}' for c in range(3))}); "
          f"regra {args.rule}; {len(bits)} bits; acurácia {res['accuracy_test']:.3f}; estados espúrios {res['spurious_rate_test']:.1%}; "
          f"varreduras até convergir (média) {res['mean_sweeps_test']:.1f}.\n",
          "\n**Matriz de confusão (linhas = real, colunas = Hopfield)**\n", "| real \\ Hopfield | " + " | ".join(STATES) + " |", "|---|---|---|---|"]
    md += ["| " + s + " | " + " | ".join(f"{int(cm.loc[s, t]):,}" for t in STATES) + " |" for s in STATES]
    md += ["\n**Por estado**\n", "| Estado | Taxa real | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    md += [f"| {STATES[c]} | {res['class_rate_test'][STATES[c]]:.3f} | {pr[c]:.3f} | {rc[c]:.3f} | {f1[c]:.3f} |" for c in range(3)]
    md += [f"\n**Alarme (M ≥ {alarm:g} na semana seguinte; taxa base {res['alarm_test']['base_rate']:.3f})**\n",
           "| Decisor | Precision | Recall | F1 | Fração de alarmes |", "|---|---|---|---|---|"]
    for name, v in res["alarm_test"].items():
        if isinstance(v, dict):
            md.append(f"| {name} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['alarm_rate']:.3f} |")
    aw = res["alarm_test_with_signature"]
    md += [f"\n**Só semanas com assinatura** (evento M ≥ {alarm:g} recente na célula ou nos vizinhos; n = {aw['n']:,}; "
           f"{res['signature_share_of_alarms_test']:.0%} dos alarmes reais estão aqui; taxa base {aw['base_rate']:.3f})\n",
           "| Decisor | Precision | Recall | F1 |", "|---|---|---|---|"]
    for name, v in aw.items():
        if isinstance(v, dict):
            md.append(f"| {name} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    md += ["\n**Protótipos (bits ativos, +1)**\n"]
    md += [f"- {STATES[c]} {i + 1}: " + ", ".join(b["name"] for b, v in zip(bits, p) if v > 0) for i, (p, c) in enumerate(zip(protos, proto_cls))]
    (results / "hopfield_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(10, 0.35 * len(protos) + 1.6))
    ax.imshow(protos, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(protos)), [f"{STATES[c]} {i + 1}" for i, c in enumerate(proto_cls)], fontsize=8)
    ax.set_xticks(range(len(bits)), [b["name"] for b in bits], rotation=90, fontsize=6)
    ax.set_title("Hopfield — protótipos armazenados (vermelho = +1, azul = −1)")
    fig.savefig(results / "fig_hopfield_prototypes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    M = cm.to_numpy()
    ax.imshow(M / M.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:,}\n({M[i, j] / M[i].sum():.0%})", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] / M[i].sum() > 0.6 else "black")
    ax.set_xticks(range(3), STATES)
    ax.set_yticks(range(3), STATES)
    ax.set(xlabel="estado dado pela rede de Hopfield", ylabel="estado real (semana seguinte)",
           title=f"Hopfield — teste (alarme = M ≥ {alarm:g})")
    fig.savefig(results / "fig_hopfield_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nresultados em {results / 'hopfield_results.json'} | memória em {mdir / 'hopfield.json'}")


if __name__ == "__main__":
    main()
