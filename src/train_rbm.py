#!/usr/bin/env python3
"""
SeismicAI - Máquina de Boltzmann Restrita (RBM): aprendizado NÃO supervisionado de features.

A RBM é um modelo baseado em energia com uma camada visível (as features, escaladas para [0, 1]) e
uma camada oculta binária, sem conexões dentro de cada camada. É treinada SEM os rótulos por
divergência contrastiva (CD-k): a fase positiva "puxa" a energia das amostras reais para baixo e a
fase negativa (reconstruções de Gibbs) "empurra" a energia das reconstruções para cima.

Depois do treino, dois usos:
  1. rbm_logreg : as probabilidades das unidades ocultas h = sigmoid(Wx + c) viram as entradas de
                  uma regressão logística (pré-treino não supervisionado + classificador linear);
  2. rbm_energy : a energia livre F(x) = -b·x - sum(softplus(Wx + c)) mede quão "típica" é a
                  semana; o score é calibrado em probabilidade por regressão logística 1-D
                  (Platt) na validação. Mede o que a RBM aprende sozinha, sem rótulo.
Ambos são avaliados com as mesmas métricas dos outros modelos (results/{perfil}).

Uso:
    python src/train_rbm.py                               # perfil pulse
    python src/train_rbm.py --profile global --hidden 64 --epochs 15
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from metrics import save_preds, summarize
from seismic_data import Scaler, load_tabular, paths, save_json, split_xy


class Unit01Scaler:
    """log1p nas contagens, depois escala cada feature para [0, 1] pelos quantis 0,5 % e 99,5 % do treino."""

    def __init__(self, cols):
        self.base = Scaler(cols)

    def fit(self, X):
        Z = self.base._pre(X)
        self.lo = np.quantile(Z, 0.005, axis=0)
        self.hi = np.quantile(Z, 0.995, axis=0)
        self.hi = np.where(self.hi > self.lo, self.hi, self.lo + 1e-6)
        return self

    def transform(self, X):
        return np.clip((self.base._pre(X) - self.lo) / (self.hi - self.lo), 0.0, 1.0).astype(np.float32)

    def to_dict(self):
        return {"cols": self.base.cols, "lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_dict(cls, d):
        s = cls(d["cols"])
        s.lo, s.hi = np.array(d["lo"], np.float32), np.array(d["hi"], np.float32)
        return s


class RBM:
    """RBM Bernoulli-Bernoulli (as visíveis em [0,1] são tratadas como probabilidades)."""

    def __init__(self, n_vis: int, n_hid: int, device, seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W = (torch.randn(n_vis, n_hid, generator=g) * 0.01).to(device)
        self.b = torch.zeros(n_vis, device=device)      # viés das visíveis
        self.c = torch.zeros(n_hid, device=device)      # viés das ocultas
        self.vW, self.vb, self.vc = torch.zeros_like(self.W), torch.zeros_like(self.b), torch.zeros_like(self.c)

    def h_prob(self, v):
        return torch.sigmoid(v @ self.W + self.c)

    def v_prob(self, h):
        return torch.sigmoid(h @ self.W.t() + self.b)

    def free_energy(self, v):
        return -(v @ self.b) - torch.nn.functional.softplus(v @ self.W + self.c).sum(dim=1)

    def cd_step(self, v0, k: int, lr: float, momentum: float, weight_decay: float) -> float:
        h0 = self.h_prob(v0)                       # fase positiva
        hs = torch.bernoulli(h0)
        vk, hk = v0, h0
        for _ in range(k):                         # cadeia de Gibbs (fase negativa)
            vk = self.v_prob(hs)                   # probabilidades nas visíveis: menos ruído
            hk = self.h_prob(vk)
            hs = torch.bernoulli(hk)
        n = v0.shape[0]
        dW = (v0.t() @ h0 - vk.t() @ hk) / n - weight_decay * self.W
        db = (v0 - vk).mean(dim=0)
        dc = (h0 - hk).mean(dim=0)
        self.vW.mul_(momentum).add_(lr * dW)
        self.vb.mul_(momentum).add_(lr * db)
        self.vc.mul_(momentum).add_(lr * dc)
        self.W += self.vW
        self.b += self.vb
        self.c += self.vc
        return float(((v0 - vk) ** 2).mean())

    def state_dict(self):
        return {"W": self.W.cpu(), "b": self.b.cpu(), "c": self.c.cpu()}


@torch.no_grad()
def batched(fn, X: torch.Tensor, bs: int = 16384) -> np.ndarray:
    return torch.cat([fn(X[i:i + bs]).cpu() for i in range(0, len(X), bs)]).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--data", default=None)
    ap.add_argument("--outdir", default=None, help="padrão: results/{perfil}")
    ap.add_argument("--models-dir", default=None, help="padrão: models/{perfil}")
    ap.add_argument("--hidden", type=int, default=64, help="unidades ocultas")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=1, help="passos de Gibbs (CD-k)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=None, help="subamostra o treino não supervisionado")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    outdir = Path(args.outdir or f"results/{args.profile}")
    mdir = Path(args.models_dir or f"models/{args.profile}")
    outdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    ds, feature_cols, _ = load_tabular(args.data or paths(args.profile)["dataset"])
    X, y, ids = split_xy(ds, feature_cols)
    del ds
    scaler = Unit01Scaler(feature_cols).fit(X["train"])
    V = {s: torch.tensor(scaler.transform(X[s]), device=device) for s in X}
    print(f"perfil {args.profile} | RBM {len(feature_cols)} visíveis x {args.hidden} ocultas | device {device}")

    # ---------------- treino não supervisionado (sem rótulos)
    rbm = RBM(len(feature_cols), args.hidden, device, args.seed)
    train_idx = np.arange(len(V["train"]))
    if args.max_train and args.max_train < len(train_idx):
        train_idx = np.sort(rng.choice(train_idx, args.max_train, replace=False))
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.tensor(rng.permutation(train_idx), device=device)
        momentum = 0.5 if epoch <= 5 else 0.9
        errs = []
        for i in range(0, len(perm), args.batch_size):
            errs.append(rbm.cd_step(V["train"][perm[i:i + args.batch_size]], args.k, args.lr, momentum, args.weight_decay))
        with torch.no_grad():
            fe_train = float(rbm.free_energy(V["train"][perm[:50000]]).mean())
            fe_val = float(rbm.free_energy(V["val"]).mean())
        history.append({"epoch": epoch, "recon_error": float(np.mean(errs)), "free_energy_train": fe_train,
                        "free_energy_val": fe_val, "seconds": time.time() - t0})
        print(f"época {epoch:>2}  erro de reconstrução {np.mean(errs):.4f}  energia livre treino {fe_train:8.2f}  "
              f"val {fe_val:8.2f}  {time.time() - t0:4.0f}s", flush=True)
    pd.DataFrame(history).to_csv(outdir / "history_rbm.csv", index=False)

    # ---------------- 1) features ocultas + regressão logística (supervisionado, linear)
    H = {s: batched(rbm.h_prob, V[s]) for s in V}
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(H["train"], y["train"])
    p_val, p_test = clf.predict_proba(H["val"])[:, 1], clf.predict_proba(H["test"])[:, 1]
    res = summarize("rbm_logreg", ids["val"], p_val, ids["test"], p_test)
    res.update({"profile": args.profile, "hidden": args.hidden, "epochs": args.epochs, "k": args.k,
                "total_seconds": round(time.time() - t_start, 1), "args": vars(args)})
    save_json(res, outdir / "metrics_rbm_logreg.json")
    save_preds(ids["test"], p_test, outdir / "preds_rbm_logreg.parquet")
    save_preds(ids["val"], p_val, outdir / "preds_rbm_logreg_val.parquet")
    t = res["test"]
    print(f"\nrbm_logreg teste: ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | Brier {t['brier']:.4f} | "
          f"P {t['precision']:.3f} R {t['recall']:.3f} F1 {t['f1']:.3f} | top5%: P {t['precision_top5']:.3f} R {t['recall_top5']:.3f}")

    # ---------------- 2) energia livre como score não supervisionado (calibrado na validação)
    F = {s: batched(rbm.free_energy, V[s]).reshape(-1, 1) for s in V}
    auc_raw = roc_auc_score(y["val"], F["val"][:, 0])
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(F["val"], y["val"])          # Platt: 1 peso + 1 viés
    p_val, p_test = cal.predict_proba(F["val"])[:, 1], cal.predict_proba(F["test"])[:, 1]
    res_e = summarize("rbm_energy", ids["val"], p_val, ids["test"], p_test)
    res_e.update({"profile": args.profile, "val_roc_auc_raw_free_energy": float(auc_raw),
                  "platt": {"coef": float(cal.coef_[0][0]), "intercept": float(cal.intercept_[0])}})
    save_json(res_e, outdir / "metrics_rbm_energy.json")
    save_preds(ids["test"], p_test, outdir / "preds_rbm_energy.parquet")
    save_preds(ids["val"], p_val, outdir / "preds_rbm_energy_val.parquet")
    t = res_e["test"]
    print(f"rbm_energy teste: ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | Brier {t['brier']:.4f}  "
          f"(ROC-AUC da energia livre crua na validação: {auc_raw:.3f}; direção escolhida pela calibração)")

    # ---------------- modelo e figura dos pesos
    torch.save({"rbm": rbm.state_dict(), "scaler": scaler.to_dict(), "features": feature_cols,
                "logreg_coef": clf.coef_[0].tolist(), "logreg_intercept": float(clf.intercept_[0]),
                "platt": res_e["platt"], "profile": args.profile, "args": vars(args)}, mdir / "rbm.pt")
    W = rbm.W.cpu().numpy()
    order = np.argsort(-np.abs(W).sum(axis=1))
    fig, ax = plt.subplots(figsize=(9, 0.22 * len(feature_cols) + 1.2))
    vmax = np.abs(W).max()
    im = ax.imshow(W[order], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(feature_cols)), [feature_cols[i] for i in order], fontsize=6)
    ax.set(xlabel="unidade oculta", title=f"RBM — pesos W (features visíveis x {args.hidden} ocultas)")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.savefig(outdir / "fig_rbm_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"total {time.time() - t_start:.0f}s | modelo em {mdir / 'rbm.pt'}")


if __name__ == "__main__":
    main()
