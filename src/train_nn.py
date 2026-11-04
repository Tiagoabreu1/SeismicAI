#!/usr/bin/env python3
"""
SeismicAI - treino e avaliação das redes neurais (PyTorch; usa GPU se houver).

Modelos
  mlp     : perceptron multicamadas sobre as features tabulares (janelas agregadas por célula x semana);
  lstm    : LSTM sobre a sequência das últimas 52 semanas (variáveis semanais do painel), concatenada
            com as features estáticas da célula (posição e climatologia) antes da camada de saída;
  mlp_seq : MLP com exatamente as mesmas entradas da LSTM, achatadas (52 x variáveis + estáticas),
            a "MLP com as mesmas entradas" que o notebook da Fase 1 pede como comparação.

Procedimento
  - padronização ajustada só no treino (log1p nas contagens + z-score);
  - perda: binary cross-entropy (opcionalmente ponderada com --pos-weight);
  - otimizador Adam, early stopping pela PR-AUC de validação;
  - limiar de classificação escolhido na validação (máximo F1) e aplicado ao teste;
  - importância por permutação (queda de ROC-AUC) em uma amostra do teste.

Uso (da raiz do projeto)
  python src/train_nn.py --model mlp                    # perfil pulse (padrão)
  python src/train_nn.py --model lstm --profile global
  # validação rápida em CPU (subamostra os negativos do treino; probabilidades corrigidas):
  python src/train_nn.py --model lstm --max-train 150000 --max-val 40000 --epochs 3

Saídas (em results/{perfil} e models/{perfil})
  {model}.pt, metrics_{model}.json, preds_{model}.parquet, history_{model}.csv, importance_{model}.csv
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import save_preds, summarize
from seismic_data import SEQ_LEN, Scaler, load_panel, load_tabular, paths, save_json, split_xy


# ----------------------------------------------------------------------------- modelos
class MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64, 32), dropout: float = 0.2):
        super().__init__()
        # Camadas densas com batch norm, ReLU e dropout; a última camada tem uma saída, o logit
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, d_seq: int, d_static: int, hidden: int = 64, layers: int = 1,
                 head: int = 64, dropout: float = 0.2):
        super().__init__()
        # A LSTM lê a sequência; a cabeça junta o estado final com as variáveis fixas da célula
        self.lstm = nn.LSTM(d_seq, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + d_static, head), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(head, 1))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)                  # h: [camadas, batch, hidden]
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(-1)


class MLPSeq(nn.Module):
    """MLP sobre a sequência achatada + estáticas: mesmas entradas da LSTM, sem recorrência."""

    def __init__(self, seq_len: int, d_seq: int, d_static: int, hidden=(256, 64), dropout: float = 0.2):
        super().__init__()
        self.mlp = MLP(seq_len * d_seq + d_static, hidden, dropout)

    def forward(self, seq, static):
        return self.mlp(torch.cat([seq.flatten(1), static], dim=1))


# ----------------------------------------------------------------------------- dados
class TabularData:
    """Features tabulares já padronizadas, como tensores no device."""

    def __init__(self, X, y, ids, device):
        # Tudo vai para o device de uma vez: o dataset cabe na memória
        self.y = {s: torch.tensor(y[s], dtype=torch.float32, device=device) for s in y}
        self.X = {s: torch.tensor(X[s], device=device) for s in X}
        self.ids = ids

    def n(self, s):
        return self.X[s].shape[0]

    def batch(self, s, idx):
        return (self.X[s][idx],), self.y[s][idx]


class SequenceData:
    """Sequências montadas por lote a partir do painel P[célula, semana, var] (fica no device)."""

    def __init__(self, P, cells, static, ds_ids, y, device, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        # O painel inteiro fica no device; cada lote recorta as semanas anteriores de cada célula
        self.P = torch.tensor(P, device=device)                       # [C, W, F] padronizado
        self.S = torch.tensor(static, device=device)                  # [C, d_static] padronizado
        self.offsets = torch.arange(-seq_len + 1, 1, device=device)   # semanas t-51 .. t
        self.rows, self.weeks, self.y, self.ids = {}, {}, {}, ds_ids
        for s in ds_ids:   # linha do painel de cada célula (cells está ordenado)
            self.rows[s] = torch.tensor(np.searchsorted(cells, ds_ids[s]["cell"].to_numpy()), device=device)
            self.weeks[s] = torch.tensor(ds_ids[s]["week"].to_numpy(), device=device)
            self.y[s] = torch.tensor(y[s], dtype=torch.float32, device=device)

    def n(self, s):
        return self.rows[s].shape[0]

    def batch(self, s, idx):
        r, w = self.rows[s][idx], self.weeks[s][idx]
        seq = self.P[r[:, None], w[:, None] + self.offsets[None, :]]   # [B, L, F]
        return (seq, self.S[r]), self.y[s][idx]


def subsample_train(y: np.ndarray, max_rows: int | None, seed: int):
    """Mantém todos os positivos e sorteia negativos até max_rows. Devolve (índices, taxa de negativos mantida)."""
    n = len(y)
    if not max_rows or max_rows >= n:
        return np.arange(n), 1.0
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    k = max_rows - len(pos)
    if k <= 0:
        raise ValueError(f"--max-train ({max_rows:,}) precisa ser maior que o nº de positivos do treino ({len(pos):,})")
    keep_neg = rng.choice(neg, size=min(k, len(neg)), replace=False)
    idx = np.sort(np.concatenate([pos, keep_neg]))
    return idx, len(keep_neg) / len(neg)


# ----------------------------------------------------------------------------- treino
@torch.no_grad()
# Previsão em lotes, sem gradiente; o deslocamento do logit corrige a subamostragem de negativos
def predict(model, data, s, idx=None, batch_size=8192, logit_shift=0.0) -> np.ndarray:
    model.eval()
    idx = torch.arange(data.n(s), device=data.y[s].device) if idx is None else idx
    out = []
    for i in range(0, len(idx), batch_size):
        inputs, _ = data.batch(s, idx[i:i + batch_size])
        out.append(torch.sigmoid(model(*inputs) + logit_shift).float().cpu())
    return torch.cat(out).numpy()


def train(model, data, train_idx, val_idx, args, logit_shift, log):
    device = next(model.parameters()).device
    # Perda, otimizador e redução da taxa de aprendizado quando a validação para de melhorar
    pos_weight = torch.tensor([args.pos_weight], device=device) if args.pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    y_val = data.y["val"][val_idx].cpu().numpy()
    best, best_state, bad, history = -1.0, None, 0, []
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    # Uma época: embaralha o treino, passa os lotes, mede na validação e guarda o melhor estado
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        perm = train_idx[torch.randperm(len(train_idx), generator=g).to(train_idx.device)]
        for i in range(0, len(perm), args.batch_size):
            if len(perm) - i < 2:        # BatchNorm exige lote com mais de 1 linha
                break
            inputs, yb = data.batch("train", perm[i:i + args.batch_size])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(*inputs), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(yb)
            nb += len(yb)
        p_val = predict(model, data, "val", val_idx, logit_shift=logit_shift)
        roc, ap = roc_auc_score(y_val, p_val), average_precision_score(y_val, p_val)
        sched.step(ap)
        history.append({"epoch": epoch, "train_loss": total / nb, "val_roc_auc": roc, "val_pr_auc": ap,
                        "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        log(f"época {epoch:>2}  perda {total / nb:.4f}  val ROC-AUC {roc:.4f}  PR-AUC {ap:.4f}  "
            f"lr {opt.param_groups[0]['lr']:.0e}  {time.time() - t0:5.0f}s")
        # Early stopping pela PR-AUC de validação
        if ap > best + 1e-4:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stopping (sem melhora há {args.patience} épocas)")
                break
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def permutation_importance(model, data, names_by_input, logit_shift, n_rows=60000, seed=0):
    """Queda de ROC-AUC no teste ao embaralhar cada variável (por entrada). Usa uma amostra do teste."""
    device = data.y["test"].device
    rng = np.random.default_rng(seed)
    idx = torch.tensor(np.sort(rng.choice(data.n("test"), size=min(n_rows, data.n("test")), replace=False)),
                       device=device)
    y = data.y["test"][idx].cpu().numpy()
    inputs, _ = data.batch("test", idx)
    inputs = [t.clone() for t in inputs]

    def score(ins):
        model.eval()
        out = []
        for i in range(0, len(idx), 8192):
            out.append(torch.sigmoid(model(*[t[i:i + 8192] for t in ins]) + logit_shift).cpu())
        return roc_auc_score(y, torch.cat(out).numpy())

    # Mede a ROC-AUC com os dados intactos e depois com cada variável embaralhada entre as linhas
    base = score(inputs)
    rows = []
    perm = torch.tensor(rng.permutation(len(idx)), device=device)
    for k, names in enumerate(names_by_input):
        for j, name in enumerate(names):
            saved = inputs[k][..., j].clone()
            inputs[k][..., j] = inputs[k][perm][..., j]      # embaralha a variável entre as linhas
            rows.append({"feature": name, "roc_auc_drop": base - score(inputs)})
            inputs[k][..., j] = saved
    return pd.DataFrame(rows).sort_values("roc_auc_drop", ascending=False), base


# ----------------------------------------------------------------------------- principal
def main() -> None:
    # Opções de linha de comando
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["mlp", "lstm", "mlp_seq"], required=True)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--data", default=None, help="dataset parquet (padrão: o do perfil)")
    ap.add_argument("--panel", default=None, help="painel parquet (padrão: o do perfil)")
    ap.add_argument("--outdir", default=None, help="padrão: results/{perfil}")
    ap.add_argument("--models-dir", default=None, help="padrão: models/{perfil}")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN, help="semanas de histórico na sequência (lstm / mlp_seq)")
    ap.add_argument("--hidden", type=int, default=64, help="unidades da LSTM")
    ap.add_argument("--layers", type=int, default=1, help="camadas da LSTM")
    ap.add_argument("--mlp-hidden", default="128,64,32")
    ap.add_argument("--pos-weight", type=float, default=None, help="peso da classe positiva na perda")
    ap.add_argument("--max-train", type=int, default=None, help="subamostra o treino (mantém os positivos)")
    ap.add_argument("--max-val", type=int, default=None, help="subamostra a validação durante o treino")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    # Sementes, device e pastas de saída
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    outdir = Path(args.outdir or f"results/{args.profile}")
    mdir = Path(args.models_dir or f"models/{args.profile}")
    outdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"perfil {args.profile} | modelo {args.model} | device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    t_start = time.time()

    # Carrega o dataset tabular; sequência mais longa que 52 semanas exige linhas com histórico suficiente
    files = paths(args.profile)
    ds, feature_cols, static_cols = load_tabular(args.data or files["dataset"])
    if args.seq_len > SEQ_LEN:                 # sequência mais longa: só linhas com histórico suficiente
        ds = ds[ds["week"] >= args.seq_len - 1]
    X, y, ids = split_xy(ds, feature_cols)
    del ds

    # MLP: padroniza as features tabulares (ajuste só no treino). LSTM e MLP-seq: padronizam o painel e as variáveis fixas
    if args.model == "mlp":
        scaler = Scaler(feature_cols).fit(X["train"])
        Xs = {s: scaler.transform(X[s]) for s in X}
        data = TabularData(Xs, y, ids, device)
        model = MLP(len(feature_cols), tuple(int(h) for h in args.mlp_hidden.split(",")), args.dropout).to(device)
        names_by_input = [feature_cols]
        preprocess = {"features": scaler.to_dict()}
    else:
        P, cells, seq_cols, train_mask = load_panel(args.panel or files["panel"])
        seq_scaler = Scaler(seq_cols).fit(P[:, train_mask, :])
        Ps = seq_scaler.transform(P)
        col = {c: i for i, c in enumerate(feature_cols)}
        # features estáticas por célula (iguais em todas as semanas): pega da primeira linha de cada célula
        first = ids["train"].drop_duplicates("cell").index.to_numpy()
        stat_raw = X["train"][first][:, [col[c] for c in static_cols]]
        stat_cells = ids["train"]["cell"].to_numpy()[first]
        order = np.argsort(stat_cells)
        assert np.array_equal(stat_cells[order], cells), "células do painel e do dataset não batem"
        stat_scaler = Scaler(static_cols).fit(stat_raw)
        static = stat_scaler.transform(stat_raw[order])
        data = SequenceData(Ps, cells, static, ids, y, device, seq_len=args.seq_len)
        if args.model == "lstm":
            model = LSTMNet(len(seq_cols), len(static_cols), args.hidden, args.layers, dropout=args.dropout).to(device)
        else:
            model = MLPSeq(args.seq_len, len(seq_cols), len(static_cols), dropout=args.dropout).to(device)
        names_by_input = [[f"seq:{c}" for c in seq_cols], [f"static:{c}" for c in static_cols]]
        preprocess = {"sequence": seq_scaler.to_dict(), "static": stat_scaler.to_dict(), "seq_len": args.seq_len}
    del X

    # Contagem de parâmetros, subamostragem opcional do treino e correção do logit
    n_params = sum(p.numel() for p in model.parameters())
    log(f"parâmetros treináveis: {n_params:,}")

    train_np, neg_rate = subsample_train(y["train"], args.max_train, args.seed)
    logit_shift = math.log(neg_rate) if neg_rate < 1.0 else 0.0   # corrige a prob. após subamostrar negativos
    if args.pos_weight:
        logit_shift -= math.log(args.pos_weight)                    # idem para a perda ponderada
    train_idx = torch.tensor(train_np, device=device)
    val_np = (np.sort(np.random.default_rng(args.seed).choice(data.n("val"), args.max_val, replace=False))
              if args.max_val and args.max_val < data.n("val") else np.arange(data.n("val")))
    val_idx = torch.tensor(val_np, device=device)
    log(f"treino {len(train_np):,} linhas ({int(y['train'][train_np].sum()):,} positivos; negativos mantidos "
        f"{neg_rate:.0%}) | validação {len(val_np):,} | teste {data.n('test'):,}")

    # Treino
    history = train(model, data, train_idx, val_idx, args, logit_shift, log)

    # Previsões na validação e no teste, métricas, importância por permutação e gravação de tudo
    p_val = predict(model, data, "val", logit_shift=logit_shift)
    p_test = predict(model, data, "test", logit_shift=logit_shift)
    res = summarize(args.model, ids["val"], p_val, ids["test"], p_test)
    imp, base_auc = permutation_importance(model, data, names_by_input, logit_shift, seed=args.seed)
    res.update({
        "device": str(device), "n_params": n_params, "epochs_run": len(history),
        "best_epoch": int(max(history, key=lambda h: h["val_pr_auc"])["epoch"]),
        "profile": args.profile, "train_rows": int(len(train_np)), "neg_keep_rate": neg_rate, "logit_shift": logit_shift,
        "total_seconds": round(time.time() - t_start, 1),
        "args": vars(args), "log": log_lines,
    })
    save_json(res, outdir / f"metrics_{args.model}.json")
    save_preds(ids["test"], p_test, outdir / f"preds_{args.model}.parquet")
    save_preds(ids["val"], p_val, outdir / f"preds_{args.model}_val.parquet")      # usadas pela rede de Hopfield
    pd.DataFrame(history).to_csv(outdir / f"history_{args.model}.csv", index=False)
    imp.to_csv(outdir / f"importance_{args.model}.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "model": args.model, "args": vars(args),
                "preprocess": preprocess, "logit_shift": logit_shift, "feature_cols": feature_cols,
                "profile": args.profile},
               mdir / f"{args.model}.pt")

    t = res["test"]
    log(f"\n{args.model} teste: ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | Brier {t['brier']:.4f} | "
        f"P {t['precision']:.3f} R {t['recall']:.3f} F1 {t['f1']:.3f} | top5%: P {t['precision_top5']:.3f} "
        f"R {t['recall_top5']:.3f} | {res['total_seconds']:.0f}s")
    log("importância (queda de ROC-AUC): " + ", ".join(f"{r.feature} {r.roc_auc_drop:+.4f}" for r in imp.head(8).itertuples()))


if __name__ == "__main__":
    main()
