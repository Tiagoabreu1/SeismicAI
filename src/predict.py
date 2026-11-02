#!/usr/bin/env python3
"""
SeismicAI - motor de inferência do protótipo.

A partir do catálogo bruto do projeto (opcionalmente atualizado pelo feed do USGS), calcula as
features de qualquer semana para as células ativas com AS MESMAS funções usadas no treino
(build_dataset.weekly_arrays / window_features / panel_columns) e aplica:
  - o modelo probabilístico (lstm, mlp ou mlp_seq)  -> p = P(evento-alvo na semana seguinte)
  - a rede de Hopfield                                -> estado NORMAL / ATENÇÃO / ALARME
  - a RBM                                             -> probabilidade das ocultas + score de atipicidade

Uso em linha de comando (teste rápido):
    python src/predict.py --profile pulse --model lstm            # última semana completa do catálogo
    python src/predict.py --profile pulse --model lstm --feed     # atualiza pelo USGS antes
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import (PROFILES, WEEK, SEQ_LEN, assign_grid, cell_centers, clean_events,  # noqa: E402
                           panel_columns, segment_of, week_origin, weekly_arrays, window_features)
from seismic_data import Scaler  # noqa: E402
from train_hopfield import STATES, Hopfield, classify, to_bipolar  # noqa: E402
from train_nn import LSTMNet, MLP, MLPSeq  # noqa: E402
from train_rbm import Unit01Scaler  # noqa: E402

MIN_COLS = ["time", "latitude", "longitude", "depth", "mag", "magType", "type", "id"]


# ----------------------------------------------------------------------------- feed do USGS
def fetch_usgs(start: pd.Timestamp, end: pd.Timestamp | None, min_mag: float) -> pd.DataFrame:
    """Eventos M >= min_mag entre start e end (UTC) pelo FDSN Event Web Service; divide o intervalo se exceder 20.000."""
    import requests
    from download_usgs import download_range
    end = end or (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1))
    session = requests.Session()
    df = download_range(session, start.tz_convert(None).to_pydatetime(), end.tz_convert(None).to_pydatetime(), min_mag,
                        retries=2)
    if df.empty:
        return pd.DataFrame(columns=MIN_COLS)
    return df[[c for c in MIN_COLS if c in df.columns]]


# ----------------------------------------------------------------------------- modelos
def load_model(path: Path, device="cpu"):
    b = torch.load(path, map_location=device, weights_only=False)
    a = b["args"]
    kind, pre = b["model"], b["preprocess"]
    if kind == "mlp":
        feat_scaler = Scaler.from_dict(pre["features"])
        model = MLP(len(feat_scaler.cols), tuple(int(h) for h in a["mlp_hidden"].split(",")), a["dropout"])
        bundle = {"kind": kind, "feat_scaler": feat_scaler}
    else:
        seq_scaler, stat_scaler = Scaler.from_dict(pre["sequence"]), Scaler.from_dict(pre["static"])
        if kind == "lstm":
            model = LSTMNet(len(seq_scaler.cols), len(stat_scaler.cols), a["hidden"], a["layers"], dropout=a["dropout"])
        else:
            model = MLPSeq(pre["seq_len"], len(seq_scaler.cols), len(stat_scaler.cols), dropout=a["dropout"])
        bundle = {"kind": kind, "seq_scaler": seq_scaler, "stat_scaler": stat_scaler, "seq_len": int(pre.get("seq_len", SEQ_LEN))}
    model.load_state_dict(b["state_dict"])
    model.eval()
    bundle.update({"model": model, "logit_shift": float(b.get("logit_shift", 0.0)), "feature_cols": b["feature_cols"]})
    return bundle


class Engine:
    """Carrega perfil, células, catálogo e modelos; calcula features e previsões para qualquer semana."""

    def __init__(self, profile: str = "pulse", model: str = "lstm", root: Path = ROOT):
        self.profile, self.root = profile, Path(root)
        proc, mdir = self.root / "data" / "processed", self.root / "models" / profile
        self.cfg = json.loads((proc / f"summary_{profile}.json").read_text(encoding="utf-8"))
        self.cells = pd.read_csv(proc / f"cells_{profile}.csv")
        self.active = self.cells["cell"].to_numpy()
        self.origin = week_origin(self.cfg)
        self.model_name = model
        self.model = load_model(mdir / f"{model}.pt")
        hp = mdir / "hopfield.json"
        self.hop = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else None
        if self.hop:
            self.hop_net = Hopfield(np.array(self.hop["prototypes"]), self.hop["rule"])
            self.hop_cls = np.array(self.hop["prototype_states"])
        rp = mdir / "rbm.pt"
        self.rbm = torch.load(rp, map_location="cpu", weights_only=False) if rp.exists() else None
        self.events: pd.DataFrame | None = None
        self._cache: dict = {}

    # ------------------------------------------------------------------ catálogo
    def load_catalog(self, with_feed: bool = False, raw_path: Path | None = None) -> dict:
        raw_path = raw_path or (self.root / self.cfg["input"])
        raw = pd.read_csv(raw_path, low_memory=False)
        info = {"catalogo": str(raw_path), "eventos_projeto": int(len(raw)), "feed": 0}
        if with_feed:
            last = pd.to_datetime(raw["time"], utc=True).max()
            new = fetch_usgs(last - pd.Timedelta(days=2), None, self.cfg["min_mag"])
            info["feed"] = int(len(new))
            raw = pd.concat([raw, new], ignore_index=True)
        df, stats = clean_events(raw, self.cfg, end=None)          # sem corte no fim: usa tudo até hoje
        self.events = assign_grid(df, self.cfg["cell"], self.origin)
        self._cache = {}
        data_end = self.events["time"].max()
        self.n_weeks = int((data_end - self.origin) // WEEK)        # semanas 0..n_weeks-1 completas
        info.update({"eventos_limpos": int(len(df)), "ultimo_evento": str(data_end), "limpeza": stats,
                     "ultima_semana_completa": int(self.n_weeks - 1),
                     "inicio_ultima_semana_completa": str(self.week_start(self.n_weeks - 1).date())})
        return info

    def week_start(self, week: int) -> pd.Timestamp:
        return self.origin + week * WEEK

    def week_of(self, ts) -> int:
        return int((pd.Timestamp(ts, tz="UTC") - self.origin) // WEEK)

    # ------------------------------------------------------------------ features
    def arrays(self):
        if "A" not in self._cache:
            A = weekly_arrays(self.events, self.cfg, self.active, self.n_weeks)
            feats = window_features(A, self.cfg)
            self._cache.update({"A": A, "feats": feats, "panel": panel_columns(A, feats, self.cfg)})
        return self._cache["A"], self._cache["feats"], self._cache["panel"]

    def static_frame(self) -> pd.DataFrame:
        c = self.cells
        st = pd.DataFrame({"cell": c["cell"], "lat_c": c["lat_c"], "lon_c": c["lon_c"],
                           "cos_lon": np.cos(np.radians(c["lon_c"])), "sin_lon": np.sin(np.radians(c["lon_c"])),
                           "rate_target_train": c["rate_target_train"], "rate_any_train": c["rate_any_train"],
                           "mean_n_train": c["mean_n_train"]})
        return st

    def feature_frame(self, week: int) -> pd.DataFrame:
        """Uma linha por célula ativa com as features tabulares e estáticas da semana `week`."""
        assert self.model.get("seq_len", SEQ_LEN) - 1 <= week <= self.n_weeks - 1, f"semana {week} fora do intervalo com histórico"
        _, feats, _ = self.arrays()
        df = self.static_frame().copy()
        for k, arr in feats.items():
            df[k] = arr[:, week].astype(np.float32)
        df["week"] = week
        df["week_start"] = self.week_start(week).tz_localize(None)
        df["segment"] = self.cells["segment"].to_numpy() if "segment" in self.cells else None
        return df

    def sequences(self, week: int, rows: np.ndarray | None = None) -> np.ndarray:
        """[n, 52, variáveis] com as variáveis do painel nas semanas week-51..week, na ordem do treino."""
        _, _, panel = self.arrays()
        cols = self.model["seq_scaler"].cols
        L = self.model.get("seq_len", SEQ_LEN)
        rows = np.arange(len(self.active)) if rows is None else rows
        seq = np.stack([panel[c][rows, week - L + 1:week + 1] for c in cols], axis=-1)
        return seq.astype(np.float32)

    # ------------------------------------------------------------------ previsão
    @torch.no_grad()
    def probabilities(self, week: int, rows: np.ndarray | None = None) -> np.ndarray:
        m = self.model
        feat = self.feature_frame(week)
        if rows is not None:
            feat = feat.iloc[rows]
        if m["kind"] == "mlp":
            X = torch.tensor(m["feat_scaler"].transform(feat[m["feature_cols"]].to_numpy(np.float32)))
            logits = m["model"](X)
        else:
            seq = torch.tensor(m["seq_scaler"].transform(self.sequences(week, rows)))
            static = torch.tensor(m["stat_scaler"].transform(feat[m["stat_scaler"].cols].to_numpy(np.float32)))
            logits = m["model"](seq, static)
        return torch.sigmoid(logits + m["logit_shift"]).numpy()

    @torch.no_grad()
    def rbm_scores(self, feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        r = self.rbm
        sc = Unit01Scaler.from_dict(r["scaler"])
        V = torch.tensor(sc.transform(feat[r["features"]].to_numpy(np.float32)))
        W, b, c = r["rbm"]["W"], r["rbm"]["b"], r["rbm"]["c"]
        H = torch.sigmoid(V @ W + c)
        p_h = torch.sigmoid(H @ torch.tensor(r["logreg_coef"], dtype=torch.float32) + r["logreg_intercept"])
        F = -(V @ b) - torch.nn.functional.softplus(V @ W + c).sum(dim=1)
        p_e = torch.sigmoid(r["platt"]["coef"] * F + r["platt"]["intercept"])
        return p_h.numpy(), p_e.numpy()

    def predict(self, week: int) -> pd.DataFrame:
        """Tabela por célula: features resumidas, p do modelo, estado da Hopfield, scores da RBM."""
        feat = self.feature_frame(week)
        feat["p"] = self.probabilities(week)
        if self.hop:
            S = to_bipolar(feat, self.hop["bits"])
            pred, spurious, _, _ = classify(self.hop_net, self.hop_cls, S)
            feat["state"] = [STATES[c] for c in pred]
            feat["spurious"] = spurious
        if self.rbm:
            feat["p_rbm"], feat["p_rbm_energy"] = self.rbm_scores(feat)
        feat["label_week_start"] = feat["week_start"] + WEEK
        return feat

    def cell_history(self, cell: int, weeks: np.ndarray) -> pd.DataFrame:
        """p e atividade semanal de uma célula ao longo de várias semanas (para o gráfico de detalhe)."""
        row = int(np.flatnonzero(self.active == cell)[0])
        A, _, _ = self.arrays()
        p = np.array([self.probabilities(int(w), np.array([row]))[0] for w in weeks])
        return pd.DataFrame({"week": weeks, "week_start": [self.week_start(int(w)).tz_localize(None) for w in weeks],
                             "p": p, "n": A["N"][row, weeks], "max_mag": A["MAXM"][row, weeks], "nbig": A["NBIG"][row, weeks]})

    def events_in_week(self, week: int, min_mag: float | None = None) -> pd.DataFrame:
        ev = self.events[self.events["week"] == week]
        if min_mag:
            ev = ev[ev["mag"] >= min_mag]
        return ev.sort_values("time")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--model", default="lstm")
    ap.add_argument("--feed", action="store_true", help="atualiza o catálogo pelo feed do USGS antes de prever")
    ap.add_argument("--week", type=int, default=None, help="índice da semana (padrão: última completa)")
    args = ap.parse_args()
    eng = Engine(args.profile, args.model)
    info = eng.load_catalog(with_feed=args.feed)
    print(json.dumps({k: v for k, v in info.items() if k != "limpeza"}, indent=2, ensure_ascii=False))
    week = args.week if args.week is not None else eng.n_weeks - 1
    out = eng.predict(week)
    cols = ["cell", "segment", "lat_c", "lon_c", "p"] + (["state"] if eng.hop else []) + ["n_4w", "maxmag_4w", "nbig_4w"]
    print(f"\nsemana {week} (início {eng.week_start(week).date()}) -> previsão para a semana de {eng.week_start(week + 1).date()}")
    print(out.sort_values("p", ascending=False)[cols].head(15).to_string(index=False))
    if eng.hop:
        print("\nestados:", out["state"].value_counts().to_dict())


if __name__ == "__main__":
    main()
