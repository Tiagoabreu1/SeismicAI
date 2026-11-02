#!/usr/bin/env python3
"""
SeismicAI Pulse - publicação das previsões no Firebase (Cloud Firestore).

Gera um documento por semana, lido pelo site (web/) e por qualquer outro cliente:

  weeks/{AAAA-MM-DD}   (id = início da semana prevista)
      week, week_start, label_week_start, kind ("ao_vivo" | "historico"), profile, model,
      label_mag, alarm_mag, catalog_end, stats {...},
      cells  [ {c, lat, lon, seg, p, st, n1, n4, mx4, nb4, nv1, (y, mx)} ... ]   uma entrada por célula
      events [ {lat, lon, mag, t} ... ]   eventos M >= alvo: da semana de entrada (ao vivo) ou da semana
                                          prevista (histórico = o que de fato aconteceu)
  meta/latest
      profile, model, label_mag, alarm_mag, latest (id da semana ao vivo), weeks [ {id, kind, ...} ],
      metrics (teste), hopfield (teste), updated_at

Cada semana cabe em um documento (~100 KB), então o site faz uma leitura por semana.

Uso:
    python src/publish_firebase.py --feed                     # semana corrente, catálogo atualizado pelo USGS
    python src/publish_firebase.py --feed --backfill-test 52  # + últimas 52 semanas do teste (modo histórico)
    python src/publish_firebase.py --dry-run                  # grava em web/public/data/ em vez do Firestore

Credenciais: chave de conta de serviço do projeto Firebase em secrets/serviceAccount.json
(ou --credentials, ou a variável GOOGLE_APPLICATION_CREDENTIALS). Instale: pip install firebase-admin
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import Engine  # noqa: E402

CELL_KEYS = {"cell": "c", "lat_c": "lat", "lon_c": "lon", "segment": "seg", "p": "p", "state": "st",
             "n_1w": "n1", "n_4w": "n4", "maxmag_4w": "mx4", "nbig_4w": "nb4", "nviz_1w": "nv1"}


def jsonable(obj):
    """Converte tipos numpy/pandas para tipos nativos e troca NaN por null (JSON válido no navegador)."""
    return _sanitize(json.loads(json.dumps(obj, default=_default, ensure_ascii=False)))


def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o is pd.NaT or (isinstance(o, float) and np.isnan(o)):
        return None
    raise TypeError(f"não serializável: {type(o)}")


def cells_payload(df: pd.DataFrame, with_truth: bool) -> list[dict]:
    out = []
    for r in df.itertuples(index=False):
        d = {v: getattr(r, k) for k, v in CELL_KEYS.items() if hasattr(r, k)}
        d["p"] = round(float(d["p"]), 4)
        for k in ("mx4",):
            d[k] = round(float(d[k]), 1)
        for k in ("n1", "n4", "nb4", "nv1"):
            d[k] = int(d[k])
        if with_truth:
            d["y"] = int(r.y)
            d["mx"] = round(float(r.next_maxmag), 1) if hasattr(r, "next_maxmag") and not pd.isna(r.next_maxmag) else None
        out.append(d)
    return out


def events_payload(ev: pd.DataFrame) -> list[dict]:
    return [{"lat": round(float(r.latitude), 3), "lon": round(float(r.longitude), 3), "mag": round(float(r.mag), 1),
             "t": pd.Timestamp(r.time).strftime("%Y-%m-%dT%H:%MZ")} for r in ev.itertuples(index=False)]


def stats_payload(df: pd.DataFrame, alarm_mag: float, with_truth: bool) -> dict:
    s = {"n_cells": int(len(df)), "p_mean": round(float(df["p"].mean()), 4), "p_max": round(float(df["p"].max()), 4)}
    if "state" in df:
        s["n_alarme"] = int((df["state"] == "ALARME").sum())
        s["n_atencao"] = int((df["state"] == "ATENÇÃO").sum())
    if with_truth:
        top = df.sort_values("p", ascending=False).head(max(1, int(round(0.05 * len(df)))))
        s.update({"n_hits": int(df["y"].sum()), "top5_precision": round(float(top["y"].mean()), 3)})
        if "state" in df:
            al = df["state"] == "ALARME"
            s.update({"alarms": int(al.sum()), "alarm_hits": int((al & (df["next_maxmag"] >= alarm_mag)).sum())})
    return s


def live_document(eng: Engine, info: dict, model: str, alarm_mag: float) -> dict:
    week = eng.n_weeks - 1
    df = eng.predict(week)
    ev = eng.events_in_week(week, eng.cfg["label_mag"])
    label_start = eng.week_start(week + 1).date()
    return jsonable({
        "id": str(label_start), "week": int(week), "week_start": str(eng.week_start(week).date()),
        "label_week_start": str(label_start), "kind": "ao_vivo", "profile": eng.profile, "model": model,
        "label_mag": eng.cfg["label_mag"], "alarm_mag": alarm_mag,
        "catalog_end": pd.Timestamp(info["ultimo_evento"]).strftime("%Y-%m-%dT%H:%MZ"),
        "stats": stats_payload(df, alarm_mag, False), "cells": cells_payload(df, False), "events": events_payload(ev),
    })


def historical_documents(profile: str, model: str, alarm_mag: float, cells: pd.DataFrame, n_last: int) -> list[dict]:
    rdir, proc = ROOT / "results" / profile, ROOT / "data" / "processed"
    preds = pd.read_parquet(rdir / f"preds_{model}.parquet")
    hop_f = rdir / "preds_hopfield.parquet"
    hop = pd.read_parquet(hop_f)[["cell", "week", "hopfield_state", "next_maxmag"]] if hop_f.exists() else None
    events = pd.read_parquet(proc / f"events_clean_{profile}.parquet")
    label_mag = float(json.loads((proc / f"summary_{profile}.json").read_text(encoding="utf-8"))["label_mag"])
    weeks = np.sort(preds["week"].unique())[-n_last:]
    docs = []
    for w in weeks:
        feats = pd.read_parquet(proc / f"dataset_{profile}.parquet", filters=[("week", "==", int(w))])
        df = feats.merge(preds.loc[preds["week"] == w, ["cell", "p"]], on="cell").merge(cells, on="cell", how="left")
        if hop is not None:
            df = df.merge(hop[hop["week"] == w], on=["cell", "week"], how="left").rename(columns={"hopfield_state": "state"})
        else:
            df["next_maxmag"] = np.nan
        ws = pd.Timestamp(df["week_start"].iloc[0]).date()
        ls = pd.Timestamp(df["label_week_start"].iloc[0]).date()
        ev = events[(events["week"] == w + 1) & (events["mag"] >= label_mag)]
        docs.append(jsonable({
            "id": str(ls), "week": int(w), "week_start": str(ws), "label_week_start": str(ls), "kind": "historico",
            "profile": profile, "model": model, "label_mag": label_mag, "alarm_mag": alarm_mag, "catalog_end": None,
            "stats": stats_payload(df, alarm_mag, True), "cells": cells_payload(df, True), "events": events_payload(ev),
        }))
    return docs


def meta_document(profile: str, model: str, label_mag: float, alarm_mag: float, cell_deg: float,
                  latest_id: str | None, docs: list[dict]) -> dict:
    rdir = ROOT / "results" / profile
    metrics, hop = {}, {}
    f = rdir / f"metrics_{model}.json"
    if f.exists():
        t = json.loads(f.read_text(encoding="utf-8"))["test"]
        metrics = {k: round(t[k], 4) for k in ("roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "precision_top5", "recall_top5")}
    f = rdir / "hopfield_results.json"
    if f.exists():
        h = json.loads(f.read_text(encoding="utf-8"))
        hop = {"accuracy": round(h["accuracy_test"], 4), "alarm": h["alarm_test"]["hopfield"], "base_rate": h["alarm_test"]["base_rate"],
               "per_state": h["per_state_test"]}
    weeks = sorted(({"id": d["id"], "kind": d["kind"], "week_start": d["week_start"], "label_week_start": d["label_week_start"],
                     "n_alarme": d["stats"].get("n_alarme"), "n_hits": d["stats"].get("n_hits")} for d in docs),
                   key=lambda x: x["label_week_start"], reverse=True)
    return jsonable({"profile": profile, "model": model, "label_mag": label_mag, "alarm_mag": alarm_mag, "cell_deg": cell_deg,
                     "latest": latest_id, "weeks": weeks, "metrics": metrics, "hopfield": hop,
                     "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def write_local(docs: list[dict], meta: dict, outdir: Path) -> None:
    (outdir / "weeks").mkdir(parents=True, exist_ok=True)
    for d in docs:
        (outdir / "weeks" / f"{d['id']}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def write_firestore(docs: list[dict], meta: dict, credentials_path: str | None, merge_meta: bool) -> None:
    import firebase_admin
    from firebase_admin import credentials, firestore
    cred_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or str(ROOT / "secrets" / "serviceAccount.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    db = firestore.client()
    for d in docs:
        db.collection("weeks").document(d["id"]).set(d)
        print(f"  weeks/{d['id']} ({d['kind']}, {len(json.dumps(d)) / 1024:.0f} KB)")
    if merge_meta:                                   # mantém as semanas já publicadas que não foram regeradas agora
        snap = db.collection("meta").document("latest").get()
        if snap.exists:
            old = snap.to_dict()
            known = {w["id"]: w for w in old.get("weeks", [])}
            known.update({w["id"]: w for w in meta["weeks"]})
            meta["weeks"] = sorted(known.values(), key=lambda x: x["label_week_start"], reverse=True)
            meta["latest"] = meta["latest"] or old.get("latest")
    db.collection("meta").document("latest").set(meta)
    print(f"  meta/latest ({len(meta['weeks'])} semanas listadas; ao vivo = {meta['latest']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="pulse")
    ap.add_argument("--model", default="lstm")
    ap.add_argument("--feed", action="store_true", help="atualiza o catálogo pelo USGS antes de prever a semana corrente")
    ap.add_argument("--no-live", action="store_true", help="não publica a semana corrente")
    ap.add_argument("--backfill-test", type=int, default=0, metavar="N", help="publica também as últimas N semanas do teste")
    ap.add_argument("--dry-run", action="store_true", help="grava JSON em web/public/data/ em vez do Firestore")
    ap.add_argument("--credentials", default=None, help="chave da conta de serviço (JSON)")
    ap.add_argument("--outdir", default=str(ROOT / "web" / "public" / "data"))
    args = ap.parse_args()

    proc = ROOT / "data" / "processed"
    summary = json.loads((proc / f"summary_{args.profile}.json").read_text(encoding="utf-8"))
    hop_f = ROOT / "models" / args.profile / "hopfield.json"
    alarm_mag = float(json.loads(hop_f.read_text(encoding="utf-8"))["alarm_mag"]) if hop_f.exists() else float(summary["big_mag"])
    cells = pd.read_csv(proc / f"cells_{args.profile}.csv")[["cell", "segment"]]

    docs, latest_id = [], None
    if not args.no_live:
        eng = Engine(args.profile, args.model, ROOT)
        info = eng.load_catalog(with_feed=args.feed)
        print(f"catálogo até {info['ultimo_evento'][:10]} ({info['eventos_limpos']:,} eventos; {info['feed']} do feed)")
        live = live_document(eng, info, args.model, alarm_mag)
        latest_id = live["id"]
        docs.append(live)
        print(f"semana ao vivo: {live['id']} | ALARME {live['stats'].get('n_alarme')} | ATENÇÃO {live['stats'].get('n_atencao')} | p máx {live['stats']['p_max']:.1%}")
    if args.backfill_test:
        hist = historical_documents(args.profile, args.model, alarm_mag, cells, args.backfill_test)
        docs += hist
        print(f"histórico: {len(hist)} semanas ({hist[0]['id']} .. {hist[-1]['id']})")
    meta = meta_document(args.profile, args.model, float(summary["label_mag"]), alarm_mag, float(summary["cell"]), latest_id, docs)

    if args.dry_run:
        write_local(docs, meta, Path(args.outdir))
        print(f"dry-run: {len(docs)} documentos em {args.outdir} (abra o site com ?source=local)")
    else:
        write_firestore(docs, meta, args.credentials, merge_meta=True)
        print("publicado no Firestore")


if __name__ == "__main__":
    main()
