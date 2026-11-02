#!/usr/bin/env python3
"""
SeismicAI - download do catálogo sísmico do USGS (ComCat / FDSN Event Web Service).

Baixa todos os eventos com magnitude >= MINMAG entre START e END, em blocos anuais.
Padrão: M >= 4,0 desde 2000-01-01 (perfil pulse). Para o perfil global: --start 1990-01-01 --minmag 4.5
O USGS limita cada consulta a 20.000 eventos; se um bloco ultrapassar esse limite,
o script divide o intervalo ao meio automaticamente (recursivo).

Uso (da raiz do projeto):
    python src/download_usgs.py
    python src/download_usgs.py --start 1990-01-01 --minmag 4.5 --outdir data/raw

Saída:
    data/raw/usgs_m40_full.csv   todas as colunas retornadas pelo USGS (entrega "dataset")
    data/raw/usgs_m40_min.csv    só as colunas usadas na modelagem (arquivo menor)

Documentação da API: https://earthquake.usgs.gov/fdsnws/event/1/
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Console do Windows (cp1252) não imprime "≥" e afins: força UTF-8 na saída (sem efeito em Linux/macOS)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import pandas as pd
import requests

API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
LIMIT = 20_000  # limite de eventos por consulta imposto pelo USGS
MIN_COLS = ["time", "latitude", "longitude", "depth", "mag", "magType", "type", "id"]
HEADERS = {"User-Agent": "SeismicAI academic project (python-requests)"}


def fetch_chunk(session: requests.Session, start: datetime, end: datetime,
                minmag: float, retries: int = 5, pause: float = 0.5):
    """Uma consulta à API. Retorna um DataFrame, ou None se o intervalo excede o limite."""
    params = {
        "format": "csv",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": minmag,
        "orderby": "time-asc",
        "limit": LIMIT,
    }
    for attempt in range(1, retries + 1):
        try:
            r = session.get(API_URL, params=params, headers=HEADERS, timeout=180)
        except requests.RequestException as exc:
            print(f"    erro de rede ({exc}); tentativa {attempt}/{retries}", file=sys.stderr)
            time.sleep(5 * attempt)
            continue

        if r.status_code == 200:
            time.sleep(pause)  # gentileza com o servidor
            text = r.text.strip()
            return pd.read_csv(io.StringIO(text)) if text else pd.DataFrame()
        if r.status_code == 204:  # sem eventos no intervalo
            return pd.DataFrame()
        if r.status_code == 400 and "exceed" in r.text.lower():
            return None  # mais de 20.000 eventos: quem chamou divide o intervalo
        if r.status_code in (429, 500, 502, 503, 504):
            print(f"    HTTP {r.status_code}; aguardando ({attempt}/{retries})", file=sys.stderr)
            time.sleep(10 * attempt)
            continue
        raise RuntimeError(f"HTTP {r.status_code} para {params}: {r.text[:300]}")

    raise RuntimeError(f"Falha após {retries} tentativas: {start:%Y-%m-%d} .. {end:%Y-%m-%d}")


def download_range(session, start, end, minmag, depth=0, retries=5) -> pd.DataFrame:
    """Baixa [start, end); divide o intervalo se ele exceder o limite do USGS."""
    df = fetch_chunk(session, start, end, minmag, retries=retries)
    if df is None or len(df) >= LIMIT:
        if end - start <= timedelta(days=1):
            raise RuntimeError("Um único dia excede 20.000 eventos; aumente --minmag.")
        mid = start + (end - start) / 2
        print(f"    {'  ' * depth}{start:%Y-%m-%d}..{end:%Y-%m-%d} excede o limite; dividindo")
        return pd.concat(
            [download_range(session, start, mid, minmag, depth + 1, retries),
             download_range(session, mid, end, minmag, depth + 1, retries)],
            ignore_index=True,
        )
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2000-01-01", help="data inicial (UTC); perfil global: 1990-01-01")
    ap.add_argument("--end", default=None, help="data final (UTC); padrão: agora")
    ap.add_argument("--minmag", type=float, default=4.0, help="magnitude mínima; perfil global: 4.5")
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    if args.end:
        end = datetime.fromisoformat(args.end)
    else:
        end = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "m" + str(args.minmag).replace(".", "")

    session = requests.Session()
    parts: list[pd.DataFrame] = []
    t0 = time.time()
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(datetime(chunk_start.year + 1, 1, 1), end)
        df = download_range(session, chunk_start, chunk_end, args.minmag)
        parts.append(df)
        total = sum(len(p) for p in parts)
        print(f"{chunk_start.year}: {len(df):>6} eventos   (acumulado {total:>7}, {time.time() - t0:4.0f}s)")
        chunk_start = chunk_end

    full = pd.concat(parts, ignore_index=True)
    n_before = len(full)
    full = (full.drop_duplicates(subset="id", keep="last")
                .sort_values("time")
                .reset_index(drop=True))

    full_path = outdir / f"usgs_{tag}_full.csv"
    min_path = outdir / f"usgs_{tag}_min.csv"
    full.to_csv(full_path, index=False)
    full[MIN_COLS].to_csv(min_path, index=False)

    print("\nResumo")
    print(f"  eventos baixados      : {n_before}")
    print(f"  eventos únicos (id)   : {len(full)}")
    print(f"  período               : {full['time'].min()}  ->  {full['time'].max()}")
    print(f"  tipos de evento       : {full['type'].value_counts().to_dict()}")
    print(f"  arquivo completo      : {full_path}  ({full_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  arquivo p/ modelagem  : {min_path}  ({min_path.stat().st_size / 1e6:.1f} MB)  <- envie este")


if __name__ == "__main__":
    main()
