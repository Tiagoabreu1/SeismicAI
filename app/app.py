"""
SeismicAI Pulse - protótipo (Streamlit).

Mapa semanal do Círculo de Fogo com a probabilidade de terremoto (M >= 4,0) na semana seguinte por
célula, dada pelo modelo escolhido (LSTM por padrão), e o estado de alerta decidido pela rede de
Hopfield (NORMAL / ATENÇÃO / ALARME). Dois modos:
  - Ao vivo    : catálogo do projeto, opcionalmente atualizado pelo feed do USGS, última semana completa;
  - Histórico  : qualquer semana do conjunto de teste, com o que de fato aconteceu na semana seguinte.

Rodar (da raiz do projeto, com data/, models/ e results/ gerados):
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import Engine  # noqa: E402

STATE_COLOR = {"ALARME": "#c1121f", "ATENÇÃO": "#f0a202", "NORMAL": "#6c757d"}
MODEL_LABEL = {"lstm": "LSTM", "mlp": "MLP (janelas)", "mlp_seq": "MLP (entradas da LSTM)"}
PROFILE_LABEL = {"pulse": "SeismicAI Pulse — Círculo de Fogo, M ≥ 4,0, células 2°",
                 "global": "Global — M ≥ 5,0, células 3°"}

st.set_page_config(page_title="SeismicAI Pulse", page_icon="🌋", layout="wide")


# ----------------------------------------------------------------------------- carga (com cache)
def available_profiles() -> list[str]:
    out = []
    for p in ("pulse", "global"):
        if (ROOT / "models" / p).exists() and any((ROOT / "models" / p).glob("*.pt")) \
                and (ROOT / "data" / "processed" / f"summary_{p}.json").exists():
            out.append(p)
    return out


def available_models(profile: str) -> list[str]:
    return [m for m in ("lstm", "mlp", "mlp_seq") if (ROOT / "models" / profile / f"{m}.pt").exists()]


@st.cache_resource(show_spinner="Carregando modelos e catálogo…")
def get_engine(profile: str, model: str, feed_version: int) -> tuple[Engine, dict]:
    eng = Engine(profile, model, ROOT)
    info = eng.load_catalog(with_feed=feed_version > 0)
    return eng, info


@st.cache_data(show_spinner=False)
def load_test_preds(profile: str, model: str) -> pd.DataFrame | None:
    f = ROOT / "results" / profile / f"preds_{model}.parquet"
    if not f.exists():
        return None
    p = pd.read_parquet(f)
    h = ROOT / "results" / profile / "preds_hopfield.parquet"
    if h.exists():
        hp = pd.read_parquet(h)[["cell", "week", "hopfield_state", "true_state", "next_maxmag"]]
        p = p.merge(hp, on=["cell", "week"], how="left")
    return p


@st.cache_data(show_spinner=False)
def load_week_features(profile: str, week: int) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / f"dataset_{profile}.parquet", filters=[("week", "==", week)])


@st.cache_data(show_spinner=False)
def load_events(profile: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / f"events_clean_{profile}.parquet")


@st.cache_data(show_spinner=False)
def load_summary(profile: str) -> dict:
    return json.loads((ROOT / "data" / "processed" / f"summary_{profile}.json").read_text(encoding="utf-8"))


def read_md(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


# ----------------------------------------------------------------------------- figuras
def map_figure(df: pd.DataFrame, profile: str, title: str, overlay: pd.DataFrame | None, label_mag: float) -> go.Figure:
    fig = go.Figure()
    cmax = max(0.5, float(np.nanmax(df["p"])) if len(df) else 0.5)
    has_state = "state" in df.columns
    order = ["NORMAL", "ATENÇÃO", "ALARME"] if has_state else ["todas"]
    for st_name in order:
        sub = df[df["state"] == st_name] if has_state else df
        if sub.empty:
            continue
        hover = [f"<b>{r.segment or ''}</b><br>célula {int(r.cell)} · {r.lat_c:.0f}°, {r.lon_c:.0f}°"
                 f"<br>p (semana seguinte): <b>{r.p:.1%}</b>" + (f"<br>estado: <b>{r.state}</b>" if has_state else "")
                 + f"<br>eventos: {int(r.n_1w)} (1 sem) · {int(r.n_4w)} (4 sem)"
                 f"<br>mag. máx 4 sem: {r.maxmag_4w:.1f} · grandes 4 sem: {int(r.nbig_4w)} · vizinhos 1 sem: {int(r.nviz_1w)}"
                 for r in sub.itertuples()]
        line = dict(width=2.2, color=STATE_COLOR["ALARME"]) if st_name == "ALARME" else \
            dict(width=1.4, color=STATE_COLOR["ATENÇÃO"]) if st_name == "ATENÇÃO" else dict(width=0)
        fig.add_trace(go.Scattergeo(
            lon=sub["lon_c"], lat=sub["lat_c"], mode="markers", name=st_name if has_state else "células",
            marker=dict(symbol="square", size=9 if profile == "pulse" else 8, color=sub["p"], colorscale="YlOrRd",
                        cmin=0, cmax=cmax, line=line, showscale=(st_name == order[-1]),
                        colorbar=dict(title="p", tickformat=".0%", len=0.7)),
            text=hover, hoverinfo="text"))
    if overlay is not None and len(overlay):
        fig.add_trace(go.Scattergeo(
            lon=overlay["longitude"], lat=overlay["latitude"], mode="markers",
            name=f"eventos M ≥ {label_mag:g} ({len(overlay)})",
            marker=dict(symbol="circle-open", size=6 + 5 * (overlay["mag"] - label_mag).clip(lower=0) ** 1.5,
                        color="#1d3557", line=dict(width=1.2, color="#1d3557")),
            text=[f"M {m:.1f} · {t:%d/%m %H:%M} UTC" for m, t in zip(overlay["mag"], overlay["time"])], hoverinfo="text"))
    geo = dict(projection_type="natural earth", showland=True, landcolor="#efefef", showocean=True, oceancolor="#e3edf7",
               showcountries=True, countrycolor="#d0d0d0", coastlinecolor="#b0b0b0", showframe=False)
    if profile == "pulse":                      # Pacífico no centro
        geo.update(projection_rotation=dict(lon=180), lataxis_range=[-62, 70])
    else:
        geo.update(lataxis_range=[-70, 75])
    fig.update_geos(**geo)
    fig.update_layout(title=title, height=560, margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h", y=-0.02, x=0))
    return fig


def history_figure(h: pd.DataFrame, title: str, label_mag: float, big_mag: float) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=h["week_start"], y=h["n"], name=f"eventos M ≥ {label_mag:g} na semana",
                         marker_color="#9fb3c8"), secondary_y=False)
    big = h[h["nbig"] > 0]
    if len(big):
        fig.add_trace(go.Scatter(x=big["week_start"], y=big["n"], mode="markers", name=f"semana com M ≥ {big_mag:g}",
                                 marker=dict(symbol="star", size=11, color="#c1121f")), secondary_y=False)
    fig.add_trace(go.Scatter(x=h["week_start"], y=h["p"], mode="lines", name="p (semana seguinte)",
                             line=dict(color="#1d3557", width=2)), secondary_y=True)
    fig.update_yaxes(title_text="eventos", secondary_y=False)
    fig.update_yaxes(title_text="probabilidade", tickformat=".0%", range=[0, 1], secondary_y=True)
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
    return fig


def table_view(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    cols = {"segment": "Segmento", "lat_c": "Lat", "lon_c": "Lon", "p": "p", "state": "Estado", "n_1w": "Ev. 1 sem",
            "n_4w": "Ev. 4 sem", "maxmag_4w": "Mag. máx 4 sem", "nbig_4w": "Grandes 4 sem", "nviz_1w": "Vizinhos 1 sem",
            "p_rbm_energy": "Atipicidade (RBM)"}
    keep = [c for c in cols if c in df.columns]
    out = df.sort_values("p", ascending=False).head(n)[["cell"] + keep].rename(columns=cols)
    out = out.rename(columns={"cell": "Célula"})
    for c in ("p", "Atipicidade (RBM)"):
        if c in out:
            out[c] = (out[c] * 100).round(1).astype(str) + " %"
    for c in ("Ev. 1 sem", "Ev. 4 sem", "Grandes 4 sem", "Vizinhos 1 sem"):
        if c in out:
            out[c] = out[c].astype(int)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- app
profiles = available_profiles()
if not profiles:
    st.error("Nenhum modelo encontrado. Rode o pipeline (build_dataset, train_nn, train_hopfield) antes de abrir o app.")
    st.stop()

with st.sidebar:
    st.title("SeismicAI Pulse")
    profile = st.selectbox("Perfil", profiles, format_func=lambda p: PROFILE_LABEL[p])
    model = st.selectbox("Modelo probabilístico", available_models(profile), format_func=lambda m: MODEL_LABEL[m])
    mode = st.radio("Modo", ["Ao vivo", "Histórico (teste)"], horizontal=True)
    summary = load_summary(profile)
    label_mag, big_mag = float(summary["label_mag"]), float(summary["big_mag"])
    if "feed_version" not in st.session_state:
        st.session_state.feed_version = 0
    if mode == "Ao vivo":
        if st.button("Atualizar catálogo pelo feed do USGS", help="Baixa os eventos desde o último do catálogo do projeto (precisa de internet)."):
            st.session_state.feed_version += 1
    segments = sorted(s for s in pd.read_csv(ROOT / "data" / "processed" / f"cells_{profile}.csv")["segment"].dropna().unique())
    seg_filter = st.multiselect("Segmentos (vazio = todos)", segments)
    st.caption("Protótipo acadêmico. Não é sistema de alerta nem previsão determinística de terremotos; "
               "as probabilidades são estimativas estatísticas a partir do catálogo do USGS.")

hop_file = ROOT / "models" / profile / "hopfield.json"
hop_meta = json.loads(hop_file.read_text(encoding="utf-8")) if hop_file.exists() else None
alarm_mag = float(hop_meta["alarm_mag"]) if hop_meta else big_mag

if hop_meta and hop_meta.get("model") != model:
    st.sidebar.caption(f"A rede de Hopfield foi calibrada com as probabilidades da {MODEL_LABEL.get(hop_meta['model'], hop_meta['model'])}; "
                       f"no modo histórico os estados vêm dessa calibração.")
st.title("🌋 SeismicAI Pulse")
st.markdown(f"Probabilidade de pelo menos um terremoto **M ≥ {label_mag:g}** na **semana seguinte**, por célula, "
            f"estimada pela {MODEL_LABEL[model]} a partir das últimas 52 semanas do catálogo do USGS. "
            + (f"A rede de Hopfield resume cada célula em um estado de alerta; o **ALARME** é associado a terremotos "
               f"**M ≥ {alarm_mag:g}**." if hop_meta else ""))

# ----------------------------------------------------------------------------- modo ao vivo
if mode == "Ao vivo":
    try:
        eng, info = get_engine(profile, model, st.session_state.feed_version)
    except Exception as exc:  # feed indisponível: volta ao catálogo do projeto
        st.warning(f"Não foi possível atualizar pelo feed do USGS ({exc}). Usando o catálogo do projeto.")
        st.session_state.feed_version = 0
        eng, info = get_engine(profile, model, 0)
    week = eng.n_weeks - 1
    df = eng.predict(week)
    if seg_filter:
        df = df[df["segment"].isin(seg_filter)]
    ws, ls = eng.week_start(week).date(), eng.week_start(week + 1).date()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Catálogo até", pd.Timestamp(info["ultimo_evento"]).strftime("%d/%m/%Y"),
              help=f"{info['eventos_limpos']:,} eventos limpos; {info['feed']} vindos do feed do USGS")
    c2.metric("Semana prevista", f"{ls:%d/%m/%Y}", help=f"features da semana de {ws:%d/%m/%Y}")
    c3.metric("Células", f"{len(df):,}")
    if "state" in df:
        c4.metric("Em ALARME", int((df["state"] == "ALARME").sum()))
        c5.metric("Em ATENÇÃO", int((df["state"] == "ATENÇÃO").sum()))
    else:
        c4.metric("p média", f"{df['p'].mean():.1%}")
        c5.metric("p máxima", f"{df['p'].max():.1%}")
    overlay = eng.events_in_week(week, label_mag)
    st.plotly_chart(map_figure(df, profile, f"Semana de {ls:%d/%m/%Y} — probabilidade por célula "
                                            f"(círculos: eventos M ≥ {label_mag:g} da última semana)", overlay, label_mag))
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Células com maior probabilidade")
        st.dataframe(table_view(df), hide_index=True)
    with right:
        st.subheader("Detalhe da célula")
        opts = df.sort_values("p", ascending=False)
        chosen = st.selectbox("Célula", opts["cell"].tolist(),
                              format_func=lambda c: f"{int(c)} · {opts.set_index('cell').loc[c, 'segment']} · "
                                                    f"{opts.set_index('cell').loc[c, 'lat_c']:.0f}°, {opts.set_index('cell').loc[c, 'lon_c']:.0f}°")
        weeks = np.arange(max(51, week - 103), week + 1)
        h = eng.cell_history(int(chosen), weeks)
        st.plotly_chart(history_figure(h, "Últimos 2 anos: eventos por semana e p prevista", label_mag, big_mag))
        ev = eng.events[(eng.events["cell"] == chosen) & (eng.events["week"] >= week - 3)].sort_values("time", ascending=False)
        if len(ev):
            st.caption(f"Eventos na célula nas últimas 4 semanas ({len(ev)}):")
            st.dataframe(ev[["time", "mag", "depth"]].rename(columns={"time": "Hora (UTC)", "mag": "Mag.", "depth": "Prof. (km)"})
                         .head(12), hide_index=True)

# ----------------------------------------------------------------------------- modo histórico
else:
    preds = load_test_preds(profile, model)
    if preds is None:
        st.error(f"results/{profile}/preds_{model}.parquet não encontrado: treine o modelo antes.")
        st.stop()
    weeks = np.sort(preds["week"].unique())
    starts = {int(w): pd.Timestamp(preds.loc[preds["week"] == w, "label_week_start"].iloc[0]) - pd.Timedelta(days=7) for w in weeks}
    # semanas mais "interessantes" primeiro no seletor: as com mais eventos-alvo na semana seguinte
    hits = preds.groupby("week")["y"].sum()
    default = int(hits.idxmax())
    week = st.select_slider("Semana (início) — arraste para percorrer o período de teste",
                            options=[int(w) for w in weeks], value=default,
                            format_func=lambda w: f"{starts[w]:%d/%m/%Y}")
    feats = load_week_features(profile, week)
    df = feats.merge(preds[preds["week"] == week][["cell", "p"] + [c for c in ("hopfield_state", "next_maxmag") if c in preds.columns]], on="cell")
    cells = pd.read_csv(ROOT / "data" / "processed" / f"cells_{profile}.csv")[["cell", "segment"]]
    df = df.merge(cells, on="cell", how="left")
    if "hopfield_state" in df:
        df = df.rename(columns={"hopfield_state": "state"})
    if seg_filter:
        df = df[df["segment"].isin(seg_filter)]
    events = load_events(profile)
    nxt = events[(events["week"] == week + 1) & (events["mag"] >= label_mag)]
    ls = starts[week] + pd.Timedelta(days=7)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Semana prevista", f"{ls:%d/%m/%Y}")
    c2.metric("Células", f"{len(df):,}")
    c3.metric(f"Com M ≥ {label_mag:g} de fato", int(df["y"].sum()), help="células em que houve evento-alvo na semana seguinte")
    top = df.sort_values("p", ascending=False).head(max(1, int(0.05 * len(df))))
    c4.metric("Acertos no top 5 %", f"{top['y'].mean():.0%}", help="fração das 5 % células mais prováveis que tiveram evento")
    if "state" in df:
        c5.metric("ALARMES / acertos", f"{int((df['state'] == 'ALARME').sum())} / {int(((df['state'] == 'ALARME') & (df['next_maxmag'] >= alarm_mag)).sum())}",
                  help=f"alarmes emitidos / alarmes seguidos de M ≥ {alarm_mag:g}")
    st.plotly_chart(map_figure(df, profile, f"Semana de {ls:%d/%m/%Y} — probabilidade prevista e o que aconteceu "
                                            f"(círculos: eventos M ≥ {label_mag:g} dessa semana)", nxt, label_mag))
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Células com maior probabilidade")
        tv = table_view(df.assign(**{"Ocorreu?": np.where(df["y"] == 1, "sim", "não")}))
        st.dataframe(tv, hide_index=True)
    with right:
        st.subheader("Detalhe da célula")
        opts = df.sort_values("p", ascending=False)
        chosen = st.selectbox("Célula", opts["cell"].tolist(),
                              format_func=lambda c: f"{int(c)} · {opts.set_index('cell').loc[c, 'segment']} · "
                                                    f"{opts.set_index('cell').loc[c, 'lat_c']:.0f}°, {opts.set_index('cell').loc[c, 'lon_c']:.0f}°")
        ph = preds[preds["cell"] == chosen].sort_values("week")
        panel = pd.read_parquet(ROOT / "data" / "processed" / f"panel_{profile}.parquet", filters=[("cell", "==", int(chosen))])
        h = ph[["week", "label_week_start", "p"]].merge(panel[["week", "n", "nbig"]], on="week")
        h["week_start"] = pd.to_datetime(h["label_week_start"]) - pd.Timedelta(days=7)
        st.plotly_chart(history_figure(h, "Período de teste: eventos por semana e p prevista", label_mag, big_mag))

# ----------------------------------------------------------------------------- desempenho e documentação
with st.expander("Desempenho dos modelos (conjunto de teste)"):
    rdir = ROOT / "results" / profile
    md = read_md(rdir / "results_table.md")
    if md:
        st.markdown(md)
    else:
        st.info("Rode src/make_report.py para gerar a tabela.")
    hop_md = read_md(rdir / "hopfield_table.md")
    if hop_md:
        st.markdown(hop_md)
    figs = [f for f in ("fig_roc.png", "fig_pr.png", "fig_calibration.png", "fig_auc_by_year.png", "fig_map_week.png",
                        "fig_hopfield_confusion.png", "fig_hopfield_prototypes.png", f"fig_importance_{model}.png",
                        "fig_rbm_weights.png") if (rdir / f).exists()]
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]):
            col.image(str(rdir / f))

with st.expander("Como funciona, limitações e uso responsável"):
    st.markdown(f"""
**Pipeline.** Catálogo ComCat do USGS (M ≥ {summary['min_mag']:g}, desde {summary['start']}) → limpeza → grade de
{summary['cell']:g}° × {summary['cell']:g}° e semanas (segunda a domingo, UTC) → agregados por célula e vizinhos →
**rede neural** (sequência das últimas 52 semanas) → probabilidade de evento na semana seguinte →
**rede de Hopfield** (estado de alerta) → este mapa.

**Modelos.** Baselines estatísticos (persistência, climatologia, taxa recente, combinado), regressão logística,
MLP, LSTM, RBM (não supervisionada) e Hopfield (saída). Métricas no painel acima.

**Limitações.** As probabilidades refletem sobretudo *onde* a sismicidade é crônica e, em segundo lugar, as
sequências em curso (réplicas); terremotos grandes sem precursores no catálogo não têm padrão reconhecível.
O catálogo global tem completude variável no espaço e no tempo. Células de {summary['cell']:g}° são grosseiras
e a semana é um horizonte curto. Não há física de falhas no modelo.

**Uso responsável.** O SeismicAI Pulse tem finalidade acadêmica e experimental. Os resultados são estimativas
produzidas a partir dos dados e das hipóteses do projeto. Ele **não** é um sistema oficial de alerta, **não** faz
previsão determinística de terremotos e **não** substitui as instituições responsáveis pelo monitoramento sísmico.
""")
