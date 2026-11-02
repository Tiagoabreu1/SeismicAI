// SeismicAI Pulse — site estático. Lê meta/latest e weeks/{id} do Cloud Firestore (ou de data/*.json com
// ?source=local, para testar sem Firebase) e desenha as células no mapa.

const params = new URLSearchParams(location.search);
const cfg = window.FIREBASE_CONFIG || {};
const useLocal = params.get("source") === "local" || !cfg.projectId || cfg.projectId.startsWith("SEU");
const FB = "https://www.gstatic.com/firebasejs/10.14.1";

// ---------------------------------------------------------------- fonte de dados
let fs = null;                                             // { db, doc, getDoc }, carregado só quando há Firebase
async function firestore() {
  if (!fs) {
    const [{ initializeApp }, { getFirestore, doc, getDoc }] = await Promise.all([
      import(`${FB}/firebase-app.js`), import(`${FB}/firebase-firestore.js`)]);
    fs = { db: getFirestore(initializeApp(cfg)), doc, getDoc };
  }
  return fs;
}

async function readDoc(collection, id) {
  if (useLocal) {
    const path = collection === "meta" ? "data/meta.json" : `data/weeks/${id}.json`;
    const r = await fetch(path, { cache: "no-store" });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
  }
  const { db, doc, getDoc } = await firestore();
  const snap = await getDoc(doc(db, collection, id));
  if (!snap.exists()) throw new Error(`${collection}/${id} não encontrado`);
  return snap.data();
}

// ---------------------------------------------------------------- mapa
// Fundo vetorial próprio (países do Natural Earth, 110 m): sem tiles externos, sem chave de API, modo escuro nativo.
const map = L.map("map", { worldCopyJump: false, minZoom: 2, maxZoom: 8, zoomControl: true,
  maxBounds: [[-85, 10], [85, 350]], maxBoundsViscosity: 0.8 }).setView([5, 180], 3);
L.control.attribution({ prefix: false }).addAttribution(
  'Dados: <a href="https://earthquake.usgs.gov/">USGS</a> · Mapa: <a href="https://www.naturalearthdata.com/">Natural Earth</a> · <a href="https://leafletjs.com/">Leaflet</a>').addTo(map);
const landLayer = L.layerGroup().addTo(map);
const cellLayer = L.layerGroup().addTo(map);
const eventLayer = L.layerGroup().addTo(map);

// Pacífico contíguo: longitudes negativas (Américas) desenhadas a leste de 180°
const shiftLon = (lon) => (lon < 0 ? lon + 360 : lon);

// Mesma ideia para os polígonos dos países: anel que cruza o antimeridiano -> só os vértices negativos
// ganham +360; anel inteiro no hemisfério oeste -> tudo +360; anel que cruza o meridiano 0 -> fica como está.
function shiftRing(ring) {
  const lons = ring.map((c) => c[0]);
  const min = Math.min(...lons), max = Math.max(...lons);
  if (max - min > 180) return ring.map(([x, y]) => [x < 0 ? x + 360 : x, y]);
  if (max <= 0) return ring.map(([x, y]) => [x + 360, y]);
  return ring;
}
function shiftGeometry(geom) {
  if (geom.type === "Polygon") return { type: "Polygon", coordinates: geom.coordinates.map(shiftRing) };
  if (geom.type === "MultiPolygon") return { type: "MultiPolygon", coordinates: geom.coordinates.map((poly) => poly.map(shiftRing)) };
  return geom;
}
async function drawLand() {
  try {
    const r = await fetch("countries.json", { cache: "force-cache" });
    const gj = await r.json();
    gj.features = gj.features.map((f) => ({ ...f, geometry: shiftGeometry(f.geometry) }));
    L.geoJSON(gj, { style: { color: "#3a4a62", weight: 0.6, fillColor: "#1b2433", fillOpacity: 1 }, interactive: false }).addTo(landLayer);
  } catch (err) {
    console.warn("sem mapa de fundo:", err.message);
  }
}
drawLand();

const STOPS = [[0, [255, 255, 204]], [0.15, [254, 217, 118]], [0.35, [253, 141, 60]], [0.65, [227, 26, 28]], [1, [128, 0, 38]]];
function colorFor(p, pmax) {
  const t = Math.max(0, Math.min(1, p / pmax));
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i], f = (t - t0) / (t1 - t0);
      const c = c0.map((v, k) => Math.round(v + f * (c1[k] - v)));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(128,0,38)";
}
const STATE_STYLE = {
  ALARME: { color: "#c1121f", weight: 2.5 },
  "ATENÇÃO": { color: "#f0a202", weight: 1.6 },
  NORMAL: { color: "#6b7a90", weight: 0.4 },
};

const fmtPct = (p) => `${(100 * p).toFixed(p < 0.1 ? 1 : 0)} %`;
const fmtDate = (iso) => { const [y, m, d] = iso.split("-"); return `${d}/${m}/${y}`; };

let meta = null, current = null;

function drawWeek(week) {
  current = week;
  cellLayer.clearLayers();
  eventLayer.clearLayers();
  const half = (meta.cell_deg || 2) / 2;
  const pmax = Math.max(0.7, week.stats.p_max || 0);
  const hist = week.kind === "historico";
  const sorted = [...week.cells].sort((a, b) => b.p - a.p);

  for (const c of sorted.slice().reverse()) {               // menos prováveis primeiro: as fortes ficam por cima
    const lon = shiftLon(c.lon);
    const st = STATE_STYLE[c.st] || STATE_STYLE.NORMAL;
    const t = Math.min(1, c.p / pmax);
    const rect = L.rectangle([[c.lat - half, lon - half], [c.lat + half, lon + half]], {
      color: st.color, weight: st.weight, fillColor: colorFor(c.p, pmax), fillOpacity: 0.35 + 0.6 * t,
    });
    rect.bindTooltip(
      `<b>${c.seg || "—"}</b><br>célula ${c.c} · ${c.lat}°, ${c.lon}°<br>p: <b>${fmtPct(c.p)}</b>${c.st ? ` · ${c.st}` : ""}` +
      (hist ? `<br>ocorreu: <b>${c.y ? "sim" : "não"}</b>${c.mx ? ` (M ${c.mx.toFixed(1)})` : ""}` : ""),
      { className: "cell-tip", sticky: true });
    rect.on("click", () => showDetail(c, hist));
    rect.addTo(cellLayer);
  }
  for (const e of week.events || []) {
    L.circleMarker([e.lat, shiftLon(e.lon)], {
      radius: 3 + 2.2 * Math.max(0, e.mag - week.label_mag) ** 1.4, color: "#7cc7ff", weight: 1.3, fill: false,
    }).bindTooltip(`M ${e.mag.toFixed(1)} · ${e.t.replace("T", " ").replace("Z", " UTC")}`, { className: "cell-tip" })
      .addTo(eventLayer);
  }
  eventLayer[document.getElementById("show-events").checked ? "addTo" : "remove"](map);

  // KPIs
  const s = week.stats;
  document.getElementById("kpi-week").textContent = fmtDate(week.label_week_start);
  document.getElementById("kpi-cells").textContent = s.n_cells;
  document.getElementById("kpi-alarme").textContent = s.n_alarme ?? "—";
  document.getElementById("kpi-atencao").textContent = s.n_atencao ?? "—";
  if (hist) {
    document.getElementById("kpi-extra-label").textContent = "acertos no top 5 %";
    document.getElementById("kpi-extra").textContent = s.top5_precision != null ? fmtPct(s.top5_precision) : "—";
    document.getElementById("kpi-extra").title = `${s.n_hits} células tiveram evento; alarmes: ${s.alarms ?? "—"}, seguidos de M ≥ ${week.alarm_mag}: ${s.alarm_hits ?? "—"}`;
  } else {
    document.getElementById("kpi-extra-label").textContent = "p máxima";
    document.getElementById("kpi-extra").textContent = fmtPct(s.p_max);
    document.getElementById("kpi-extra").title = week.catalog_end ? `catálogo até ${week.catalog_end.replace("T", " ")}` : "";
  }

  // ranking
  const ol = document.getElementById("ranking");
  ol.innerHTML = "";
  for (const c of sorted.slice(0, 12)) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="name" title="${c.seg || ""}">${c.seg || "célula " + c.c} · ${c.lat}°, ${c.lon}°</span>` +
      `<span class="p">${c.st === "ALARME" ? '<span class="dot alarme"></span>' : c.st === "ATENÇÃO" ? '<span class="dot atencao"></span>' : ""}${fmtPct(c.p)}</span>`;
    li.onclick = () => { showDetail(c, hist); map.panTo([c.lat, shiftLon(c.lon)]); };
    ol.appendChild(li);
  }
  showDetail(sorted[0], hist);
}

function showDetail(c, hist) {
  const el = document.getElementById("detail");
  const rows = [
    ["Probabilidade (semana seguinte)", `<span class="big">${fmtPct(c.p)}</span>`],
    ["Estado (Hopfield)", c.st ? `<span class="state ${c.st}">${c.st}</span>` : "—"],
    ["Eventos na última semana", c.n1], ["Eventos nas últimas 4 semanas", c.n4],
    ["Magnitude máxima (4 sem.)", c.mx4 ? c.mx4.toFixed(1) : "—"], [`Eventos M ≥ ${current.alarm_mag} (4 sem.)`, c.nb4],
    ["Eventos nos vizinhos (1 sem.)", c.nv1],
  ];
  if (hist) rows.push(["O que aconteceu na semana seguinte", c.y ? `evento (M máx ${c.mx ? c.mx.toFixed(1) : "?"})` : "nenhum evento"]);
  el.innerHTML = `<h2>Célula ${c.c}</h2><p><b>${c.seg || "fora dos segmentos"}</b> · ${c.lat}°, ${c.lon}°</p>` +
    `<table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>`;
}

// ---------------------------------------------------------------- inicialização
async function init() {
  const badge = document.getElementById("source-badge");
  badge.textContent = useLocal ? "dados locais" : "Firestore";
  try {
    meta = await readDoc("meta", "latest");
  } catch (err) {
    badge.textContent = "erro ao carregar";
    document.getElementById("detail").innerHTML = `<h2>Erro</h2><p class="muted">${err.message}</p>`;
    return;
  }
  for (const el of document.querySelectorAll(".alarm-mag")) el.textContent = meta.alarm_mag;
  for (const el of document.querySelectorAll(".label-mag")) el.textContent = meta.label_mag;
  for (const el of document.querySelectorAll(".cell-deg")) el.textContent = meta.cell_deg;
  if (meta.metrics && meta.metrics.roc_auc) {
    document.getElementById("metrics-line").textContent =
      `Modelo ${meta.model.toUpperCase()} no conjunto de teste (2021–2025): ROC-AUC ${meta.metrics.roc_auc.toFixed(3)}, ` +
      `PR-AUC ${meta.metrics.pr_auc.toFixed(3)}, Brier ${meta.metrics.brier.toFixed(4)}; ` +
      `entre as 5 % células mais prováveis de cada semana, ${fmtPct(meta.metrics.precision_top5)} têm evento. Atualizado em ${meta.updated_at.slice(0, 16).replace("T", " ")} UTC.`;
  }
  const sel = document.getElementById("week-select");
  sel.innerHTML = "";
  for (const w of meta.weeks) {
    const o = document.createElement("option");
    o.value = w.id;
    o.textContent = `${fmtDate(w.label_week_start)} · ${w.kind === "ao_vivo" ? "ao vivo" : "teste"}${w.n_alarme != null ? ` · ${w.n_alarme} alarmes` : ""}`;
    sel.appendChild(o);
  }
  sel.value = meta.latest || meta.weeks[0].id;
  sel.onchange = async () => drawWeek(await readDoc("weeks", sel.value));
  document.getElementById("show-events").onchange = (e) => eventLayer[e.target.checked ? "addTo" : "remove"](map);
  drawWeek(await readDoc("weeks", sel.value));
}

init();
