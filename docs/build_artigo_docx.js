// Gera docs/artigo.docx a partir de docs/artigo.md (subconjunto de Markdown usado no artigo).
// Uso: npm install -g docx image-size   (uma vez)   →   node docs/build_artigo_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, BorderStyle, LevelFormat, ExternalHyperlink,
} = require("docx");

const DOCS = __dirname;
const md = fs.readFileSync(path.join(DOCS, "artigo.md"), "utf8").split(/\r?\n/);
const INK = "1D3557", MUTED = "5F6B7A", LINE = "D0D7E2";
const PAGE_W = 11906, MARGIN = 1247, TEXT_W = PAGE_W - 2 * MARGIN;   // A4, margens 2,2 cm (DXA)

// ---- runs com **negrito**, *itálico* e links http(s)
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|https?:\/\/[^\s)]+)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    else if (tok.startsWith("*")) out.push(new TextRun({ text: tok.slice(1, -1), italics: true, ...base }));
    else out.push(new ExternalHyperlink({ link: tok, children: [new TextRun({ text: tok, style: "Hyperlink", ...base })] }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const children = [];
let para = [], table = null, figN = 0, refMode = false;

function flushPara() {
  if (para.length) {
    const text = para.join(" ").replace(/\s+/g, " ").trim();
    if (text) children.push(new Paragraph({ children: runs(text), spacing: { after: 140, line: 300 }, alignment: AlignmentType.JUSTIFIED }));
    para = [];
  }
}
function flushTable() {
  if (!table) return;
  const [header, ...rows] = table;
  const n = header.length;
  const widths = [Math.round(TEXT_W * 0.40), ...Array(n - 1).fill(Math.round((TEXT_W * 0.60) / (n - 1)))];
  const cell = (txt, i, isHead, bold) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, fill: "E8EEF5", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, keepNext: true,
      children: [new TextRun({ text: txt.replace(/\*\*/g, ""), bold: isHead || bold, size: 19, color: isHead ? INK : "000000" })] })],
  });
  const trs = [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, i, true, false)) })];
  for (const r of rows) {
    const bold = r[0].includes("**");
    trs.push(new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, bold)) }));
  }
  children.push(new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, rows: trs,
    borders: { top: { style: BorderStyle.SINGLE, size: 6, color: LINE }, bottom: { style: BorderStyle.SINGLE, size: 6, color: LINE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: LINE }, insideVertical: { style: BorderStyle.NONE } } }));
  children.push(new Paragraph({ children: [new TextRun({ text: "Tabela 1 — Desempenho no conjunto de teste (2021–2025). Brier: erro quadrático médio das probabilidades (menor é melhor).", italics: true, size: 18, color: MUTED })],
    spacing: { before: 80, after: 200 }, alignment: AlignmentType.CENTER }));
  table = null;
}

for (let i = 0; i < md.length; i++) {
  const line = md[i];
  if (line.startsWith("|")) {
    flushPara();
    if (/^\|[-| ]+\|$/.test(line.trim())) continue;
    const cells = line.trim().slice(1, -1).split("|").map((c) => c.trim());
    (table = table || []).push(cells);
    continue;
  } else flushTable();

  if (line.startsWith("# ")) {
    flushPara();
    children.push(new Paragraph({ children: [new TextRun({ text: line.slice(2), bold: true, size: 40, color: INK })], spacing: { after: 120 } }));
  } else if (line.startsWith("## ")) {
    flushPara(); refMode = /refer/i.test(line);
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: line.slice(3), bold: true, size: 26, color: INK })], spacing: { before: 280, after: 100 } }));
  } else if (line.startsWith("![")) {
    flushPara();
    const m = line.match(/^!\[(.*)\]\((.*)\)/);
    const file = path.join(DOCS, m[2]);
    const buf = fs.readFileSync(file);
    const { width, height } = require("image-size").imageSize ? require("image-size").imageSize(buf) : require("image-size")(buf);
    const wCm = m[2].includes("map") ? 15.5 : 11.0;
    const w = Math.round(wCm * 37.8), h = Math.round(w * height / width);
    figN++;
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: w, height: h } })] }));
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figura ${figN} — ${m[1]}.`, italics: true, size: 18, color: MUTED })] }));
  } else if (line.trim() === "---") {
    flushPara();
  } else if (line.trim() === "") {
    flushPara();
  } else if (line.startsWith("**") && i < 8) {          // subtítulo
    flushPara();
    children.push(new Paragraph({ children: [new TextRun({ text: line.replace(/\*\*/g, ""), italics: true, size: 24, color: MUTED })], spacing: { after: 120 } }));
  } else if (/^\d+\.\s/.test(line) && refMode) {          // referências
    flushPara();
    let text = line.replace(/^\d+\.\s/, "");
    while (md[i + 1] && /^\s{3}\S/.test(md[i + 1])) text += " " + md[++i].trim();
    children.push(new Paragraph({ numbering: { reference: "refs", level: 0 }, children: runs(text, { size: 19 }), spacing: { after: 60 } }));
  } else if (/^\s{3}\S/.test(line) && para.length) {
    para.push(line.trim());
  } else {
    para.push(line.trim());
  }
}
flushPara(); flushTable();

const doc = new Document({
  creator: "SeismicAI Pulse", title: "SeismicAI Pulse — artigo de divulgação",
  styles: { default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [{ id: "Hyperlink", name: "Hyperlink", basedOn: "Normal", run: { color: "1F5FBF", underline: {} } }] },
  numbering: { config: [{ reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 480, hanging: 360 } } } }] }] },
  sections: [{ properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    children }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(path.join(DOCS, "artigo.docx"), buf); console.log("artigo.docx gerado:", (buf.length / 1024).toFixed(0), "KB"); });
