// Renders spec.json into an IEEE-style two-column .docx that mirrors the AgroBranch paper's formatting.
const fs = require("fs");
const d = require("docx");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Header, Footer, AlignmentType, WidthType,
        BorderStyle, ShadingType, PageNumber, SectionType, TabStopType, VerticalAlign } = d;

const spec = JSON.parse(fs.readFileSync("spec.json", "utf8"));
const FONT = "Times New Roman";
const run = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 20, bold: o.bold, italics: o.italics, color: o.color,
                                           subScript: o.sub, superScript: o.sup });

// inline markup: <i>..</i> <b>..</b> <sub>..</sub> <sup>..</sup>
function runs(str, base = {}) {
  const out = []; const re = /<(\/?)(i|b|sub|sup)>/g; let st = { ...base }; let last = 0, m;
  const push = (t) => { if (t) out.push(run(t, st)); };
  while ((m = re.exec(str))) {
    push(str.slice(last, m.index)); last = m.index + m[0].length;
    const on = m[1] === ""; const k = m[2];
    if (k === "i") st.italics = on || base.italics; if (k === "b") st.bold = on || base.bold;
    if (k === "sub") st.sub = on; if (k === "sup") st.sup = on;
  }
  push(str.slice(last)); return out;
}

const body = (text) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: runs(text) });
const h1 = (text) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 260, after: 140 }, outlineLevel: 0, children: [run(text, { bold: true })] });
const h2 = (text) => new Paragraph({ keepNext: true, spacing: { before: 160, after: 100 }, outlineLevel: 1, children: [run(text, { bold: true, italics: true })] });

function figure(b) {
  const px = (inch) => Math.round(inch * 96);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(b.file), transformation: { width: px(b.w), height: px(b.h) },
        altText: { title: b.cap.slice(0, 40), description: b.cap, name: "figure" } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 }, children: [run(b.cap, { size: 16 })] }),
  ];
}

function equation(b) {
  return new Paragraph({ spacing: { before: 100, after: 160 }, tabStops: [{ type: TabStopType.CENTER, position: 2430 }, { type: TabStopType.RIGHT, position: 4860 }],
    children: [new TextRun({ text: "\t", font: FONT, size: 20 }), ...runs(b.runs, { italics: true }), new TextRun({ text: "\t", font: FONT, size: 20 }), run(b.num)] });
}

const bd = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "auto" });
function table(b) {
  const cm = { top: 40, bottom: 40, left: 60, right: 60 };
  const mk = (txt, w, hdr, bold, keep = true) => new TableCell({
    width: { size: w, type: WidthType.DXA }, margins: cm, verticalAlign: VerticalAlign.CENTER,
    borders: hdr ? { bottom: bd(12), top: bd(4), left: bd(4), right: bd(4) } : { top: bd(4), bottom: bd(4), left: bd(4), right: bd(4) },
    children: [new Paragraph({ alignment: AlignmentType.LEFT, keepNext: keep, children: [run(String(txt), { size: 15, bold: hdr || bold, color: "000000" })] })] });
  const rows = [new TableRow({ tableHeader: true, cantSplit: true, children: b.header.map((h, i) => mk(h, b.widths[i], true)) })];
  b.rows.forEach((r, ri) => rows.push(new TableRow({ cantSplit: true, children: r.map((c, i) => mk(c, b.widths[i], false, b.bold_row === ri, ri < b.rows.length - 1)) })));
  return [
    new Paragraph({ keepNext: true, spacing: { before: 60, after: 120 }, children: [run(b.title, { size: 16 })] }),
    new Table({ width: { size: 4860, type: WidthType.DXA }, columnWidths: b.widths, rows,
      borders: { top: bd(4), bottom: bd(4), left: bd(4), right: bd(4), insideHorizontal: bd(4), insideVertical: bd(4) } }),
    new Paragraph({ spacing: { before: 80, after: 200 }, alignment: AlignmentType.CENTER, children: [run(b.note, { size: 16 })] }),
  ];
}

function algo(b) {
  const top = { style: BorderStyle.SINGLE, size: 12, color: "000000" }, none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const line = (lab, txt) => {
    const lead = txt.match(/^ */)[0].length; const t = txt.slice(lead);
    return new Paragraph({ spacing: { after: 20 }, indent: { left: 360 + lead * 55, hanging: 360 }, tabStops: [{ type: TabStopType.LEFT, position: 360 + lead * 55 }],
      children: [run(lab === "Input" ? "Input:" : lab + ":", { size: 16, bold: lab === "Input" }), new TextRun({ text: "\t", font: FONT, size: 16 }), ...runs(t, { size: 16 })] });
  };
  const cell = (children, borders) => new TableCell({ width: { size: 4860, type: WidthType.DXA }, margins: { top: 50, bottom: 50, left: 80, right: 80 }, borders, children });
  return [
    new Table({ width: { size: 4860, type: WidthType.DXA }, columnWidths: [4860], borders: { top: top, bottom: top, left: none, right: none, insideHorizontal: none, insideVertical: none },
      rows: [new TableRow({ cantSplit: true, children: [cell([new Paragraph({ spacing: { after: 0 }, children: [run(b.title, { size: 17, bold: true })] })], { top, bottom: thin, left: none, right: none })] }),
             new TableRow({ cantSplit: true, children: [cell(b.lines.map(([l, t]) => line(l, t)), { top: none, bottom: top, left: none, right: none })] })] }),
    new Paragraph({ spacing: { after: 160 }, children: [] }),
  ];
}

function pageHeader() {
  return new Header({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run(spec.header, { size: 16, color: "888888" })] })] });
}
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] });

const first = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 200 }, children: [run(spec.title, { size: 32, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [run(spec.author, { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [run(spec.affil, { size: 20, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [run(spec.email, { size: 20 })] }),
];

const second = [
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: [run("Abstract—", { bold: true, italics: true }), run(spec.abstract)] }),
  new Paragraph({ spacing: { after: 200 }, children: [run("Keywords—", { bold: true, italics: true }), run(spec.keywords, { italics: true })] }),
];
for (const b of spec.blocks) {
  if (b.t === "h1") second.push(h1(b.text)); else if (b.t === "h2") second.push(h2(b.text)); else if (b.t === "p") second.push(body(b.text));
  else if (b.t === "fig") second.push(...figure(b)); else if (b.t === "eq") second.push(equation(b));
  else if (b.t === "table") second.push(...table(b)); else if (b.t === "algo") second.push(...algo(b));
}
second.push(h1("REFERENCES"));
for (const r of spec.refs) second.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 }, children: [run(r)] }));

const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 1080, bottom: 1080, left: 1080, header: 708, footer: 708 } };
const doc = new Document({
  creator: "Author", title: spec.title, description: "ScamShield IEEE-format research paper",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [
    { properties: { type: SectionType.CONTINUOUS, page, column: { count: 1 } }, children: first },
    { properties: { type: SectionType.CONTINUOUS, page, column: { count: 2, space: 360 } }, headers: { default: pageHeader() }, footers: { default: footer }, children: second },
  ],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("ScamShield_IEEE_Research_Paper.docx", buf); console.log("written", buf.length); });
