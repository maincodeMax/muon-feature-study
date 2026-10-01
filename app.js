// Where Muon's Advantage Lives — figures and the arm bench.
// Every number is from the MOSS camera-ready (symmetrized pair scores, three seeds per arm).
"use strict";

// Maincode palette: gold = Muon, neutral gray = AdamW, off-white = Muon vs AdamW.
const C = { muon: "#e2b53d", adam: "#a8a8a8", cross: "#f2f2f2", ink: "#ededed", ink2: "#a1a1a1", rule: "#2e2e2e", paper: "#0a0a0a" };

// ── data ────────────────────────────────────────────────────────────────
const ARMS = {
  "000": { name: "full AdamW", loss: 3.350, rep: 0.593, qk: 6.7, ov: 31.5,
           verdict: "The baseline: AdamW on every matrix. Hand a circuit to Muon to see what it buys." },
  "100": { name: "muon-QK", loss: 3.360, rep: 0.588, qk: 21.0, ov: 28.7,
           verdict: "Scoring circuit only. Its stable rank triples, yet loss and reproducibility don’t improve. The rank change is a passenger." },
  "010": { name: "muon-VO", loss: 3.301, rep: 0.632, qk: 5.5, ov: 76.2,
           verdict: "Value circuit only. 86% of the loss gap closes, and the features are the most reproducible of any arm." },
  "110": { name: "muon-attn", loss: 3.299, rep: 0.630, qk: 21.3, ov: 76.6,
           verdict: "All of attention. About the same as the value circuit alone: adding W_Q W_K contributes little." },
  "001": { name: "muon-MLP", loss: 3.368, rep: 0.603, qk: null, ov: null,
           verdict: "MLP only. No loss gain at all (it ends slightly behind AdamW), and only part of the reproducibility gap." },
  "101": { name: "complement", loss: 3.369, rep: 0.606, qk: null, ov: null,
           verdict: "Muon everywhere except the value circuit. The whole loss gain disappears, so the value circuit is necessary for it." },
  "111": { name: "full Muon", loss: 3.292, rep: 0.618, qk: null, ov: 64.7,
           verdict: "Muon on every hidden matrix: the full gain." },
  "011": null,
};

// ── tiny svg helpers ────────────────────────────────────────────────────
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;
const line = (x1, y1, x2, y2, col, wd = 1, dash = "") =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${wd}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const text = (x, y, s, o = {}) =>
  `<text x="${x}" y="${y}" font-size="${o.size || 12}" text-anchor="${o.anchor || "start"}"${o.fill ? ` style="fill:${o.fill}"` : ""}${o.cls ? ` class="${o.cls}"` : ""}${o.base ? ` dominant-baseline="${o.base}"` : ""}>${esc(s)}</text>`;
const dot = (x, y, r, col) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${col}" stroke="${C.paper}" stroke-width="2"/>`;
const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const small = (w) => w < 460;

// ── figures ─────────────────────────────────────────────────────────────
const CHARTS = {
  toy(w) {
    const toks = ["The", "cat", "sat", "on", "a", "mat", ".", "A", "dog", "ran", "."];
    const rows = [
      ["A · feature i", C.muon, [0, .9, 0, 0, 0, .8, 0, 0, .95, 0, .3], ""],
      ["B · feature j₁", C.cross, [0, .85, 0, 0, 0, .7, 0, 0, 1, 0, .4], "ρ = 0.93"],
      ["B · feature j₂", C.cross, [.6, 0, 0, 0, .7, 0, 0, .5, 0, 0, 0], "ρ = 0.01"],
      ["B · feature j₃", C.cross, [0, 0, .5, 0, 0, 0, 0, 0, 0, .9, 0], "ρ = −0.10"],
    ];
    const sm = small(w), lab = sm ? 46 : 118, rho = sm ? 58 : 78, fs = sm ? 9.5 : 12;
    if (sm) rows.forEach((r) => (r[0] = r[0].replace(" · feature ", " · ")));
    const cw = (w - lab - rho) / toks.length, rh = 30, top = 26, h = top + rows.length * rh + 6;
    let b = "";
    toks.forEach((t, i) => (b += text(lab + cw * (i + 0.5), 14, t, { size: fs, anchor: "middle" })));
    rows.forEach(([name, col, acts, r], k) => {
      const y = top + k * rh;
      if (k === 1) b += `<rect x="0" y="${y}" width="${w}" height="${rh}" fill="${C.muon}" opacity="0.08"/>`;
      b += text(lab - 10, y + rh / 2, name, { size: fs, anchor: "end", base: "middle", fill: col });
      acts.forEach((a, i) => {
        const bh = Math.max(a, 0.04) * (rh - 10);
        b += `<rect x="${lab + cw * i + cw * 0.18}" y="${y + rh - 4 - bh}" width="${cw * 0.64}" height="${bh}" fill="${col}" opacity="${a > 0.05 ? 1 : 0.25}"/>`;
      });
      if (r) b += text(w, y + rh / 2, r, { size: fs, anchor: "end", base: "middle", fill: k === 1 ? C.muon : C.ink2, cls: k === 1 ? "t-val" : "" });
    });
    return svg(w, h, b);
  },

  dots(w) {
    const groups = [
      ["Muon vs Muon", C.muon, [0.6178, 0.6200, 0.6177], 0.618],
      ["Muon vs AdamW", C.cross, [0.5926, 0.5900, 0.5949, 0.5891, 0.5890, 0.5903, 0.5943, 0.5903, 0.5947], 0.592],
      ["AdamW vs AdamW", C.adam, [0.5901, 0.5959, 0.5926], 0.593],
    ];
    const h = 250, L = 46, R = 8, T = 28, B = 40, fs = small(w) ? 10.5 : 12;
    const y = scale(0.584, 0.628, h - B, T), gw = (w - L - R) / 3;
    let b = "";
    [0.59, 0.60, 0.61, 0.62].forEach((t) => (b += line(L, y(t), w - R, y(t), C.rule) + text(L - 8, y(t), t.toFixed(2), { size: fs, anchor: "end", base: "middle" })));
    b += line(L, y(0.593), w - R, y(0.593), C.adam, 1.2, "5 4");
    b += text(L + 6, y(0.593) + 16, "AdamW seed ceiling", { size: fs - 1, fill: C.adam });
    groups.forEach(([name, col, vals, mean], g) => {
      const cx = L + gw * (g + 0.5), span = gw * 0.5;
      b += line(cx - span / 2 - 6, y(mean), cx + span / 2 + 6, y(mean), col, 3);
      vals.forEach((v, i) => (b += dot(cx - span / 2 + (span * i) / Math.max(1, vals.length - 1), y(v), small(w) ? 4.5 : 5.5, col)));
      b += text(cx, y(Math.max(...vals)) - 12, mean.toFixed(3), { size: fs + 4, anchor: "middle", fill: col, cls: "t-val" });
      if (small(w)) {
        const [a, c] = name.split(" vs ");
        b += text(cx, h - 22, a, { size: fs, anchor: "middle", fill: col }) + text(cx, h - 8, "vs " + c, { size: fs, anchor: "middle", fill: col });
      } else b += text(cx, h - 12, name, { size: fs, anchor: "middle", fill: col });
    });
    return svg(w, h, b);
  },

  early(w) {
    const steps = [500, 1000, 2000, 3000, 4000, 5100];
    const mu = [0.629, 0.634, 0.629, 0.623, 0.619, 0.618], ad = [0.602, 0.616, 0.604, 0.595, 0.590, 0.593];
    const h = 210, L = 46, R = small(w) ? 54 : 64, T = 30, B = 34, fs = small(w) ? 10.5 : 12;
    const x = scale(0, 5100, L, w - R), y = scale(0.585, 0.642, h - B, T);
    let b = `<rect x="${x(0)}" y="${T - 6}" width="${x(500) - x(0)}" height="${h - B - T + 6}" fill="${C.muon}" opacity="0.08"/>`;
    [0.59, 0.61, 0.63].forEach((t) => (b += line(L, y(t), w - R, y(t), C.rule) + text(L - 8, y(t), t.toFixed(2), { size: fs, anchor: "end", base: "middle" })));
    [500, 2000, 4000, 5100].forEach((s) => (b += text(x(s), h - 12, s.toLocaleString("en"), { size: fs, anchor: "middle" })));
    const path = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(steps[i])},${y(v)}`).join("");
    b += `<path d="${path(ad)}" fill="none" stroke="${C.adam}" stroke-width="2.4"/><path d="${path(mu)}" fill="none" stroke="${C.muon}" stroke-width="2.4"/>`;
    steps.forEach((s, i) => (b += dot(x(s), y(ad[i]), 4.2, C.adam) + dot(x(s), y(mu[i]), 4.2, C.muon)));
    b += text(x(5100) + 10, y(0.618) + 1, "Muon", { size: fs, base: "middle", fill: C.muon });
    b += text(x(5100) + 10, y(0.593), "AdamW", { size: fs, base: "middle", fill: C.adam });
    b += `<path d="M${x(5100)},${y(0.629) - 7} l2.1,4.6 5,.5 -3.8,3.3 1.1,4.9 -4.4,-2.6 -4.4,2.6 1.1,-4.9 -3.8,-3.3 5,-.5z" fill="${C.muon}"/>`;
    b += text(x(5100) - 12, y(0.629) - 12, small(w) ? "switch run: 0.629" : "500 Muon steps, then AdamW: 0.629", { size: fs - 0.5, anchor: "end", fill: C.muon });
    b += text(x(250), T - 12, "≤ 500", { size: fs - 1, anchor: "middle", fill: C.muon });
    return svg(w, h, b);
  },

  rankreg(w) {
    const lam = ["0", "1e-4", "1e-3", "1e-2", "3e-2"], rank = [31.5, 50.4, 63.5, 61.2, 64.0], loss = [3.350, 3.350, 3.382, 3.476, 3.549];
    const stack = small(w), gap = stack ? 0 : 26, pw = stack ? w : (w - gap) / 2, ph = 180, T = 28, B = 30, L = 36, fs = 12.5;
    const h = stack ? ph * 2 + 18 : ph;
    const panel = (ox, oy, vals, d0, d1, ref, title, ticks, fmt) => {
      const x = (i) => ox + L + (i * (pw - L - 8)) / 4, y = scale(d0, d1, oy + ph - B, oy + T);
      let b = text(ox, oy + 12, title, { size: fs + 0.5, fill: C.ink });
      ticks.forEach((t) => (b += line(ox + L, y(t), ox + pw, y(t), C.rule) + text(ox + L - 6, y(t), fmt(t), { size: fs - 1, anchor: "end", base: "middle" })));
      b += line(ox + L, y(ref), ox + pw, y(ref), C.muon, 1.2, "4 3") + text(ox + pw, y(ref) - 5, "Muon", { size: fs - 1, anchor: "end", fill: C.muon });
      b += `<path d="${vals.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("")}" fill="none" stroke="${C.adam}" stroke-width="2.2"/>`;
      vals.forEach((v, i) => (b += dot(x(i), y(v), 3.8, C.adam)));
      lam.forEach((l, i) => (b += text(x(i), oy + ph - 10, l, { size: fs - 1.5, anchor: "middle" })));
      return b;
    };
    return svg(w, h,
      panel(0, 0, rank, 20, 72, 64.7, "value-circuit rank", [30, 50, 70], (t) => t) +
      panel(stack ? 0 : pw + gap, stack ? ph + 18 : 0, loss, 3.26, 3.57, 3.292, "final val loss", [3.3, 3.4, 3.5], (t) => t.toFixed(1)));
  },

  layers(w) {
    const rows = [["L6 → L6", 0.618, 0.590], ["L6 → L3", 0.590, 0.565], ["L6 → L9", 0.570, 0.552]];
    const h = 168, L = 70, R = 18, T = 26, rh = 40, fs = 12.5;
    const x = scale(0.545, 0.625, L, w - R);
    let b = "";
    rows.forEach(([n, m, a], k) => {
      const y = T + 10 + k * rh;
      b += text(0, y, n, { size: fs, base: "middle", fill: C.ink });
      b += line(x(a), y, x(m), y, C.rule, 5);
      b += dot(x(a), y, 5, C.adam) + dot(x(m), y, 5, C.muon);
      b += text(x(a), y - 11, a.toFixed(3), { size: fs - 1, anchor: "middle", fill: C.adam });
      b += text(x(m), y - 11, m.toFixed(3), { size: fs - 1, anchor: "middle", fill: C.muon });
    });
    b += line(L, h - 26, w - R, h - 26, C.ink2, 0.8);
    [0.55, 0.57, 0.59, 0.61].forEach((t) => (b += text(x(t), h - 8, t.toFixed(2), { size: fs - 1, anchor: "middle" })));
    return svg(w, h, b);
  },
};

function drawCharts() {
  document.querySelectorAll(".chart[data-chart]").forEach((el) => {
    const w = Math.round(el.clientWidth);
    if (!w || el.dataset.w === String(w)) return;
    el.dataset.w = w;
    el.innerHTML = CHARTS[el.dataset.chart](w);
  });
}

// ── the arm bench ───────────────────────────────────────────────────────
const LOSS = scale(3.380, 3.280, 0, 100);   // reversed: lower loss sits to the right
const REP = scale(0.580, 0.645, 0, 100);

function initBench() {
  const bench = document.getElementById("bench");
  if (!bench) return;
  const state = { qk: false, vo: false, mlp: false };
  const stamps = [...bench.querySelectorAll(".stamp")];
  const $ = (id) => document.getElementById(id);

  bench.querySelectorAll('[data-scale="loss"] .ref').forEach((r) => (r.style.left = LOSS(+r.dataset.ref) + "%"));
  bench.querySelectorAll('[data-scale="rep"] .ref').forEach((r) => (r.style.left = REP(+r.dataset.ref) + "%"));
  const riders = { loss: bench.querySelector('[data-scale="loss"] .rider'), rep: bench.querySelector('[data-scale="rep"] .rider') };

  function render() {
    const key = `${+state.qk}${+state.vo}${+state.mlp}`, arm = ARMS[key];
    stamps.forEach((s) => s.setAttribute("aria-pressed", String(state[s.dataset.k])));
    if (!arm) {
      $("arm-name").textContent = "not run";
      $("arm-sub").textContent = "";
      $("verdict").textContent = "We never trained Muon on the value circuit and MLP with AdamW on scoring. Every other combination is a real arm.";
      $("ranks").textContent = "";
      Object.values(riders).forEach((r) => (r.style.opacity = "0"));
      return;
    }
    const pin = state.vo ? C.muon : key === "000" ? C.adam : C.cross;
    for (const [k, r] of Object.entries(riders)) {
      r.style.opacity = "1";
      r.style.setProperty("--x", (k === "loss" ? LOSS(arm.loss) : REP(arm.rep)).toFixed(2) + "%");
      r.style.setProperty("--pin", pin);
    }
    $("loss-val").textContent = arm.loss.toFixed(3);
    $("rep-val").textContent = arm.rep.toFixed(3);
    $("arm-name").textContent = arm.name;
    $("arm-sub").textContent = "· three seeds";
    $("verdict").textContent = arm.verdict;
    const f = (v) => (v == null ? "n/m" : v.toFixed(1));
    $("ranks").textContent = arm.qk == null && arm.ov == null
      ? "circuit stable ranks not measured for this arm"
      : `circuit stable rank · scoring ${f(arm.qk)} · value ${f(arm.ov)}`;
  }

  stamps.forEach((s) => s.addEventListener("click", () => { state[s.dataset.k] = !state[s.dataset.k]; render(); }));
  bench.querySelectorAll(".preset").forEach((p) => p.addEventListener("click", () => {
    const [q, v, m] = p.dataset.preset.split("").map((c) => c === "1");
    Object.assign(state, { qk: q, vo: v, mlp: m });
    render();
  }));
  render();
}

// ── citation copy ───────────────────────────────────────────────────────
function initCopy() {
  const btn = document.getElementById("copy");
  if (!btn || !navigator.clipboard) { if (btn) btn.hidden = true; return; }
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(document.getElementById("bibtex").textContent);
      btn.classList.add("done");
      setTimeout(() => btn.classList.remove("done"), 1600);
    } catch { /* clipboard blocked: the text is still selectable */ }
  });
}

drawCharts();
initBench();
initCopy();
let rt;
window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(drawCharts, 120); });
if (document.fonts) document.fonts.ready.then(() => { document.querySelectorAll(".chart").forEach((e) => delete e.dataset.w); drawCharts(); });
