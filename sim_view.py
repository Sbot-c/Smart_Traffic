"""
sim_view.py - animated canvas that replays a recorded episode.

The simulator records every vehicle's passage over every link as
(vehicle type, link, time entered, time reached the stop line, time departed).
The browser interpolates positions from those timestamps, stacks queued vehicles
behind the stop line, colours the signal heads from the phase log, and draws
pedestrians waiting at each junction. Two recordings with the same random seed can
be shown side by side on one clock (e.g. Fixed-time vs Q-learning).
"""
from __future__ import annotations

import json

TEMPLATE = r"""
<!doctype html>
<html><head><meta charset="utf-8">
<style>
  :root { --bg:#fcfcfb; --panel:#f2f1ec; --ink:#0b0b0b; --muted:#52514e; --line:#d9d7cf; }
  body { margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg);
         color:var(--ink); }
  .bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:6px 2px 10px 2px; }
  button { border:1px solid var(--line); background:#fff; padding:6px 14px; border-radius:6px;
           font-size:14px; cursor:pointer; }
  button.primary { background:#2a78d6; color:#fff; border-color:#2a78d6; }
  select { padding:5px 6px; border-radius:6px; border:1px solid var(--line); font-size:13px; }
  input[type=range] { flex:1; min-width:160px; accent-color:#2a78d6; }
  .clock { font-variant-numeric: tabular-nums; font-weight:600; min-width:64px; }
  .wrap { display:flex; gap:12px; flex-wrap:wrap; }
  .cell { flex:1; min-width:300px; }
  .title { font-weight:600; font-size:14px; margin:2px 0 4px 2px; }
  .stats { font-size:12px; color:var(--muted); margin:4px 2px; font-variant-numeric: tabular-nums; }
  canvas { width:100%; height:auto; background:#e9e7df; border-radius:8px; display:block; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted);
            margin-top:8px; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px;
        vertical-align:-1px; }
</style></head>
<body>
<div class="bar">
  <button id="play" class="primary">&#9654; Play</button>
  <button id="reset">&#8634; Reset</button>
  <label style="font-size:13px;color:#52514e">Speed
    <select id="speed">
      <option value="1">1x</option><option value="4">4x</option>
      <option value="10" selected>10x</option><option value="30">30x</option>
      <option value="60">60x</option>
    </select></label>
  <input id="scrub" type="range" min="0" max="900" step="1" value="0">
  <span class="clock" id="clock">00:00</span>
</div>
<div class="wrap" id="wrap"></div>
<div class="legend">
  <span><span class="sw" style="background:#cfd3d9;outline:1px solid #8a887f"></span>Car</span>
  <span><span class="sw" style="background:#19c2b4"></span>Autonomous vehicle</span>
  <span><span class="sw" style="background:#c48a45"></span>Truck</span>
  <span><span class="sw" style="background:#f5ab25"></span>Bus</span>
  <span><span class="sw" style="background:#ff4d4f"></span>Emergency vehicle</span>
  <span><span class="sw" style="background:#7b4fd6"></span>Pedestrians waiting</span>
  <span><span class="sw" style="background:#22a447"></span>/<span class="sw" style="background:#f2c14e;margin-left:4px"></span>/<span class="sw" style="background:#d64545;margin-left:4px"></span>Signal</span>
</div>
<script>
const PAYLOADS = __PAYLOADS__;
const NAMES = __NAMES__;
const HV = [[0,-1],[1,0],[0,1],[-1,0]];
const COLORS = ["#cfd3d9", "#19c2b4", "#c48a45", "#f5ab25", "#ff4d4f"];
const SIZE = 640;                 // canvas pixels (square)
const LW = 4.2;                   // lateral pixels per lane (exaggerated)
let t = 0, playing = false, speed = 10, last = null;
const tMax = Math.min(...PAYLOADS.map(p => p.t_max));
document.getElementById("scrub").max = Math.floor(tMax);

function prep(P) {
  const B = P.block, E = P.entry, W = 2 * E + 2 * B;
  const s = SIZE / W;
  const segs = Array.from({length: 36}, () => []);
  let id = 0;
  for (const g of P.segments) {
    segs[g[1]].push({vt: g[0], te: g[2], ta: g[3], td: g[4], id: id++});
  }
  const maxDur = [];
  for (let L = 0; L < 36; L++) {
    segs[L].sort((a, b) => a.te - b.te);
    let m = 0; for (const g of segs[L]) m = Math.max(m, g.td - g.te);
    maxDur.push(Math.min(m, 3600));
  }
  const streetLanes = r => (r === 1 ? 3 : 2);
  const geom = [];
  for (let L = 0; L < 36; L++) {
    const i = Math.floor(L / 4), a = L % 4, h = (a + 2) % 4;
    const r = Math.floor(i / 3), c = i % 3;
    const cx = (E + c * B) * s, cy = (E + r * B) * s;
    const d = HV[h];
    const horiz = (h === 1 || h === 3);
    const half = horiz ? 2 * LW : streetLanes(r) * LW;      // half width of crossing road
    const stop = [cx - d[0] * (half + 2), cy - d[1] * (half + 2)];
    let start;
    if (P.up[L] >= 0) {
      const ui = P.up[L], ur = Math.floor(ui / 3), uc = ui % 3;
      const ux = (E + uc * B) * s, uy = (E + ur * B) * s;
      const uhalf = horiz ? 2 * LW : streetLanes(ur) * LW;
      start = [ux + d[0] * (uhalf + 2), uy + d[1] * (uhalf + 2)];
    } else {
      start = [cx - d[0] * E * s, cy - d[1] * E * s];
    }
    const right = [-d[1], d[0]];
    geom.push({i, a, h, d, right, stop, start, lanes: P.lanes[L],
               len: Math.hypot(stop[0] - start[0], stop[1] - start[1])});
  }
  return {P, s, segs, maxDur, geom, B, E};
}

function lowerBound(arr, x) {      // first index with te >= x
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m].te < x) lo = m + 1; else hi = m; }
  return lo;
}

function signalColor(P, step, i, a) {
  const ph = P.phase[step][i], sw = P.switched[step][i];
  const tau = t - step * P.dt;
  const mine = (a % 2 === 0) ? 0 : 1;          // approaches N,S -> phase 0 ; E,W -> phase 1
  if (sw && tau < 3) return mine === 1 - ph ? "#f2c14e" : "#d64545";
  if (sw && tau < P.lost) return "#d64545";
  return mine === ph ? "#22a447" : "#d64545";
}

function drawRoads(ctx, S) {
  const {B, E, s} = S;
  ctx.fillStyle = "#e9e7df"; ctx.fillRect(0, 0, SIZE, SIZE);
  // blocks (buildings) between roads
  ctx.fillStyle = "#dcd9cf";
  const xs = [0, 1, 2].map(c => (E + c * B) * s);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    const x0 = c === 0 ? 0 : xs[c - 1] + 2 * LW + 6, x1 = c === 3 ? SIZE : xs[c] - 2 * LW - 6;
    const y0 = r === 0 ? 0 : xs[r - 1] + (r - 1 === 1 ? 3 : 2) * LW + 6;
    const y1 = r === 3 ? SIZE : xs[r] - (r === 1 ? 3 : 2) * LW - 6;
    if (x1 > x0 && y1 > y0) { ctx.beginPath(); ctx.roundRect(x0, y0, x1 - x0, y1 - y0, 6); ctx.fill(); }
  }
  ctx.fillStyle = "#4a4d52";
  for (let r = 0; r < 3; r++) {                 // streets (E-W)
    const y = (E + r * B) * s, hw = (r === 1 ? 3 : 2) * LW;
    ctx.fillRect(0, y - hw, SIZE, 2 * hw);
  }
  for (let c = 0; c < 3; c++) {                 // avenues (N-S)
    const x = (E + c * B) * s, hw = 2 * LW;
    ctx.fillRect(x - hw, 0, 2 * hw, SIZE);
  }
  if (S.P.bus_lane) {                           // red bus lanes on Street 2
    ctx.fillStyle = "rgba(200,60,50,0.55)";
    const y = (E + B) * s;
    ctx.fillRect(0, y + 2 * LW, SIZE, LW); ctx.fillRect(0, y - 3 * LW, SIZE, LW);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.setLineDash([6, 6]); ctx.lineWidth = 1;
  for (let r = 0; r < 3; r++) { const y = (E + r * B) * s;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(SIZE, y); ctx.stroke(); }
  for (let c = 0; c < 3; c++) { const x = (E + c * B) * s;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, SIZE); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.font = "600 11px -apple-system, Segoe UI, sans-serif"; ctx.fillStyle = "#33312c";
  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3), c = i % 3;
    ctx.fillText("J" + (i + 1), (E + c * B) * s + 2 * LW + 5, (E + r * B) * s - (r === 1 ? 3 : 2) * LW - 5);
  }
}

function draw(S, canvas, statsEl) {
  const ctx = canvas.getContext("2d");
  const P = S.P;
  const step = Math.min(P.phase.length - 1, Math.max(0, Math.floor(t / P.dt)));
  drawRoads(ctx, S);
  let nMoving = 0, nQueued = 0;
  // vehicles
  for (let L = 0; L < 36; L++) {
    const G = S.geom[L], segs = S.segs[L];
    const lo = lowerBound(segs, t - S.maxDur[L] - 1);
    const queued = [], moving = [];
    for (let k = lo; k < segs.length; k++) {
      const g = segs[k];
      if (g.te > t) break;
      if (t >= g.td) continue;
      if (t >= g.ta) queued.push(g); else moving.push(g);
    }
    queued.sort((a, b) => a.ta - b.ta);
    const qs = 7.5 * S.s * 1.0 + 0.6;
    const nl = G.lanes;
    let rank = 0;
    for (const g of queued) {
      let lane, row, along;
      if (g.vt === 4) { lane = -0.5; along = 3; }
      else { lane = rank % nl; row = Math.floor(rank / nl); along = row * qs + qs / 2 + 1; rank++; }
      drawVeh(ctx, G, lane, along, g, nl);
      nQueued++;
    }
    const tailRows = Math.ceil(rank / nl);
    for (const g of moving) {
      const p = (t - g.te) / Math.max(0.1, g.ta - g.te);
      let along = (1 - p) * G.len;
      const lane = g.vt === 4 ? -0.5 : (g.id % nl);
      const tail = tailRows * qs + qs / 2 + 1;
      if (g.vt !== 4 && along < tail) along = tail + (g.id % 3) * 0.8;
      drawVeh(ctx, G, lane, along, g, nl);
      nMoving++;
    }
  }
  // signals
  for (let L = 0; L < 36; L++) {
    const G = S.geom[L];
    const col = signalColor(P, step, G.i, G.a);
    ctx.strokeStyle = col; ctx.lineWidth = 3;
    const w = G.lanes * LW;
    const x0 = G.stop[0] + G.right[0] * 0.5, y0 = G.stop[1] + G.right[1] * 0.5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + G.right[0] * w, y0 + G.right[1] * w); ctx.stroke();
  }
  // pedestrians
  for (let i = 0; i < 9; i++) {
    const n = P.ped[step][i];
    if (!n) continue;
    const r = Math.floor(i / 3), c = i % 3;
    const cx = (S.E + c * S.B) * S.s, cy = (S.E + r * S.B) * S.s;
    const off = (r === 1 ? 3 : 2) * LW + 7;
    ctx.fillStyle = "#7b4fd6";
    const shown = Math.min(n, 16);
    for (let k = 0; k < shown; k++) {
      const corner = k % 4, j = Math.floor(k / 4);
      const sx = corner % 2 === 0 ? -1 : 1, sy = corner < 2 ? -1 : 1;
      ctx.beginPath();
      ctx.arc(cx + sx * (2 * LW + 6 + j * 3), cy + sy * (off), 1.8, 0, 6.283); ctx.fill();
    }
    if (n > 16) { ctx.font = "600 10px sans-serif"; ctx.fillText(String(n), cx + 2 * LW + 6, cy + off + 12); }
  }
  // incident marker
  const inc = P.incident;
  if (inc && t >= inc.t_start && t < inc.t_end) {
    const G = S.geom[inc.junction * 4 + inc.approach];
    const mx = G.stop[0] - G.d[0] * 40 + G.right[0] * LW * 2, my = G.stop[1] - G.d[1] * 40 + G.right[1] * LW * 2;
    ctx.fillStyle = "#e5383b"; ctx.beginPath(); ctx.moveTo(mx, my - 9); ctx.lineTo(mx + 8, my + 6);
    ctx.lineTo(mx - 8, my + 6); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "700 10px sans-serif"; ctx.fillText("!", mx - 2, my + 4);
  }
  if (P.event) {
    ctx.fillStyle = "rgba(123,79,214,0.85)"; ctx.font = "600 11px sans-serif";
    ctx.fillText("Arena", (S.E + 2 * S.B) * S.s + 26, (S.E + 1.5 * S.B) * S.s);
  }
  // weather overlay
  if (P.weather === "Sandstorm") { ctx.fillStyle = "rgba(214,170,100,0.28)"; ctx.fillRect(0, 0, SIZE, SIZE); }
  if (P.weather === "Rain") {
    ctx.strokeStyle = "rgba(60,110,200,0.22)"; ctx.lineWidth = 1;
    for (let k = 0; k < 140; k++) { const x = (k * 97 + t * 40) % SIZE, y = (k * 53 + t * 90) % SIZE;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 3, y + 8); ctx.stroke(); }
  }
  statsEl.textContent = `vehicles moving ${nMoving} · queued ${nQueued} · ` +
    `pedestrians waiting ${P.ped[step].reduce((a, b) => a + b, 0)}`;
}

function drawVeh(ctx, G, lane, along, g, nl) {
  const latOff = (lane + 0.5) * LW;
  const x = G.stop[0] - G.d[0] * along + G.right[0] * latOff;
  const y = G.stop[1] - G.d[1] * along + G.right[1] * latOff;
  const big = (g.vt === 2 || g.vt === 3);
  const lenPx = big ? 8.5 : 5.0, wPx = big ? 3.6 : 3.0;
  ctx.fillStyle = COLORS[g.vt];
  if (g.vt === 4 && Math.floor(t * 2) % 2 === 0) ctx.fillStyle = "#2f6fdb";
  const horiz = G.d[1] === 0;
  const w = horiz ? lenPx : wPx, h = horiz ? wPx : lenPx;
  ctx.fillRect(x - w / 2, y - h / 2, w, h);
}

const states = PAYLOADS.map(prep);
const wrap = document.getElementById("wrap");
const views = states.map((S, k) => {
  const cell = document.createElement("div"); cell.className = "cell";
  const title = document.createElement("div"); title.className = "title"; title.textContent = NAMES[k];
  const cv = document.createElement("canvas"); cv.width = SIZE; cv.height = SIZE;
  const st = document.createElement("div"); st.className = "stats";
  cell.appendChild(title); cell.appendChild(cv); cell.appendChild(st); wrap.appendChild(cell);
  return {S, cv, st};
});
function render() {
  for (const v of views) draw(v.S, v.cv, v.st);
  const m = Math.floor(t / 60), sec = Math.floor(t % 60);
  document.getElementById("clock").textContent = String(m).padStart(2, "0") + ":" + String(sec).padStart(2, "0");
  document.getElementById("scrub").value = Math.floor(t);
}
function loop(ts) {
  if (!playing) { last = null; return; }
  if (last !== null) { t += (ts - last) / 1000 * speed; }
  last = ts;
  if (t >= tMax) { t = tMax; playing = false; document.getElementById("play").innerHTML = "&#9654; Play"; }
  render();
  if (playing) requestAnimationFrame(loop);
}
document.getElementById("play").onclick = () => {
  playing = !playing;
  document.getElementById("play").innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  if (playing) { if (t >= tMax) t = 0; requestAnimationFrame(loop); }
};
document.getElementById("reset").onclick = () => { t = 0; render(); };
document.getElementById("speed").onchange = e => { speed = Number(e.target.value); };
document.getElementById("scrub").oninput = e => { t = Number(e.target.value); render(); };
t = __START__;
render();
</script></body></html>
"""


def render_html(payloads: list[dict], names: list[str], start: float = 0.0) -> str:
    slim = []
    for p in payloads:
        q = {k: p[k] for k in ("dt", "lost", "t_max", "block", "entry", "lanes", "up",
                               "segments", "phase", "switched", "ped", "incident",
                               "weather")}
        q["bus_lane"] = p.get("bus_lane", False)
        q["event"] = p.get("event", False)
        slim.append(q)
    html = TEMPLATE.replace("__PAYLOADS__", json.dumps(slim, separators=(",", ":")))
    html = html.replace("__NAMES__", json.dumps(names))
    html = html.replace("__START__", str(float(start)))
    return html
