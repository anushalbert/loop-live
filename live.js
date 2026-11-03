/* Loop Live: in-browser detection, tracking and short-horizon path prediction from one camera.
   Same idea as Loop's recorded-video rules (src/replay/cctv.py): footprints, swept-box time to
   collision, rough metres from each road user's typical width. Nothing leaves the device. */
'use strict';

const S = 640;                                   // detector input (YOLO11n export is fixed at 640)
const CLS = { 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck' };
const WIDTH_M = { car: 1.8, truck: 2.5, bus: 2.5, motorcycle: 0.8, bicycle: 0.6, person: 0.5 };
const GROUP = { car: 'veh', bus: 'veh', truck: 'veh', motorcycle: 'two', bicycle: 'two', person: 'ped' };
const COLOR = { veh: '#3aa0ff', two: '#25d17a', ped: '#c08cff' };
const AMBER = '#ffb020', RED = '#ff3b4e';
const CONF = 0.35, NMS_IOU = 0.5;
const FOOT = 0.25;                               // footprint = bottom quarter of the box
const HORIZON = 2.0;                             // seconds of predicted path
const RED_TTC = 1.0, RED_CLOSING = 4.2;          // < 1 s and closing > ~15 km/h
const AMBER_TTC = 2.0, AMBER_CLOSING = 2.5;
const FAST_CLOSING = 6.0;                        // > ~22 km/h: on slow devices, warn on the first look
const LOST_S = 0.8, VEL_WINDOW = 0.6;

const $ = (id) => document.getElementById(id);
const video = $('video'), view = $('view'), ctx = view.getContext('2d');
const pre = document.createElement('canvas'); pre.width = pre.height = S;
const pctx = pre.getContext('2d', { willReadFrequently: true });

let session = null, inputName = 'images', modelBytes = null;
let running = false, stream = null, fileUrl = null, isFile = false;
let tracks = [], nextId = 1, pairState = new Map(), warnings = [], lastLogged = new Map();
let frameW = 1, frameH = 1, cameraMoving = false, detTimes = [], rate = 10;
// a warning must hold for 2 looks, except on slow devices where 2 looks would come too late
const needStreak = (p) => rate >= 5 || p.closing < FAST_CLOSING ? 2 : 1;

/* ---------- detector ---------- */
// GitHub Pages cannot send the headers that let the CPU detector use several threads; a tiny
// service worker (sw.js) adds them, after one reload.
async function isolate() {
  if (self.crossOriginIsolated || !('serviceWorker' in navigator) || !window.isSecureContext) return false;
  try {
    const reg = await navigator.serviceWorker.register('sw.js');
    if (sessionStorage.getItem('coi-reloaded')) return false;          // tried once: run single-threaded
    sessionStorage.setItem('coi-reloaded', '1');
    if (!navigator.serviceWorker.controller) {
      await new Promise(r => { if (reg.active) r(); else (reg.installing || reg.waiting)?.addEventListener('statechange', e => { if (e.target.state === 'activated') r(); }); });
    }
    location.reload();
    return true;
  } catch (e) { console.warn('no service worker', e); return false; }
}

async function loadDetector() {
  ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  modelBytes = new Uint8Array(await (await fetch('model/yolo11n.onnx')).arrayBuffer());
  const tries = [];
  if (navigator.gpu && !new URLSearchParams(location.search).has('cpu')) tries.push(['webgpu', 'WebGPU (fast)']);
  tries.push(['wasm', 'CPU (slower)']);
  for (const [prov, label] of tries) {
    try {
      session = await ort.InferenceSession.create(modelBytes, { executionProviders: [prov], graphOptimizationLevel: 'all' });
      inputName = session.inputNames[0];
      // warm-up run (first WebGPU run compiles shaders)
      const blank = () => ({ [inputName]: new ort.Tensor('float32', new Float32Array(3 * S * S), [1, 3, S, S]) });
      await session.run(blank());
      const t0 = performance.now(); await session.run(blank()); const ms = performance.now() - t0;
      // some devices offer WebGPU without a real GPU behind it: fall back to the CPU when it is slow
      if (prov === 'webgpu' && ms > 400) { console.warn('WebGPU slow', ms); session = null; continue; }
      $('ep').textContent = `${label}, ${Math.round(ms)} ms per look`;
      return;
    } catch (e) { console.warn('provider failed', prov, e); session = null; }
  }
  throw new Error('No way to run the detector on this browser.');
}

function letterbox(src, w, h) {
  const r = Math.min(S / w, S / h), nw = Math.round(w * r), nh = Math.round(h * r);
  const px = Math.floor((S - nw) / 2), py = Math.floor((S - nh) / 2);
  pctx.fillStyle = 'rgb(114,114,114)'; pctx.fillRect(0, 0, S, S);
  pctx.drawImage(src, 0, 0, w, h, px, py, nw, nh);
  const d = pctx.getImageData(0, 0, S, S).data, n = S * S, f = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) { f[i] = d[j] / 255; f[n + i] = d[j + 1] / 255; f[2 * n + i] = d[j + 2] / 255; }
  return { tensor: new ort.Tensor('float32', f, [1, 3, S, S]), r, px, py };
}

function iou(a, b) {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (iw <= 0 || ih <= 0) return 0;
  const i = iw * ih;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i);
}

async function detect(w, h) {
  const { tensor, r, px, py } = letterbox(video, w, h);
  const out = (await session.run({ [inputName]: tensor }))[session.outputNames[0]];
  const d = out.data, N = out.dims[2], cand = [];
  for (let i = 0; i < N; i++) {
    let best = 0, bc = -1;
    for (const c in CLS) { const s = d[(4 + +c) * N + i]; if (s > best) { best = s; bc = +c; } }
    if (best < CONF) continue;
    const cx = d[i], cy = d[N + i], bw = d[2 * N + i], bh = d[3 * N + i];
    const box = [(cx - bw / 2 - px) / r, (cy - bh / 2 - py) / r, (cx + bw / 2 - px) / r, (cy + bh / 2 - py) / r]
      .map((v, k) => Math.max(0, Math.min(k % 2 ? h : w, v)));
    if (box[2] - box[0] < 4 || box[3] - box[1] < 4) continue;
    cand.push({ box, score: best, cls: CLS[bc] });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cand) {
    if (keep.length >= 80) break;
    if (keep.every(k => GROUP[k.cls] !== GROUP[c.cls] || iou(k.box, c.box) < NMS_IOU)) keep.push(c);
  }
  return keep;
}

/* ---------- tracking ---------- */
const foot = (b) => [b[0], b[3] - FOOT * (b[3] - b[1]), b[2], b[3]];
const center = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
const shift = (b, dx, dy) => [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy];
const mpp = (T) => WIDTH_M[T.cls] / Math.max(T.box[2] - T.box[0], 4);

function velocity(hist) {
  // last VEL_WINDOW seconds, but at least 4 looks (slow devices see only ~3 frames a second)
  const t1 = hist[hist.length - 1].t;
  let h = hist.filter(p => p.t >= t1 - VEL_WINDOW);
  if (h.length < 4) h = hist.slice(-4);
  if (h.length < 3 || t1 - h[0].t > 1.5) return [0, 0];
  const mt = h.reduce((s, p) => s + p.t, 0) / h.length, mx = h.reduce((s, p) => s + p.x, 0) / h.length,
        my = h.reduce((s, p) => s + p.y, 0) / h.length;
  let den = 0, nx = 0, ny = 0;
  for (const p of h) { const dt = p.t - mt; den += dt * dt; nx += dt * (p.x - mx); ny += dt * (p.y - my); }
  return den > 1e-6 ? [nx / den, ny / den] : [0, 0];
}

function updateTracks(dets, t, W) {
  const pred = tracks.map(T => { const dt = Math.min(t - T.lastT, 0.4); return shift(T.box, T.v[0] * dt, T.v[1] * dt); });
  const pairs = [];
  tracks.forEach((T, i) => dets.forEach((D, j) => {
    if (GROUP[T.cls] !== GROUP[D.cls]) return;
    let s = iou(pred[i], D.box);
    if (s < 0.15) {                      // small fast objects: fall back to centre distance
      // until a track has a speed, allow a bigger jump (slow devices see a fast car move a whole length)
      const [ax, ay] = center(pred[i]), [bx, by] = center(D.box);
      const gate = (T.hits < 3 ? 1.6 : 0.8) * Math.max(T.box[2] - T.box[0], T.box[3] - T.box[1]);
      const dist = Math.hypot(ax - bx, ay - by);
      if (dist < gate) s = 0.15 * (1 - dist / gate) + 0.01; else return;
    }
    pairs.push([s, i, j]);
  }));
  pairs.sort((a, b) => b[0] - a[0]);
  const usedT = new Set(), usedD = new Set();
  for (const [, i, j] of pairs) {
    if (usedT.has(i) || usedD.has(j)) continue;
    usedT.add(i); usedD.add(j);
    const T = tracks[i], D = dets[j];
    T.box = D.box; T.lastT = t; T.hits++;
    T.votes[D.cls] = (T.votes[D.cls] || 0) + 1;
    T.cls = Object.entries(T.votes).sort((a, b) => b[1] - a[1])[0][0];
    const f = center(foot(T.box));
    T.hist.push({ t, x: f[0], y: f[1] });
    while (T.hist.length && T.hist[0].t < t - 1.6) T.hist.shift();
    T.v = velocity(T.hist);
  }
  dets.forEach((D, j) => {
    if (usedD.has(j)) return;
    const f = center(foot(D.box));
    tracks.push({ id: nextId++, cls: D.cls, votes: { [D.cls]: 1 }, box: D.box, lastT: t, hits: 1, v: [0, 0], hist: [{ t, x: f[0], y: f[1] }] });
  });
  tracks = tracks.filter(T => t - T.lastT <= LOST_S);

  // camera moving: nearly every confirmed road user, parked ones included, drifts the same way
  const conf = tracks.filter(T => T.hits >= 3 && T.lastT === t);
  if (conf.length >= 4) {
    const mv = conf.filter(T => Math.hypot(T.v[0], T.v[1]) > 0.08 * W);
    const mx = mv.reduce((s, T) => s + T.v[0], 0), my = mv.reduce((s, T) => s + T.v[1], 0), mn = Math.hypot(mx, my) || 1;
    const same = mv.filter(T => (T.v[0] * mx + T.v[1] * my) / (Math.hypot(T.v[0], T.v[1]) * mn) > 0.85);
    cameraMoving = same.length >= 4 && same.length >= 0.7 * conf.length;
  } else cameraMoving = false;
}

/* ---------- prediction: when do two footprints, moving as now, start to overlap? ---------- */
function sweptTTC(a, b, va, vb, horizon) {
  const rv = [vb[0] - va[0], vb[1] - va[1]];
  let t0 = 0, t1 = horizon;
  for (const ax of [0, 1]) {
    const loA = a[ax], hiA = a[ax + 2], loB = b[ax], hiB = b[ax + 2];
    if (Math.abs(rv[ax]) < 1e-6) { if (hiB < loA || hiA < loB) return Infinity; continue; }
    let e = (loA - hiB) / rv[ax], x = (hiA - loB) / rv[ax];
    if (e > x) [e, x] = [x, e];
    t0 = Math.max(t0, e); t1 = Math.min(t1, x);
    if (t0 > t1) return Infinity;
  }
  return t0;
}

function overlapSmall(a, b) {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (iw <= 0 || ih <= 0) return 0;
  return iw * ih / Math.max(Math.min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1e-6);
}

function assess(t) {
  // boxes cut off by the image edge have false sizes and jumpy centres: shown, never warned on
  const inside = (b) => b[0] > 3 && b[1] > 3 && b[2] < frameW - 3 && b[3] < frameH - 3;
  const conf = tracks.filter(T => T.hits >= 3 && t - T.lastT < 0.25 && inside(T.box));
  const now = new Map();
  for (let i = 0; i < conf.length; i++) for (let k = i + 1; k < conf.length; k++) {
    const A = conf[i], B = conf[k];
    const g = new Set([GROUP[A.cls], GROUP[B.cls]]);
    if (g.has('ped') && g.has('two') && overlapSmall(A.box, B.box) > 0.2) continue;   // rider on a bike
    if (g.size === 1 && g.has('ped')) continue;                                          // people walking together
    const fa = foot(A.box), fb = foot(B.box);
    if (overlapSmall(fa, fb) > 0.25) continue;                 // already overlapping: occlusion, not a prediction
    const ttc = sweptTTC(fa, fb, A.v, B.v, AMBER_TTC);
    if (!isFinite(ttc)) continue;
    const ca = center(fa), cb = center(fb), d = [cb[0] - ca[0], cb[1] - ca[1]], dn = Math.hypot(d[0], d[1]) || 1;
    const closing = -(d[0] * (B.v[0] - A.v[0]) + d[1] * (B.v[1] - A.v[1])) / dn * Math.min(mpp(A), mpp(B));
    const level = ttc < RED_TTC && closing > RED_CLOSING ? 2 : closing > AMBER_CLOSING ? 1 : 0;
    if (!level) continue;
    const key = A.id < B.id ? `${A.id}-${B.id}` : `${B.id}-${A.id}`;
    const prev = pairState.get(key);
    const streak = prev && prev.level >= level ? prev.streak + 1 : 1;
    now.set(key, { a: A, b: B, ttc, closing, level, streak });
  }
  pairState = now;
  if (cameraMoving) return;
  for (const [key, p] of pairState) {
    if (p.level < 2 || p.streak < needStreak(p)) continue;
    if (t - (lastLogged.get(key) ?? -1e9) < 3) continue;
    lastLogged.set(key, t);
    warnings.unshift({ t, a: p.a.cls, b: p.b.cls, ttc: p.ttc, kmh: p.closing * 3.6 });
    renderLog();
  }
}

/* ---------- drawing ---------- */
const NAME = { car: 'car', bus: 'bus', truck: 'truck', motorcycle: 'two-wheeler', bicycle: 'bicycle', person: 'person' };
function clock(t) {
  if (isFile) { const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, '0'); return `${m}:${s} in video`; }
  return new Date().toLocaleTimeString();
}
function renderLog() {
  $('nw').textContent = warnings.length;
  $('log').innerHTML = warnings.length ? warnings.slice(0, 50).map(w =>
    `<li><span class="tag" style="background:${RED}">about to collide</span> ${NAME[w.a]} and ${NAME[w.b]} · in ${w.ttc.toFixed(1)} s · closing ~${Math.round(w.kmh)} km/h · ${clock(w.t)}</li>`).join('')
    : '<li class="none">None yet.</li>';
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function draw() {
  if (!running) return;
  requestAnimationFrame(draw);
  if (video.readyState < 2) return;
  const W = view.width, H = view.height, sx = W / video.videoWidth, sy = H / video.videoHeight;
  ctx.drawImage(video, 0, 0, W, H);
  ctx.fillStyle = 'rgba(0,12,32,0.35)'; ctx.fillRect(0, 0, W, H);
  const t = now();
  const level = new Map();
  for (const p of pairState.values()) {
    if (cameraMoving || p.streak < needStreak(p)) continue;
    for (const T of [p.a, p.b]) level.set(T.id, Math.max(level.get(T.id) || 0, p.level));
  }
  const lw = Math.max(2, W / 400);
  const twos = tracks.filter(T => T.hits >= 3 && GROUP[T.cls] === 'two');
  for (const T of tracks) {
    if (T.hits < 3) continue;
    if (GROUP[T.cls] === 'ped' && twos.some(B => overlapSmall(T.box, B.box) > 0.3)) continue;   // rider: drawn as the bike
    const dt = Math.max(0, Math.min(t - T.lastT, 0.3));
    const b = shift(T.box, T.v[0] * dt, T.v[1] * dt).map((v, k) => v * (k % 2 ? sy : sx));
    const lv = level.get(T.id) || 0, col = lv === 2 ? RED : lv === 1 ? AMBER : COLOR[GROUP[T.cls]];
    const f = foot(b), [fx, fy] = center(f), fw = f[2] - f[0];
    // predicted path: a ribbon from the footprint to where it will be in HORIZON seconds
    const ex = fx + T.v[0] * sx * HORIZON, ey = fy + T.v[1] * sy * HORIZON;
    if (Math.hypot(ex - fx, ey - fy) > 6) {
      const gr = ctx.createLinearGradient(fx, fy, ex, ey);
      gr.addColorStop(0, col + 'cc'); gr.addColorStop(1, col + '00');
      ctx.strokeStyle = gr; ctx.lineCap = 'round'; ctx.lineWidth = Math.min(Math.max(lw * 2, fw * 0.3), W / 40);
      ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(ex, ey); ctx.stroke();
    }
    ctx.shadowColor = col; ctx.shadowBlur = lv ? 22 : 10;
    ctx.fillStyle = col + (lv ? '55' : '33'); ctx.strokeStyle = col; ctx.lineWidth = lv === 2 ? lw * 2 : lw;
    roundRect(b[0], b[1], b[2] - b[0], b[3] - b[1], Math.min(10, (b[2] - b[0]) / 5)); ctx.fill(); ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.beginPath(); ctx.ellipse(fx, b[3], fw / 2, Math.max(3, fw / 10), 0, 0, Math.PI * 2);
    ctx.fillStyle = col + '55'; ctx.fill();
  }
  let red = false;
  ctx.font = `600 ${Math.max(13, W / 70)}px system-ui, sans-serif`;
  for (const p of pairState.values()) {
    if (cameraMoving || p.streak < needStreak(p)) continue;
    if (p.level === 2) red = true;
    const ca = center(foot(p.a.box)), cb = center(foot(p.b.box));
    const ax = ca[0] * sx, ay = ca[1] * sy, bx = cb[0] * sx, by = cb[1] * sy, col = p.level === 2 ? RED : AMBER;
    ctx.setLineDash([lw * 3, lw * 3]); ctx.strokeStyle = col; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); ctx.setLineDash([]);
    const label = `${p.ttc.toFixed(1)} s`, mx = (ax + bx) / 2, my = (ay + by) / 2 - 10, tw = ctx.measureText(label).width;
    ctx.fillStyle = col; roundRect(mx - tw / 2 - 6, my - 14, tw + 12, 22, 6); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.fillText(label, mx - tw / 2, my + 2);
  }
  $('banner').style.display = red ? 'block' : 'none';
  $('hold').style.display = cameraMoving ? 'block' : 'none';
  $('n').textContent = tracks.filter(T => T.hits >= 3).length;
}

const now = () => isFile ? video.currentTime : performance.now() / 1000;

async function detectLoop() {
  while (running) {
    if (video.readyState >= 2 && !video.paused && !video.ended) {
      const t = now(), w = video.videoWidth, h = video.videoHeight;
      frameW = w; frameH = h;
      const dets = await detect(w, h);
      updateTracks(dets, t, w);
      assess(t);
      const pt = t;                       // looks per second of the scene (a slowed-down video counts slowed)
      detTimes.push(pt); while (detTimes.length && detTimes[0] < pt - 3) detTimes.shift();
      if (detTimes.length > 1) { rate = (detTimes.length - 1) / (detTimes[detTimes.length - 1] - detTimes[0]); $('fps').textContent = rate.toFixed(1); }
    } else if (video.ended) {
      stop(false); return;
    }
    await new Promise(r => setTimeout(r, 0));
  }
}

/* ---------- sources ---------- */
function reset() {
  tracks = []; pairState = new Map(); warnings = []; lastLogged = new Map(); detTimes = []; renderLog();
}
async function begin() {
  await new Promise(r => { if (video.readyState >= 1) r(); else video.onloadedmetadata = r; });
  const W = Math.min(video.videoWidth, 1280);
  view.width = W; view.height = Math.round(video.videoHeight * W / video.videoWidth);
  view.hidden = false; $('empty').style.display = 'none';
  const sp = +new URLSearchParams(location.search).get('speed');
  if (isFile && sp > 0) video.playbackRate = sp;     // for testing: play a file slower, like a faster device
  await video.play();
  running = true; $('stop').disabled = false;
  try { await navigator.wakeLock?.request('screen'); } catch (e) { /* not essential */ }
  requestAnimationFrame(draw);
  detectLoop();
}
function stop(clearView = true) {
  running = false; $('stop').disabled = true;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (!clearView) return;
  video.pause();
  $('banner').style.display = 'none'; $('hold').style.display = 'none';
}

async function listCameras(selected) {
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  $('cam').innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${c.label || 'Camera ' + (i + 1)}</option>`).join('')
    || '<option value="">No camera found</option>';
  if (selected) $('cam').value = selected;
}

async function startCamera() {
  if (!session) return;
  stop(); reset(); isFile = false;
  if (fileUrl) { URL.revokeObjectURL(fileUrl); fileUrl = null; }
  const id = $('cam').value;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: id
      ? { deviceId: { exact: id }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } });
  } catch (e) {
    $('empty').style.display = 'block'; $('empty').textContent = 'The camera could not be opened: ' + e.message + '. Allow camera access for this page and try again.';
    return;
  }
  video.srcObject = stream; video.loop = false;
  await listCameras(stream.getVideoTracks()[0].getSettings().deviceId);
  await begin();
}

async function startFile(f) {
  if (!session || !f) return;
  stop(); reset(); isFile = true;
  if (fileUrl) URL.revokeObjectURL(fileUrl);
  fileUrl = URL.createObjectURL(f);
  video.srcObject = null; video.src = fileUrl; video.loop = false;
  await begin();
}

$('startCam').onclick = startCamera;
$('stop').onclick = () => stop(true);
$('file').onchange = (e) => startFile(e.target.files[0]);
$('cam').onchange = () => { if (running && !isFile) startCamera(); };

(async () => {
  if (await isolate()) return;
  $('startCam').disabled = true;
  try {
    await loadDetector();
    $('empty').textContent = 'Ready. Start the camera, or use a video file of a road.';
    $('startCam').disabled = false;
    if (navigator.mediaDevices?.enumerateDevices) listCameras().catch(() => {});
  } catch (e) {
    $('empty').textContent = 'The detector could not load: ' + e.message;
    $('ep').textContent = 'failed';
  }
})();
window.__live = { get tracks() { return tracks; }, get warnings() { return warnings; }, get pairs() { return pairState; } };
