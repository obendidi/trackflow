import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';
import { BrowserByteTracker } from './tracker.js';

env.allowLocalModels = false;

const PALETTE = ['#34d399', '#38bdf8', '#a78bfa', '#fbbf24', '#f472b6', '#60a5fa', '#4ade80'];
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fpsVal');
const activeEl = document.getElementById('activeVal');
const uniqueEl = document.getElementById('uniqueVal');
const deviceEl = document.getElementById('deviceVal');

const modelSelect = document.getElementById('modelSelect');
const privacySelect = document.getElementById('privacySelect');
const peopleOnlyChk = document.getElementById('peopleOnly');
const threshSlider = document.getElementById('threshSlider');
const threshVal = document.getElementById('threshVal');

let detector = null;
let loadedModelId = null;
let running = false;
let tracker = new BrowserByteTracker();
let frameIdx = 0;
let records = [];

threshSlider.addEventListener('input', () => {
  threshVal.textContent = Number(threshSlider.value).toFixed(2);
  tracker.highThresh = Number(threshSlider.value);
});

async function ensureDetector() {
  const modelId = modelSelect.value;
  if (detector && loadedModelId === modelId) return detector;

  const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
  const device = hasWebGPU ? 'webgpu' : 'wasm';
  deviceEl.textContent = device.toUpperCase();
  statusEl.textContent = `Loading ${modelId} on ${device.toUpperCase()} (cached in browser after first download)…`;

  try {
    detector = await pipeline('object-detection', modelId, {
      device,
      dtype: device === 'webgpu' ? 'fp32' : 'q8',
    });
  } catch (err) {
    deviceEl.textContent = 'WASM';
    statusEl.textContent = `WebGPU unavailable (${err.message}); falling back to WASM…`;
    detector = await pipeline('object-detection', modelId, { device: 'wasm', dtype: 'q8' });
  }
  loadedModelId = modelId;
  statusEl.textContent = `Ready · ${modelId} running 100% locally in your browser.`;
  return detector;
}

function drawPrivacyBackdrop(tracks, mode) {
  if (mode === 'skeleton') {
    ctx.fillStyle = '#0e1320';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#1b243b';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 48) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 48) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }
    return;
  }

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (mode === 'none') return;

  for (const tr of tracks) {
    const { xmin, ymin, xmax, ymax } = tr.box;
    const w = Math.max(4, xmax - xmin);
    const h = Math.max(4, ymax - ymin);
    ctx.save();
    ctx.filter = 'blur(16px)';
    if (mode === 'face-blur') {
      const fh = Math.max(8, h * 0.3);
      ctx.beginPath();
      ctx.ellipse(xmin + w * 0.5, ymin + fh * 0.5, w * 0.42, fh * 0.5, 0, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.beginPath();
      ctx.rect(xmin, ymin, w, h);
      ctx.clip();
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }
    ctx.restore();
  }
}

function drawSkeleton(tr, color) {
  const { xmin, ymin, xmax, ymax } = tr.box;
  const bw = xmax - xmin, bh = ymax - ymin, cx = xmin + bw * 0.5;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.arc(cx, ymin + bh * 0.12, Math.max(4, Math.min(bw * 0.18, bh * 0.08)), 0, Math.PI * 2);
  ctx.stroke();

  const seg = (x1, y1, x2, y2) => {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  };
  seg(cx, ymin + bh * 0.22, cx, ymin + bh * 0.56);
  seg(cx - bw * 0.24, ymin + bh * 0.26, cx + bw * 0.24, ymin + bh * 0.26);
  seg(cx - bw * 0.24, ymin + bh * 0.26, cx - bw * 0.34, ymin + bh * 0.52);
  seg(cx + bw * 0.24, ymin + bh * 0.26, cx + bw * 0.34, ymin + bh * 0.52);
  seg(cx, ymin + bh * 0.56, cx - bw * 0.16, ymax - 2);
  seg(cx, ymin + bh * 0.56, cx + bw * 0.16, ymax - 2);
}

function renderTracks(tracks, mode) {
  drawPrivacyBackdrop(tracks, mode);
  for (const tr of tracks) {
    const col = PALETTE[tr.id % PALETTE.length];
    if (mode === 'skeleton') drawSkeleton(tr, col);

    if (tr.trail.length >= 2) {
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.beginPath();
      tr.trail.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
    }

    const { xmin, ymin, xmax, ymax } = tr.box;
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

    const lbl = `#${tr.id} ${tr.label} ${tr.score.toFixed(2)}`;
    ctx.font = '12px ui-monospace, monospace';
    const tw = ctx.measureText(lbl).width + 8;
    ctx.fillStyle = col;
    ctx.fillRect(xmin, Math.max(0, ymin - 18), tw, 18);
    ctx.fillStyle = '#090d1a';
    ctx.fillText(lbl, xmin + 4, Math.max(13, ymin - 5));
  }
}

async function loop() {
  if (!running || video.paused || video.ended) return;
  const detFn = await ensureDetector();

  const t0 = performance.now();
  const off = document.createElement('canvas');
  off.width = canvas.width;
  off.height = canvas.height;
  off.getContext('2d').drawImage(video, 0, 0, off.width, off.height);

  const rawDets = await detFn(off.toDataURL('image/jpeg', 0.8), {
    threshold: 0.15,
    percentage: false,
  });

  const peopleOnly = peopleOnlyChk.checked;
  const filtered = rawDets.filter(d => !peopleOnly || d.label.toLowerCase() === 'person');
  const tracks = tracker.update(filtered);

  const dt = Math.max(1, performance.now() - t0);
  fpsEl.textContent = (1000 / dt).toFixed(1);
  activeEl.textContent = String(tracks.length);
  uniqueEl.textContent = String(tracker.uniqueCount);

  renderTracks(tracks, privacySelect.value);

  for (const tr of tracks) {
    records.push({
      frame_idx: frameIdx,
      track_id: tr.id,
      class_name: tr.label,
      confidence: Number(tr.score.toFixed(4)),
      x1: Number(tr.box.xmin.toFixed(1)),
      y1: Number(tr.box.ymin.toFixed(1)),
      x2: Number(tr.box.xmax.toFixed(1)),
      y2: Number(tr.box.ymax.toFixed(1)),
    });
  }
  frameIdx++;
  requestAnimationFrame(loop);
}

document.getElementById('webcamBtn').addEventListener('click', async () => {
  await ensureDetector();
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 360;
  tracker.reset();
  records = [];
  frameIdx = 0;
  running = true;
  loop();
});

document.getElementById('videoFile').addEventListener('change', async e => {
  const file = e.target.files?.[0];
  if (!file) return;
  await ensureDetector();
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  await video.play();
  canvas.width = Math.min(960, video.videoWidth || 640);
  canvas.height = Math.round(canvas.width * ((video.videoHeight || 360) / (video.videoWidth || 640)));
  tracker.reset();
  records = [];
  frameIdx = 0;
  running = true;
  loop();
});

document.getElementById('stopBtn').addEventListener('click', () => {
  running = false;
  video.pause();
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  statusEl.textContent = 'Stopped.';
});

document.getElementById('exportBtn').addEventListener('click', () => {
  const blob = new Blob([records.map(r => JSON.stringify(r)).join('\n') + '\n'], {
    type: 'application/x-ndjson',
  });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'trackflow_tracks.jsonl';
  a.click();
});
