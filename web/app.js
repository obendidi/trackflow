import { pipeline, env, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';
import { BrowserByteTracker } from './tracker.js';

env.allowLocalModels = false;

const PALETTE = ['#34d399', '#38bdf8', '#a78bfa', '#fbbf24', '#f472b6', '#60a5fa', '#4ade80'];
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const offscreen = document.createElement('canvas');
const offCtx = offscreen.getContext('2d', { willReadFrequently: true });

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

// Models known to use AveragePool ceil_mode=1 (unsupported in ORT WebGPU EP) run on WASM directly.
const forceWasmModels = new Set(['onnx-community/rtdetr_v2_r18vd-ONNX']);

let detector = null;
let loadedModelId = null;
let activeDevice = null;
let loadingPromise = null;
let running = false;
let tracker = new BrowserByteTracker();
let frameIdx = 0;
let records = [];

threshSlider.addEventListener('input', () => {
  threshVal.textContent = Number(threshSlider.value).toFixed(2);
  tracker.highThresh = Number(threshSlider.value);
});

modelSelect.addEventListener('change', () => {
  detector = null;
  loadedModelId = null;
  activeDevice = null;
  tracker.reset();
});

async function loadDetector(modelId, forceWasm = false) {
  const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
  const targetDevice = !forceWasm && hasWebGPU && !forceWasmModels.has(modelId) ? 'webgpu' : 'wasm';
  deviceEl.textContent = targetDevice.toUpperCase();
  statusEl.textContent = `Loading ${modelId} on ${targetDevice.toUpperCase()} (cached in browser after first download)…`;

  try {
    const det = await pipeline('object-detection', modelId, {
      device: targetDevice,
      dtype: targetDevice === 'webgpu' ? 'fp32' : 'q8',
    });
    detector = det;
    loadedModelId = modelId;
    activeDevice = targetDevice;
    deviceEl.textContent = targetDevice.toUpperCase();
    statusEl.textContent = `Ready · ${modelId} running 100% locally on ${targetDevice.toUpperCase()}.`;
    return det;
  } catch (err) {
    if (targetDevice === 'webgpu') {
      forceWasmModels.add(modelId);
      statusEl.textContent = `WebGPU init failed (${err?.message || err}); switching to WASM…`;
      return loadDetector(modelId, true);
    }
    throw err;
  }
}

async function ensureDetector() {
  const modelId = modelSelect.value;
  if (detector && loadedModelId === modelId) return detector;
  if (loadingPromise) return loadingPromise;
  loadingPromise = loadDetector(modelId).finally(() => {
    loadingPromise = null;
  });
  return loadingPromise;
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

  try {
    let detFn = await ensureDetector();

    const t0 = performance.now();
    if (offscreen.width !== canvas.width || offscreen.height !== canvas.height) {
      offscreen.width = canvas.width;
      offscreen.height = canvas.height;
    }
    offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
    const frameImage = RawImage.fromCanvas(offscreen);

    let rawDets;
    try {
      rawDets = await detFn(frameImage, {
        threshold: 0.15,
        percentage: false,
      });
    } catch (inferErr) {
      // ORT WebGPU compiles kernels lazily on the first run() call; gracefully fall back to WASM.
      if (activeDevice === 'webgpu') {
        const modelId = modelSelect.value;
        forceWasmModels.add(modelId);
        detector = null;
        loadedModelId = null;
        statusEl.textContent = `WebGPU op unsupported for ${modelId}; falling back to WASM…`;
        detFn = await loadDetector(modelId, true);
        rawDets = await detFn(frameImage, {
          threshold: 0.15,
          percentage: false,
        });
      } else {
        throw inferErr;
      }
    }

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
  } catch (err) {
    running = false;
    statusEl.textContent = `Error: ${err?.message || err}`;
    console.error('Trackflow WebGPU playground error:', err);
    return;
  }

  if (running) requestAnimationFrame(loop);
}

document.getElementById('webcamBtn').addEventListener('click', async () => {
  try {
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
  } catch (err) {
    statusEl.textContent = `Webcam error: ${err?.message || err}`;
  }
});

document.getElementById('videoFile').addEventListener('change', async e => {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
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
  } catch (err) {
    statusEl.textContent = `Video error: ${err?.message || err}`;
  }
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
