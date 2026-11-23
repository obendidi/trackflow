import { pipeline, env, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
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

// Per-model optimal dtype/device configuration
const MODEL_CONFIG = {
  'onnx-community/rfdetr_nano-ONNX': { webgpuDtype: 'fp32', wasmDtype: 'q8', forceWasm: false },
  'onnx-community/rfdetr_base-ONNX': { webgpuDtype: 'fp32', wasmDtype: 'q8', forceWasm: false },
  'onnx-community/rtdetr_v2_r18vd-ONNX': { webgpuDtype: 'fp32', wasmDtype: 'q8', forceWasm: true },
  'Xenova/yolos-tiny': { webgpuDtype: 'fp32', wasmDtype: 'q8', forceWasm: false },
};

const runtimeForceWasm = new Set(
  Object.entries(MODEL_CONFIG)
    .filter(([, cfg]) => cfg.forceWasm)
    .map(([id]) => id)
);

let detector = null;
let loadedModelId = null;
let activeDevice = null;
let loadingPromise = null;
let sessionToken = 0;
let running = false;
let latestTracks = [];
let tracker = new BrowserByteTracker();
let frameIdx = 0;
let records = [];
let emaInferFps = 0;

threshSlider.addEventListener('input', () => {
  const val = Number(threshSlider.value);
  threshVal.textContent = val.toFixed(2);
  tracker.highThresh = val;
  tracker.newTrackThresh = Math.min(0.9, val + 0.05);
});

modelSelect.addEventListener('change', () => {
  sessionToken++;
  detector = null;
  loadedModelId = null;
  activeDevice = null;
  latestTracks = [];
  tracker.reset();
  if (running) startInferenceWorker(sessionToken);
});

async function hasRealWebGPU() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

async function loadDetector(modelId, forceWasm = false) {
  const cfg = MODEL_CONFIG[modelId] || { webgpuDtype: 'fp32', wasmDtype: 'q8' };
  const hasWebGPU = !forceWasm && !runtimeForceWasm.has(modelId) && (await hasRealWebGPU());
  const targetDevice = hasWebGPU ? 'webgpu' : 'wasm';
  const dtype = targetDevice === 'webgpu' ? cfg.webgpuDtype : cfg.wasmDtype;

  deviceEl.textContent = `${targetDevice.toUpperCase()}…`;
  statusEl.textContent = `Loading ${modelId} (${dtype}) on ${targetDevice.toUpperCase()} — cached in browser after first download…`;

  try {
    const det = await pipeline('object-detection', modelId, {
      device: targetDevice,
      dtype,
    });
    detector = det;
    loadedModelId = modelId;
    activeDevice = targetDevice;
    deviceEl.textContent = targetDevice.toUpperCase();
    statusEl.textContent = `Running ${modelId} 100% locally on ${targetDevice.toUpperCase()}.`;
    return det;
  } catch (err) {
    if (targetDevice === 'webgpu') {
      runtimeForceWasm.add(modelId);
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

function renderFrame() {
  if (!running) return;
  if (!video.paused && !video.ended && video.readyState >= 2) {
    const mode = privacySelect.value;
    drawPrivacyBackdrop(latestTracks, mode);

    for (const tr of latestTracks) {
      const col = PALETTE[tr.id % PALETTE.length];
      ctx.globalAlpha = tr.missed > 0 ? 0.65 : 1.0;

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
      ctx.lineWidth = 2.2;
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

      const lbl = `#${tr.id} ${tr.label} ${tr.score.toFixed(2)}`;
      ctx.font = '600 12px ui-monospace, monospace';
      const tw = ctx.measureText(lbl).width + 10;
      ctx.fillStyle = col;
      ctx.fillRect(xmin, Math.max(0, ymin - 19), tw, 19);
      ctx.fillStyle = '#090d1a';
      ctx.fillText(lbl, xmin + 5, Math.max(13, ymin - 5));
    }
    ctx.globalAlpha = 1.0;
  }
  requestAnimationFrame(renderFrame);
}

/**
 * Decoupled async inference worker: processes the latest video frame continuously
 * without ever blocking or stuttering the 60 FPS video render loop.
 */
async function startInferenceWorker(myToken) {
  let lastInferTs = performance.now();
  let lastVideoTime = -1;

  while (running && myToken === sessionToken) {
    if (video.paused || video.ended || video.readyState < 2) {
      await new Promise(r => setTimeout(r, 40));
      continue;
    }

    // Reset tracker cleanly when a looped video wraps back to the beginning
    if (!video.srcObject && lastVideoTime > 0 && video.currentTime + 0.5 < lastVideoTime) {
      tracker.reset();
      latestTracks = [];
    }
    lastVideoTime = video.currentTime;

    try {
      let detFn = await ensureDetector();
      if (!running || myToken !== sessionToken) break;

      const t0 = performance.now();
      // Downscale offscreen capture to <= 448px wide to accelerate RawImage pixel transfer
      const inferW = Math.min(448, canvas.width);
      const inferH = Math.max(1, Math.round(inferW * (canvas.height / Math.max(1, canvas.width))));
      if (offscreen.width !== inferW || offscreen.height !== inferH) {
        offscreen.width = inferW;
        offscreen.height = inferH;
      }
      offCtx.drawImage(video, 0, 0, inferW, inferH);
      const frameImage = RawImage.fromCanvas(offscreen);

      let rawDets;
      try {
        rawDets = await detFn(frameImage, {
          threshold: 0.18,
          percentage: false,
        });
      } catch (inferErr) {
        if (activeDevice === 'webgpu') {
          const modelId = modelSelect.value;
          runtimeForceWasm.add(modelId);
          detector = null;
          loadedModelId = null;
          statusEl.textContent = `WebGPU op unsupported for ${modelId}; switching cleanly to WASM…`;
          detFn = await loadDetector(modelId, true);
          if (!running || myToken !== sessionToken) break;
          rawDets = await detFn(frameImage, {
            threshold: 0.18,
            percentage: false,
          });
        } else {
          throw inferErr;
        }
      }

      if (!running || myToken !== sessionToken) break;

      const scaleX = canvas.width / inferW;
      const scaleY = canvas.height / inferH;
      const peopleOnly = peopleOnlyChk.checked;
      const filtered = rawDets
        .filter(d => !peopleOnly || d.label.toLowerCase() === 'person')
        .map(d => ({
          ...d,
          box: {
            xmin: d.box.xmin * scaleX,
            ymin: d.box.ymin * scaleY,
            xmax: d.box.xmax * scaleX,
            ymax: d.box.ymax * scaleY,
          },
        }));

      const now = performance.now();
      const dtSec = (now - lastInferTs) / 1000;
      lastInferTs = now;

      const tracks = tracker.update(filtered, dtSec);
      latestTracks = tracks;

      const dt = Math.max(1, now - t0);
      const instFps = 1000 / dt;
      emaInferFps = emaInferFps === 0 ? instFps : 0.75 * emaInferFps + 0.25 * instFps;
      fpsEl.textContent = emaInferFps.toFixed(1);
      activeEl.textContent = String(tracks.length);
      uniqueEl.textContent = String(tracker.uniqueCount);

      for (const tr of tracks) {
        if (tr.missed === 0) {
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
      }
      frameIdx++;
    } catch (err) {
      if (myToken !== sessionToken) break;
      running = false;
      statusEl.textContent = `Error: ${err?.message || err}`;
      console.error('Trackflow WebGPU playground error:', err);
      break;
    }

    // Yield one macrotask so UI controls stay responsive
    await new Promise(r => setTimeout(r, 0));
  }
}

function beginSession() {
  sessionToken++;
  tracker.reset();
  latestTracks = [];
  records = [];
  frameIdx = 0;
  emaInferFps = 0;
  running = true;
  requestAnimationFrame(renderFrame);
  startInferenceWorker(sessionToken);
}

document.getElementById('webcamBtn').addEventListener('click', async () => {
  try {
    await ensureDetector();
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 360;
    beginSession();
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
    beginSession();
  } catch (err) {
    statusEl.textContent = `Video error: ${err?.message || err}`;
  }
});

document.getElementById('stopBtn').addEventListener('click', () => {
  sessionToken++;
  running = false;
  latestTracks = [];
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
