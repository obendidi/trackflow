import { AutoModelForObjectDetection, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import { inputSizeFor, postprocessDetr, preprocessRGBA, scoringModeFor } from './detector.js';
import { ByteTracker, iouBufferForInterval } from './tracker.js';

env.allowLocalModels = false;

const PALETTE = ['#34d399', '#38bdf8', '#a78bfa', '#fbbf24', '#f472b6', '#60a5fa', '#4ade80'];
const DETECT_THRESHOLD = 0.15; // keep the low-score tier for ByteTrack's second association stage
const COAST_FRAMES = 3; // keep drawing (and blurring) a missed track at its predicted box this long
const BLUR_MARGIN = 0.08;

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

// Per-model dtype/device configuration
const MODEL_CONFIG = {
  'onnx-community/yolo26n-ONNX': { revision: 'a8dc7e14743e1cea8ccd493bd99b4c2827de1acf', mode: 'sigmoid', webgpuDtype: 'fp32', wasmDtype: 'q8' },
  'onnx-community/rfdetr_nano-ONNX': { revision: 'eae21cee0687a91bcf9fa071605c48d7705d2d91', webgpuDtype: 'fp32', wasmDtype: 'q8', forceWasm: false },
  'onnx-community/rfdetr_base-ONNX': { revision: '7b18fa82ba7b1d6bc9446518cf2cc1d991097a3f', webgpuDtype: 'fp32', wasmDtype: 'q8', forceWasm: false },
  'onnx-community/rtdetr_v2_r18vd-ONNX': { revision: '936f90b6a476c6da4dfe053fc521af55285976ba', webgpuDtype: 'fp32', wasmDtype: 'q8', forceWasm: true },
};

const runtimeForceWasm = new Set(
  Object.entries(MODEL_CONFIG)
    .filter(([, cfg]) => cfg.forceWasm)
    .map(([id]) => id)
);

let detector = null; // { model, prep, input, mode, id2label, modelId, device }
let worker = Promise.resolve();
let renderRequest = 0;
let fileUrl = null;
let latestFrame = null;
let sessionToken = 0;
let running = false;
let latestTracks = [];
let latestCoasting = [];
let latestDets = [];
let trails = new Map();
const tracker = new ByteTracker();
let frameIdx = 0;
let records = [];
let emaInferFps = 0;

function applyThreshold(val) {
  // Same mapping as TrackflowPipeline: high = threshold, low = 0.35x, new tracks = +0.05
  tracker.highThresh = val;
  tracker.lowThresh = Math.max(0.1, val * 0.35);
  tracker.newTrackThresh = Math.min(0.95, val + 0.05);
}
applyThreshold(Number(threshSlider.value));

threshSlider.addEventListener('input', () => {
  const val = Number(threshSlider.value);
  threshVal.textContent = val.toFixed(2);
  applyThreshold(val);
});

function resetTracking() {
  tracker.reset();
  trails = new Map();
  latestTracks = [];
  latestCoasting = [];
  latestDets = [];
  latestFrame = null;
  tracker.iouBuffer = 0;
}

modelSelect.addEventListener('change', () => {
  if (!running) return;
  sessionToken++;
  resetRecording();
  scheduleWorker(sessionToken);
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
  const device = hasWebGPU ? 'webgpu' : 'wasm';
  const dtype = device === 'webgpu' ? cfg.webgpuDtype : cfg.wasmDtype;

  deviceEl.textContent = `${device.toUpperCase()}…`;
  statusEl.textContent = `Loading ${modelId} (${dtype}) on ${device.toUpperCase()} — cached in browser after first download…`;

  try {
    const [model, prep] = await Promise.all([
      AutoModelForObjectDetection.from_pretrained(modelId, { device, dtype, revision: cfg.revision }),
      fetch(`https://huggingface.co/${modelId}/resolve/${cfg.revision}/preprocessor_config.json`).then(r => {
        if (!r.ok) throw new Error(`preprocessor_config.json: HTTP ${r.status}`);
        return r.json();
      }),
    ]);
    const loaded = {
      model,
      prep,
      input: inputSizeFor(prep),
      mode: cfg.mode || scoringModeFor(model.config.model_type),
      id2label: model.config.id2label,
      modelId,
      device,
    };
    deviceEl.textContent = device.toUpperCase();
    statusEl.textContent = `Running ${modelId} 100% locally on ${device.toUpperCase()}.`;
    return loaded;
  } catch (err) {
    if (device === 'webgpu') {
      runtimeForceWasm.add(modelId);
      statusEl.textContent = `WebGPU init failed (${err?.message || err}); switching to WASM…`;
      return loadDetector(modelId, true);
    }
    throw err;
  }
}

async function ensureDetector() {
  const modelId = modelSelect.value;
  if (detector && detector.modelId === modelId) return detector;
  if (detector) await detector.model.dispose();
  detector = null;
  detector = await loadDetector(modelId);
  return detector;
}

/** Run the detector on the current video frame; boxes come back in canvas pixels. */
async function detectFrame(det, captured) {
  const { width, height } = det.input;
  if (offscreen.width !== width || offscreen.height !== height) {
    offscreen.width = width;
    offscreen.height = height;
  }
  // Plain resize to the model's input size (what the HF image processor does; no letterbox).
  offCtx.drawImage(captured, 0, 0, width, height);
  const { data } = offCtx.getImageData(0, 0, width, height);
  const pixels = preprocessRGBA(data, width, height, det.prep);
  const out = await det.model({ pixel_values: new Tensor('float32', pixels, [1, 3, height, width]) });
  return postprocessDetr({
    logits: out.logits.data,
    logitsDims: out.logits.dims,
    boxes: out.pred_boxes.data,
    id2label: det.id2label,
    width: canvas.width,
    height: canvas.height,
    mode: det.mode,
    threshold: Math.min(DETECT_THRESHOLD, tracker.lowThresh),
    allowedLabels: peopleOnlyChk.checked ? new Set(['person']) : null,
  });
}

function expand(box, margin) {
  const w = box.xmax - box.xmin;
  const h = box.ymax - box.ymin;
  return {
    xmin: Math.max(0, box.xmin - w * margin),
    ymin: Math.max(0, box.ymin - h * margin),
    xmax: Math.min(canvas.width, box.xmax + w * margin),
    ymax: Math.min(canvas.height, box.ymax + h * margin),
  };
}

/** Every box that may contain a person: raw detections (any score) + tracks + coasting tracks. */
function privacyBoxes() {
  return [...latestDets, ...latestTracks, ...latestCoasting].map(o => expand(o.box, BLUR_MARGIN));
}

function drawPrivacyBackdrop(mode) {
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

  if (mode === 'none') {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return;
  }
  // Display the exact pixels that produced the boxes, including while inference is pending.
  if (!latestFrame) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  ctx.drawImage(latestFrame, 0, 0);

  for (const { xmin, ymin, xmax, ymax } of privacyBoxes()) {
    const w = Math.max(4, xmax - xmin);
    const h = Math.max(4, ymax - ymin);
    ctx.save();
    ctx.filter = 'blur(16px)';
    ctx.beginPath();
    if (mode === 'face-blur') {
      const fh = Math.max(8, h * 0.3);
      ctx.ellipse(xmin + w * 0.5, ymin + fh * 0.5, w * 0.42, fh * 0.5, 0, 0, Math.PI * 2);
    } else {
      ctx.rect(xmin, ymin, w, h);
    }
    ctx.clip();
    if (ctx.filter === 'blur(16px)' && 'filter' in CanvasRenderingContext2D.prototype) {
      ctx.drawImage(latestFrame, 0, 0);
    } else {
      ctx.fillStyle = '#10131a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.restore();
  }
}

function drawStickFigure(tr, color) {
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

function drawTrack(tr, mode) {
  const col = PALETTE[tr.id % PALETTE.length];
  ctx.globalAlpha = tr.coasting ? 0.5 : 1.0;
  if (mode === 'skeleton') drawStickFigure(tr, col);

  const trail = trails.get(tr.id);
  if (trail && trail.length >= 2) {
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.beginPath();
    trail.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
  }

  const { xmin, ymin, xmax, ymax } = tr.box;
  ctx.strokeStyle = col;
  ctx.lineWidth = 2.2;
  if (tr.coasting) ctx.setLineDash([6, 4]);
  ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
  ctx.setLineDash([]);

  const lbl = `#${tr.id} ${tr.label} ${tr.score.toFixed(2)}`;
  ctx.font = '600 12px ui-monospace, monospace';
  const tw = ctx.measureText(lbl).width + 10;
  ctx.fillStyle = col;
  ctx.fillRect(xmin, Math.max(0, ymin - 19), tw, 19);
  ctx.fillStyle = '#090d1a';
  ctx.fillText(lbl, xmin + 5, Math.max(13, ymin - 5));
  ctx.globalAlpha = 1.0;
}

function renderFrame() {
  if (!running) return;
  if (!video.paused && !video.ended && video.readyState >= 2) {
    const mode = privacySelect.value;
    drawPrivacyBackdrop(mode);
    for (const tr of latestCoasting) drawTrack(tr, mode);
    for (const tr of latestTracks) drawTrack(tr, mode);
  }
  renderRequest = requestAnimationFrame(renderFrame);
}

function updateTrails(tracks, coasting) {
  for (const tr of tracks) {
    const trail = trails.get(tr.id) || [];
    trail.push([(tr.box.xmin + tr.box.xmax) * 0.5, tr.box.ymax]);
    if (trail.length > 36) trail.shift();
    trails.set(tr.id, trail);
  }
  const alive = new Set([...tracks, ...coasting].map(t => t.id));
  for (const id of trails.keys()) if (!alive.has(id)) trails.delete(id);
}

/**
 * Decoupled async inference worker: processes the latest video frame continuously
 * without blocking the 60 FPS render loop.
 */
async function startInferenceWorker(myToken) {
  let lastVideoTime = -1;
  let lastFrameTs = null; // video time (files) or wall-clock time (webcam) of the last processed frame
  let emaInterval = null;
  let captured = document.createElement('canvas');

  while (running && myToken === sessionToken) {
    if (video.paused || video.ended || video.readyState < 2) {
      await new Promise(r => setTimeout(r, 40));
      continue;
    }

    try {
      let det = await ensureDetector();
      if (!running || myToken !== sessionToken) break;
      const frameTs = video.currentTime;
      if (frameTs === lastVideoTime) {
        await new Promise(r => setTimeout(r, 8));
        continue;
      }
      if (lastVideoTime > frameTs) {
        resetRecording();
        lastFrameTs = null;
        emaInterval = null;
      }
      lastVideoTime = frameTs;
      if (captured.width !== canvas.width) captured.width = canvas.width;
      if (captured.height !== canvas.height) captured.height = canvas.height;
      captured.getContext('2d').drawImage(video, 0, 0, captured.width, captured.height);
      const t0 = performance.now();
      let dets;
      try {
        dets = await detectFrame(det, captured);
      } catch (inferErr) {
        if (det.device !== 'webgpu') throw inferErr;
        runtimeForceWasm.add(det.modelId);
        await det.model.dispose();
        detector = null;
        statusEl.textContent = `WebGPU op unsupported for ${det.modelId}; switching to WASM…`;
        detector = await loadDetector(det.modelId, true);
        det = detector;
        if (!running || myToken !== sessionToken) break;
        dets = await detectFrame(det, captured);
      }
      if (!running || myToken !== sessionToken) break;

      // Widen association when inference runs slowly, so walking people keep their ID.

      if (lastFrameTs !== null && frameTs > lastFrameTs) {
        const dt = frameTs - lastFrameTs;
        emaInterval = emaInterval === null ? dt : 0.8 * emaInterval + 0.2 * dt;
        tracker.iouBuffer = iouBufferForInterval(emaInterval);
      }
      lastFrameTs = frameTs;

      const tracks = tracker.update(dets);
      const coasting = tracker.coastingTracks(COAST_FRAMES);
      updateTrails(tracks, coasting);
      const previous = latestFrame;
      latestFrame = captured;
      captured = previous || document.createElement('canvas');
      latestDets = dets;
      latestTracks = tracks;
      latestCoasting = coasting;

      const instFps = 1000 / Math.max(1, performance.now() - t0);
      emaInferFps = emaInferFps === 0 ? instFps : 0.75 * emaInferFps + 0.25 * instFps;
      fpsEl.textContent = emaInferFps.toFixed(1);
      activeEl.textContent = String(tracks.length);
      uniqueEl.textContent = String(tracker.uniqueCount);

      for (const tr of tracks) {
        records.push({
          frame_idx: frameIdx,
          timestamp_s: Number(frameTs.toFixed(4)),
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
      if (myToken !== sessionToken) break;
      stopSource();
      statusEl.textContent = `Error: ${err?.message || err}`;
      console.error('Trackflow playground error:', err);
      break;
    }

    // Yield one macrotask so UI controls stay responsive
    await new Promise(r => setTimeout(r, 0));
  }
}

function resetRecording() {
  resetTracking();
  records = [];
  frameIdx = 0;
  emaInferFps = 0;
}

function scheduleWorker(token) {
  worker = worker.then(() => startInferenceWorker(token));
}

function stopSource() {
  sessionToken++;
  running = false;
  cancelAnimationFrame(renderRequest);
  video.pause();
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  if (fileUrl) {
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(fileUrl);
    fileUrl = null;
  }
  resetTracking();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function beginSession() {
  resetRecording();
  running = true;
  renderRequest = requestAnimationFrame(renderFrame);
  scheduleWorker(sessionToken);
}

document.getElementById('webcamBtn').addEventListener('click', async () => {
  stopSource();
  const token = sessionToken;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } });
    if (token !== sessionToken) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    video.srcObject = stream;
    await video.play();
    if (token !== sessionToken) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 360;
    beginSession();
  } catch (err) {
    if (token === sessionToken) {
      stopSource();
      statusEl.textContent = `Webcam error: ${err?.message || err}`;
    }
  }
});

document.getElementById('videoFile').addEventListener('change', async e => {
  const file = e.target.files?.[0];
  if (!file) return;
  stopSource();
  const token = sessionToken;
  try {
    fileUrl = URL.createObjectURL(file);
    video.src = fileUrl;
    await video.play();
    if (token !== sessionToken) return;
    canvas.width = Math.min(960, video.videoWidth || 640);
    canvas.height = Math.round(canvas.width * ((video.videoHeight || 360) / (video.videoWidth || 640)));
    beginSession();
  } catch (err) {
    if (token === sessionToken) {
      stopSource();
      statusEl.textContent = `Video error: ${err?.message || err}`;
    }
  }
});

document.getElementById('stopBtn').addEventListener('click', () => {
  stopSource();
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
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
