/**
 * Detection pre/post-processing for the browser playground.
 *
 * Transformers.js' object-detection pipeline scores every DETR-family model with a softmax over
 * classes. RT-DETR, RF-DETR and D-FINE are trained with sigmoid (focal) classification, so that
 * inflates confidences and lets background queries through. We call the model directly and decode
 * its raw `logits` / `pred_boxes` here, mirroring `postprocess_detr_outputs` in
 * `src/trackflow/detector.py` (same sigmoid scoring, stable ordering, and class-aware NMS).
 */

// Models trained with a softmax over classes plus a trailing "no-object" class.
const SOFTMAX_MODEL_TYPES = new Set(['detr', 'yolos', 'table-transformer']);

export function scoringModeFor(modelType) {
  return SOFTMAX_MODEL_TYPES.has(modelType) ? 'softmax' : 'sigmoid';
}

/** RGBA pixels (already resized to the model input) -> CHW float32 tensor data. */
export function preprocessRGBA(rgba, width, height, prep) {
  const plane = width * height;
  const out = new Float32Array(3 * plane);
  const scale = prep.do_rescale === false ? 1 : (prep.rescale_factor ?? 1 / 255);
  const normalize = Boolean(prep.do_normalize);
  const mean = prep.image_mean ?? [0.485, 0.456, 0.406];
  const std = prep.image_std ?? [0.229, 0.224, 0.225];
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      let v = rgba[i * 4 + c] * scale;
      if (normalize) v = (v - mean[c]) / std[c];
      out[c * plane + i] = v;
    }
  }
  return out;
}

export function inputSizeFor(prep, fallback = 640) {
  const size = prep.size ?? {};
  if (typeof size === 'number') return { width: size, height: size };
  return {
    width: size.width ?? size.shortest_edge ?? fallback,
    height: size.height ?? size.shortest_edge ?? fallback,
  };
}

function iou(a, b) {
  const iw = Math.max(0, Math.min(a.xmax, b.xmax) - Math.max(a.xmin, b.xmin));
  const ih = Math.max(0, Math.min(a.ymax, b.ymax) - Math.max(a.ymin, b.ymin));
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a.xmax - a.xmin) * Math.max(0, a.ymax - a.ymin);
  const areaB = Math.max(0, b.xmax - b.xmin) * Math.max(0, b.ymax - b.ymin);
  return inter / Math.max(areaA + areaB - inter, 1e-6);
}

/** Class-aware NMS on score-sorted detections (same rule as `_nms_class_aware`). */
export function nmsClassAware(dets, iouThreshold) {
  if (dets.length <= 1 || iouThreshold <= 0 || iouThreshold >= 1) return dets;
  const kept = [];
  for (const det of dets) {
    if (!kept.some(prev => prev.classId === det.classId && iou(prev.box, det.box) >= iouThreshold)) {
      kept.push(det);
    }
  }
  return kept;
}

/**
 * Decode DETR-family outputs into pixel-space detections.
 * @param {object} o
 * @param {Float32Array} o.logits  [numQueries * numClasses]
 * @param {number[]} o.logitsDims  [1, numQueries, numClasses]
 * @param {Float32Array} o.boxes   [numQueries * 4] normalized cx, cy, w, h
 */
export function postprocessDetr({
  logits,
  logitsDims,
  boxes,
  id2label,
  width,
  height,
  mode = 'sigmoid',
  threshold = 0.15,
  allowedLabels = null,
  nmsIou = 0.65,
  maxDetections = 300,
}) {
  const numClasses = logitsDims[logitsDims.length - 1];
  const numQueries = logitsDims[logitsDims.length - 2];
  const cands = [];

  for (let q = 0; q < numQueries; q++) {
    const row = q * numClasses;
    let best = 0;
    for (let c = 1; c < numClasses; c++) if (logits[row + c] > logits[row + best]) best = c;

    let score;
    if (mode === 'softmax') {
      if (best === numClasses - 1) continue; // no-object class
      let denom = 0;
      const mx = logits[row + best];
      for (let c = 0; c < numClasses; c++) denom += Math.exp(logits[row + c] - mx);
      score = 1 / denom;
    } else {
      const x = Math.max(-60, Math.min(60, logits[row + best]));
      score = 1 / (1 + Math.exp(-x));
    }
    if (score >= threshold) cands.push({ q, classId: best, score });
  }

  cands.sort((a, b) => b.score - a.score); // stable: ties keep query order
  const dets = [];
  for (const { q, classId, score } of cands.slice(0, maxDetections)) {
    const label = String(id2label?.[classId] ?? id2label?.[String(classId)] ?? classId);
    if (allowedLabels && !allowedLabels.has(label.toLowerCase())) continue;

    const [cx, cy, w, h] = [boxes[q * 4], boxes[q * 4 + 1], boxes[q * 4 + 2], boxes[q * 4 + 3]];
    let x1, y1, x2, y2;
    if (Math.max(cx, cy, w, h) <= 2.0) {
      x1 = (cx - 0.5 * w) * width;
      y1 = (cy - 0.5 * h) * height;
      x2 = (cx + 0.5 * w) * width;
      y2 = (cy + 0.5 * h) * height;
    } else {
      x1 = cx - 0.5 * w; y1 = cy - 0.5 * h; x2 = cx + 0.5 * w; y2 = cy + 0.5 * h;
    }
    const box = {
      xmin: Math.min(Math.max(x1, 0), width),
      ymin: Math.min(Math.max(y1, 0), height),
      xmax: Math.min(Math.max(x2, 0), width),
      ymax: Math.min(Math.max(y2, 0), height),
    };
    if (box.xmax - box.xmin < 2 || box.ymax - box.ymin < 2) continue;
    dets.push({ box, score, classId, label });
  }
  return nmsClassAware(dets, nmsIou);
}
