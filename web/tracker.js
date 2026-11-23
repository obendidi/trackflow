/**
 * Production-grade BrowserByteTracker for Trackflow WebGPU/WASM playground:
 * - Class-aware NMS deduplication before association (prevents overlapping phantom IDs)
 * - Real-time delta (`dtSec`) damped velocity prediction (works cleanly from 2 FPS WASM to 60 FPS WebGPU)
 * - Hybrid IoU + normalized center-distance association (prevents ID breaks when a person moves fast between slow frames)
 * - Exponential box smoothing (eliminates bounding-box jitter)
 * - Confirmation (`minHits`) + short-term coasting (`maxCoast`) to stop flickering on missed frames
 */

function computeIoU(a, b) {
  const x1 = Math.max(a.xmin, b.xmin);
  const y1 = Math.max(a.ymin, b.ymin);
  const x2 = Math.min(a.xmax, b.xmax);
  const y2 = Math.min(a.ymax, b.ymax);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a.xmax - a.xmin) * Math.max(0, a.ymax - a.ymin);
  const areaB = Math.max(0, b.xmax - b.xmin) * Math.max(0, b.ymax - b.ymin);
  return inter / Math.max(areaA + areaB - inter, 1e-6);
}

function boxCenter(b) {
  return [(b.xmin + b.xmax) * 0.5, (b.ymin + b.ymax) * 0.5];
}

function boxDiag(b) {
  return Math.hypot(Math.max(1, b.xmax - b.xmin), Math.max(1, b.ymax - b.ymin));
}

/**
 * Class-aware Non-Maximum Suppression to remove duplicate overlapping DETR/ViT queries.
 */
export function nmsDetections(detections, iouThresh = 0.55) {
  if (!detections || detections.length <= 1) return detections || [];
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept = [];

  for (const det of sorted) {
    const w = det.box.xmax - det.box.xmin;
    const h = det.box.ymax - det.box.ymin;
    if (w < 8 || h < 8) continue;

    let duplicate = false;
    for (const prev of kept) {
      if (prev.label === det.label && computeIoU(prev.box, det.box) >= iouThresh) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) kept.push(det);
  }
  return kept;
}

/**
 * Hybrid IoU + center-distance affinity so tracks survive both tight overlap AND fast motion
 * across lower-FPS WASM frames without splitting into new IDs.
 */
function associationScore(track, det, minIou, maxNormDist = 1.65) {
  const iou = computeIoU(track.predBox, det.box);
  if (iou >= minIou) return 1.0 + iou; // Prefer genuine IoU overlap

  // Fallback: normalized center distance & compatible scale (prevents ID fragmentation on fast motion)
  const [tcx, tcy] = boxCenter(track.predBox);
  const [dcx, dcy] = boxCenter(det.box);
  const scale = Math.max(boxDiag(track.predBox), boxDiag(det.box), 40);
  const normDist = Math.hypot(dcx - tcx, dcy - tcy) / scale;
  if (normDist > maxNormDist) return -1;

  const tArea = Math.max(1, (track.predBox.xmax - track.predBox.xmin) * (track.predBox.ymax - track.predBox.ymin));
  const dArea = Math.max(1, (det.box.xmax - det.box.xmin) * (det.box.ymax - det.box.ymin));
  const areaRatio = Math.min(tArea, dArea) / Math.max(tArea, dArea);
  if (areaRatio < 0.25) return -1;

  return (1.0 - normDist / maxNormDist) * 0.8 + areaRatio * 0.2;
}

function greedyMatch(tracks, dets, minIou, allowDistFallback = true) {
  const pairs = [];
  for (let i = 0; i < tracks.length; i++) {
    for (let j = 0; j < dets.length; j++) {
      if (tracks[i].label !== dets[j].label) continue;
      const score = allowDistFallback
        ? associationScore(tracks[i], dets[j], minIou)
        : (computeIoU(tracks[i].predBox, dets[j].box) >= minIou
            ? computeIoU(tracks[i].predBox, dets[j].box)
            : -1);
      if (score > 0) pairs.push([score, i, j]);
    }
  }
  pairs.sort((a, b) => b[0] - a[0]);

  const usedT = new Set();
  const usedD = new Set();
  const matches = [];
  for (const [, i, j] of pairs) {
    if (!usedT.has(i) && !usedD.has(j)) {
      usedT.add(i);
      usedD.add(j);
      matches.push([i, j]);
    }
  }
  const unmatchedT = tracks.map((_, i) => i).filter(i => !usedT.has(i));
  const unmatchedD = dets.map((_, j) => j).filter(j => !usedD.has(j));
  return { matches, unmatchedT, unmatchedD };
}

export class BrowserByteTracker {
  constructor({
    highThresh = 0.45,
    lowThresh = 0.18,
    newTrackThresh = 0.5,
    maxAge = 24,
    minIou = 0.2,
    nmsIou = 0.55,
    minHits = 2,
    maxCoast = 4,
    smoothAlpha = 0.65,
  } = {}) {
    this.highThresh = highThresh;
    this.lowThresh = lowThresh;
    this.newTrackThresh = newTrackThresh;
    this.maxAge = maxAge;
    this.minIou = minIou;
    this.nmsIou = nmsIou;
    this.minHits = minHits;
    this.maxCoast = maxCoast;
    this.smoothAlpha = smoothAlpha;
    this.nextId = 1;
    this.tracks = [];
    this.uniqueCount = 0;
  }

  reset() {
    this.nextId = 1;
    this.tracks = [];
    this.uniqueCount = 0;
  }

  update(rawDetections, dtSec = 1 / 15) {
    const dt = Math.max(0.016, Math.min(1.5, dtSec));

    // 1. Deduplicate overlapping queries first so duplicates never spawn phantom tracks
    const detections = nmsDetections(rawDetections, this.nmsIou);

    // 2. Predict next box using real elapsed time `dt` (vx/vy in px/sec)
    for (const tr of this.tracks) {
      const damping = tr.missed > 0 ? 0.6 : 0.88;
      tr.vx *= damping;
      tr.vy *= damping;
      const dx = tr.vx * dt;
      const dy = tr.vy * dt;
      tr.predBox = {
        xmin: tr.box.xmin + dx,
        ymin: tr.box.ymin + dy,
        xmax: tr.box.xmax + dx,
        ymax: tr.box.ymax + dy,
      };
      tr.missed += 1;
      tr.elapsedSinceMeas += dt;
    }

    const high = detections.filter(d => d.score >= this.highThresh);
    const low = detections.filter(d => d.score >= this.lowThresh && d.score < this.highThresh);
    const updatedIds = new Set();

    // 3. Stage 1: match confirmed tracks first, then tentative tracks, against high-conf detections
    const confirmed = this.tracks.filter(t => t.confirmed);
    const tentative = this.tracks.filter(t => !t.confirmed);

    const s1 = greedyMatch(confirmed, high, this.minIou, true);
    for (const [ti, di] of s1.matches) {
      this._applyUpdate(confirmed[ti], high[di]);
      updatedIds.add(confirmed[ti].id);
    }

    const remHigh = s1.unmatchedD.map(i => high[i]);
    const s1b = greedyMatch(tentative, remHigh, this.minIou, true);
    for (const [ti, di] of s1b.matches) {
      this._applyUpdate(tentative[ti], remHigh[di]);
      updatedIds.add(tentative[ti].id);
    }

    // 4. Stage 2: match remaining recently-active confirmed tracks against low-conf detections
    const remConfirmed = s1.unmatchedT
      .map(i => confirmed[i])
      .filter(t => t.missed <= 3);
    const s2 = greedyMatch(remConfirmed, low, 0.3, false);
    for (const [ti, di] of s2.matches) {
      this._applyUpdate(remConfirmed[ti], low[di]);
      updatedIds.add(remConfirmed[ti].id);
    }

    // 5. Spawn new tentative tracks for unmatched high-confidence detections
    const finalUnmatchedHigh = s1b.unmatchedD.map(i => remHigh[i]);
    for (const det of finalUnmatchedHigh) {
      if (det.score < Math.max(this.highThresh, this.newTrackThresh)) continue;
      const overlapsActive = this.tracks.some(
        t => t.missed <= 2 && t.label === det.label && computeIoU(t.box, det.box) >= 0.45
      );
      if (overlapsActive) continue;

      const [cx, cy] = boxCenter(det.box);
      const instantConfirm = det.score >= 0.62 || this.minHits <= 1;
      const tr = {
        id: instantConfirm ? this.nextId++ : -1,
        box: { ...det.box },
        predBox: { ...det.box },
        lastMeasCx: cx,
        lastMeasCy: cy,
        elapsedSinceMeas: 0,
        label: det.label,
        score: det.score,
        vx: 0,
        vy: 0,
        hits: 1,
        missed: 0,
        confirmed: instantConfirm,
        trail: [[cx, det.box.ymax]],
      };
      if (instantConfirm) this.uniqueCount++;
      this.tracks.push(tr);
      updatedIds.add(tr.id);
    }

    // 6. Coast unmatched confirmed tracks gently for up to `maxCoast` steps so boxes don't blink
    for (const tr of this.tracks) {
      if (!updatedIds.has(tr.id) && tr.missed > 0 && tr.missed <= this.maxCoast) {
        tr.box = { ...tr.predBox };
      }
    }

    // 7. Prune expired tracks
    this.tracks = this.tracks.filter(t => {
      if (!t.confirmed) return t.missed <= 1;
      return t.missed <= this.maxAge;
    });

    return this.tracks
      .filter(t => t.confirmed && t.missed <= this.maxCoast)
      .sort((a, b) => a.id - b.id);
  }

  _applyUpdate(tr, det) {
    const [newCx, newCy] = boxCenter(det.box);
    const dtMeas = Math.max(0.016, tr.elapsedSinceMeas);
    const measVx = (newCx - tr.lastMeasCx) / dtMeas;
    const measVy = (newCy - tr.lastMeasCy) / dtMeas;
    tr.vx = 0.55 * tr.vx + 0.45 * measVx;
    tr.vy = 0.55 * tr.vy + 0.45 * measVy;
    tr.lastMeasCx = newCx;
    tr.lastMeasCy = newCy;
    tr.elapsedSinceMeas = 0;

    // Smooth box edges with EMA (higher weight when frame gap was large so box doesn't lag)
    const a = dtMeas > 0.25 ? 0.85 : this.smoothAlpha;
    tr.box = {
      xmin: a * det.box.xmin + (1 - a) * tr.predBox.xmin,
      ymin: a * det.box.ymin + (1 - a) * tr.predBox.ymin,
      xmax: a * det.box.xmax + (1 - a) * tr.predBox.xmax,
      ymax: a * det.box.ymax + (1 - a) * tr.predBox.ymax,
    };

    tr.score = 0.6 * det.score + 0.4 * tr.score;
    tr.label = det.label;
    tr.hits += 1;
    tr.missed = 0;

    if (!tr.confirmed && tr.hits >= this.minHits) {
      tr.confirmed = true;
      tr.id = this.nextId++;
      this.uniqueCount++;
    }

    const [smoothCx] = boxCenter(tr.box);
    tr.trail.push([smoothCx, tr.box.ymax]);
    if (tr.trail.length > 36) tr.trail.shift();
  }
}
