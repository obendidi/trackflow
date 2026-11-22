/**
 * Lightweight two-stage ByteTrack-style multi-object tracker for the Trackflow WebGPU playground.
 */

function iou(a, b) {
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

function greedyMatch(tracks, dets, minIou) {
  const pairs = [];
  for (let i = 0; i < tracks.length; i++) {
    for (let j = 0; j < dets.length; j++) {
      const score = iou(tracks[i].box, dets[j].box);
      if (score >= minIou) pairs.push([score, i, j]);
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
  constructor({ highThresh = 0.45, lowThresh = 0.15, maxAge = 24, minIou = 0.2 } = {}) {
    this.highThresh = highThresh;
    this.lowThresh = lowThresh;
    this.maxAge = maxAge;
    this.minIou = minIou;
    this.nextId = 1;
    this.tracks = [];
    this.uniqueCount = 0;
  }

  reset() {
    this.nextId = 1;
    this.tracks = [];
    this.uniqueCount = 0;
  }

  update(detections) {
    // Predict next box using constant-velocity shift
    for (const tr of this.tracks) {
      tr.box = {
        xmin: tr.box.xmin + tr.vx,
        ymin: tr.box.ymin + tr.vy,
        xmax: tr.box.xmax + tr.vx,
        ymax: tr.box.ymax + tr.vy,
      };
      tr.missed += 1;
    }

    const high = detections.filter(d => d.score >= this.highThresh);
    const low = detections.filter(d => d.score >= this.lowThresh && d.score < this.highThresh);

    const activeOut = [];
    const updatedIds = new Set();

    // Stage 1: match all tracks against high-confidence detections
    const s1 = greedyMatch(this.tracks, high, this.minIou);
    for (const [ti, di] of s1.matches) {
      this._applyUpdate(this.tracks[ti], high[di]);
      updatedIds.add(this.tracks[ti].id);
      activeOut.push(this.tracks[ti]);
    }

    // Stage 2: match remaining recently-active tracks against low-confidence detections
    const remTracks = s1.unmatchedT
      .map(i => this.tracks[i])
      .filter(t => t.missed <= 2);
    const s2 = greedyMatch(remTracks, low, 0.35);
    for (const [ti, di] of s2.matches) {
      this._applyUpdate(remTracks[ti], low[di]);
      updatedIds.add(remTracks[ti].id);
      activeOut.push(remTracks[ti]);
    }

    // Spawn new tracks for unmatched high-confidence detections
    for (const di of s1.unmatchedD) {
      const det = high[di];
      const id = this.nextId++;
      this.uniqueCount++;
      const cx = (det.box.xmin + det.box.xmax) * 0.5;
      const tr = {
        id,
        box: { ...det.box },
        label: det.label,
        score: det.score,
        vx: 0,
        vy: 0,
        missed: 0,
        trail: [[cx, det.box.ymax]],
      };
      this.tracks.push(tr);
      updatedIds.add(id);
      activeOut.push(tr);
    }

    this.tracks = this.tracks.filter(t => updatedIds.has(t.id) || t.missed <= this.maxAge);
    return activeOut;
  }

  _applyUpdate(tr, det) {
    const prevCx = (tr.box.xmin + tr.box.xmax) * 0.5;
    const prevCy = (tr.box.ymin + tr.box.ymax) * 0.5;
    const newCx = (det.box.xmin + det.box.xmax) * 0.5;
    const newCy = (det.box.ymin + det.box.ymax) * 0.5;
    tr.vx = 0.6 * tr.vx + 0.4 * (newCx - prevCx);
    tr.vy = 0.6 * tr.vy + 0.4 * (newCy - prevCy);
    tr.box = { ...det.box };
    tr.score = det.score;
    tr.label = det.label;
    tr.missed = 0;
    tr.trail.push([newCx, det.box.ymax]);
    if (tr.trail.length > 32) tr.trail.shift();
  }
}
