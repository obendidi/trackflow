/**
 * ByteTrack for the browser playground: a direct port of `ByteTracker` in
 * `src/trackflow/tracker.py` (Kalman filter, Hungarian matching, two-stage association,
 * track confirmation, spawn suppression, duplicate removal).
 *
 * Keep the two in sync: `tests/test_web_parity.py` replays the same recorded detections
 * through both implementations and requires identical IDs and boxes.
 *
 * Detections: { box: {xmin, ymin, xmax, ymax}, score, label, classId }
 */

const STD_POS = 1 / 20;
const STD_VEL = 1 / 160;

function boxIoU(a, b) {
  const iw = Math.max(0, Math.min(a.xmax, b.xmax) - Math.max(a.xmin, b.xmin));
  const ih = Math.max(0, Math.min(a.ymax, b.ymax) - Math.max(a.ymin, b.ymin));
  const inter = iw * ih;
  const areaA = Math.max(0, a.xmax - a.xmin) * Math.max(0, a.ymax - a.ymin);
  const areaB = Math.max(0, b.xmax - b.xmin) * Math.max(0, b.ymax - b.ymin);
  return inter / Math.max(areaA + areaB - inter, 1e-6);
}

/** Fraction of box `a` covered by box `b`. */
function boxIoA(a, b) {
  const iw = Math.max(0, Math.min(a.xmax, b.xmax) - Math.max(a.xmin, b.xmin));
  const ih = Math.max(0, Math.min(a.ymax, b.ymax) - Math.max(a.ymin, b.ymin));
  const areaA = Math.max(Math.max(0, a.xmax - a.xmin) * Math.max(0, a.ymax - a.ymin), 1e-6);
  return (iw * ih) / areaA;
}

function xyxyToXyah(b) {
  const w = Math.max(1, b.xmax - b.xmin);
  const h = Math.max(1, b.ymax - b.ymin);
  return [b.xmin + 0.5 * w, b.ymin + 0.5 * h, w / h, h];
}

function xyahToXyxy(m) {
  const a = Math.max(0.05, m[2]);
  const h = Math.max(1, m[3]);
  const w = a * h;
  return { xmin: m[0] - 0.5 * w, ymin: m[1] - 0.5 * h, xmax: m[0] + 0.5 * w, ymax: m[1] + 0.5 * h };
}

/**
 * Minimum-cost assignment for a rectangular cost matrix (Hungarian / Kuhn-Munkres with
 * potentials). Returns [row, col] pairs covering min(n, m) rows, like scipy's
 * `linear_sum_assignment`.
 */
export function linearSumAssignment(cost) {
  const n0 = cost.length;
  const m0 = n0 ? cost[0].length : 0;
  if (!n0 || !m0) return [];
  const transpose = n0 > m0;
  const n = transpose ? m0 : n0;
  const m = transpose ? n0 : m0;
  const at = transpose ? (i, j) => cost[j][i] : (i, j) => cost[i][j];

  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1); // p[j] = row assigned to column j (1-based)
  const way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(Infinity);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = at(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else { minv[j] -= delta; }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  const pairs = [];
  for (let j = 1; j <= m; j++) {
    if (p[j]) pairs.push(transpose ? [j - 1, p[j] - 1] : [p[j] - 1, j - 1]);
  }
  pairs.sort((a, b) => a[0] - b[0]);
  return pairs;
}

function linearAssignment(cost, maxCost, nRows, nCols) {
  const matches = [];
  const rowsUsed = new Set();
  const colsUsed = new Set();
  if (nRows && nCols) {
    const transpose = nRows > nCols;
    const rows = Math.min(nRows, nCols), cols = Math.max(nRows, nCols);
    const extended = Array.from({ length: rows }, (_, r) =>
      Array.from({ length: cols + rows }, (_, c) =>
        c >= cols ? maxCost + Number.EPSILON : (transpose ? cost[c][r] : cost[r][c])));
    for (const [row, col] of linearSumAssignment(extended)) {
      if (col >= cols) continue;
      const [r, c] = transpose ? [col, row] : [row, col];
      if (cost[r][c] <= maxCost) {
        matches.push([r, c]);
        rowsUsed.add(r);
        colsUsed.add(c);
      }
    }
  }
  const unmatchedRows = [];
  for (let r = 0; r < nRows; r++) if (!rowsUsed.has(r)) unmatchedRows.push(r);
  const unmatchedCols = [];
  for (let c = 0; c < nCols; c++) if (!colsUsed.has(c)) unmatchedCols.push(c);
  return { matches, unmatchedRows, unmatchedCols };
}

/** Grow a box by `scale` x its width/height on every side (buffered IoU). */
function expandBox(b, scale) {
  if (scale <= 0) return b;
  const w = b.xmax - b.xmin;
  const h = b.ymax - b.ymin;
  return { xmin: b.xmin - scale * w, ymin: b.ymin - scale * h, xmax: b.xmax + scale * w, ymax: b.ymax + scale * h };
}

/**
 * Buffered-IoU scale for the time between processed frames (same rule as
 * `iou_buffer_for_interval` in tracker.py): 0 up to 0.1 s, 0.3 at 0.3 s, capped at 0.5.
 * Keeps IDs from fragmenting when inference runs at a few FPS (e.g. WASM).
 */
export function iouBufferForInterval(dtSec) {
  return Math.min(0.5, Math.max(0, 1.5 * (dtSec - 0.1)));
}

/** 8-state constant-velocity Kalman filter over [cx, cy, a, h, vx, vy, va, vh]. */
class KalmanBoxFilter {
  constructor(box) {
    const meas = xyxyToXyah(box);
    this.mean = [...meas, 0, 0, 0, 0];
    const h = Math.max(10, meas[3]);
    const std = [
      2 * STD_POS * h, 2 * STD_POS * h, 1e-2, 2 * STD_POS * h,
      10 * STD_VEL * h, 10 * STD_VEL * h, 1e-5, 10 * STD_VEL * h,
    ];
    this.cov = Array.from({ length: 8 }, (_, i) =>
      Array.from({ length: 8 }, (_, j) => (i === j ? std[i] * std[i] : 0)));
  }

  predict() {
    const h = Math.max(10, this.mean[3]);
    const std = [STD_POS * h, STD_POS * h, 1e-2, STD_POS * h, STD_VEL * h, STD_VEL * h, 1e-5, STD_VEL * h];
    for (let i = 0; i < 4; i++) this.mean[i] += this.mean[i + 4];
    if (this.mean[3] < 1) { this.mean[3] = 1; this.mean[7] = 0; }
    // P = F P F^T + Q with F = [[I, I], [0, I]]
    const P = this.cov;
    const FP = P.map((row, i) => (i < 4 ? row.map((v, j) => v + P[i + 4][j]) : row.slice()));
    this.cov = FP.map(row => row.map((v, j) => (j < 4 ? v + row[j + 4] : v)));
    for (let i = 0; i < 8; i++) this.cov[i][i] += std[i] * std[i];
  }

  update(box) {
    const meas = xyxyToXyah(box);
    const h = Math.max(10, this.mean[3]);
    const r = [STD_POS * h, STD_POS * h, 1e-1, STD_POS * h].map(x => x * x);
    const P = this.cov;
    // S = H P H^T + R (top-left 4x4 block), HP = first 4 rows of P
    const S = [0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => P[i][j] + (i === j ? r[i] : 0)));
    const HP = [0, 1, 2, 3].map(i => P[i].slice());
    const Kt = solve4(S, HP); // 4x8, K = Kt^T
    const innov = [0, 1, 2, 3].map(i => meas[i] - this.mean[i]);
    for (let i = 0; i < 8; i++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += Kt[k][i] * innov[k];
      this.mean[i] += s;
    }
    // P -= K S K^T
    const KS = Array.from({ length: 8 }, (_, i) =>
      [0, 1, 2, 3].map(j => { let s = 0; for (let k = 0; k < 4; k++) s += Kt[k][i] * S[k][j]; return s; }));
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += KS[i][k] * Kt[k][j];
        P[i][j] -= s;
      }
    }
  }

  get box() {
    return xyahToXyxy(this.mean);
  }
}

/** Solve A X = B for X with A 4x4 and B 4xN (Gaussian elimination, partial pivoting). */
function solve4(A, B) {
  const a = A.map(row => row.slice());
  const b = B.map(row => row.slice());
  const n = 4;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    [a[col], a[piv]] = [a[piv], a[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = col + 1; r < n; r++) {
      const f = a[r][col] / a[col][col];
      for (let c = col; c < n; c++) a[r][c] -= f * a[col][c];
      for (let c = 0; c < b[r].length; c++) b[r][c] -= f * b[col][c];
    }
  }
  const x = b.map(row => row.map(() => 0));
  for (let r = n - 1; r >= 0; r--) {
    for (let c = 0; c < b[r].length; c++) {
      let s = b[r][c];
      for (let k = r + 1; k < n; k++) s -= a[r][k] * x[k][c];
      x[r][c] = s / a[r][r];
    }
  }
  return x;
}

export class ByteTracker {
  constructor({
    highThresh = 0.5,
    lowThresh = 0.15,
    newTrackThresh = 0.55,
    matchThresh = 0.8,
    secondMatchThresh = 0.5,
    unconfirmedMatchThresh = 0.7,
    trackBuffer = 30,
    spawnOverlapThresh = 0.8,
    duplicateIou = 0.85,
    iouBuffer = 0,
  } = {}) {
    Object.assign(this, {
      highThresh, lowThresh, newTrackThresh, matchThresh, secondMatchThresh,
      unconfirmedMatchThresh, trackBuffer, spawnOverlapThresh, duplicateIou, iouBuffer,
    });
    this.reset();
  }

  reset() {
    this.nextId = 1;
    this.nextUid = 1;
    this.frameId = 0;
    this.tracks = [];
  }

  /** Number of public IDs handed out so far. */
  get uniqueCount() {
    return this.nextId - 1;
  }

  _applyDetection(tr, det, updateLabel) {
    tr.kf.update(det.box);
    tr.score = det.score;
    if (updateLabel) { tr.label = det.label; tr.classId = det.classId; }
    tr.hits += 1;
    tr.tsu = 0;
  }

  _cost(tracks, dets, buffer = this.iouBuffer) {
    const boxes = dets.map(d => expandBox(d.box, buffer));
    return tracks.map(t => {
      const box = expandBox(t.kf.box, buffer);
      return boxes.map((b, i) => t.classId === dets[i].classId ? 1 - boxIoU(box, b) : Infinity);
    });
  }

  _activate(tr) {
    tr.isActivated = true;
    tr.id = this.nextId++;
  }

  /** Advance one frame; returns the confirmed tracks observed in this frame. */
  update(detections) {
    this.frameId += 1;

    for (const t of this.tracks) {
      t.recoveredNow = false;
      if (t.state !== 'tracked') t.kf.mean[7] = 0;
      t.kf.predict();
      t.age += 1;
      t.tsu += 1;
    }

    const high = detections.filter(d => d.score >= this.highThresh);
    const low = detections.filter(d => d.score >= this.lowThresh && d.score < this.highThresh);
    const confirmed = this.tracks.filter(t => t.isActivated && t.state === 'tracked');
    const lost = this.tracks.filter(t => t.state === 'lost');
    const unconfirmed = this.tracks.filter(t => !t.isActivated);
    const pool = [...confirmed, ...lost];
    const updated = new Set();

    // 1. confirmed + lost tracks <-> high-score detections (buffered IoU when iouBuffer > 0)
    const s1 = linearAssignment(this._cost(pool, high), this.matchThresh, pool.length, high.length);
    for (const [ti, di] of s1.matches) {
      const tr = pool[ti];
      if (tr.state === 'lost') tr.recoveredNow = true;
      tr.state = 'tracked';
      this._applyDetection(tr, high[di], true);
      updated.add(tr.uid);
    }

    // 2. remaining tracked tracks <-> low-score detections
    const remTracked = s1.unmatchedRows.map(i => pool[i]).filter(t => t.state === 'tracked');
    if (remTracked.length && low.length) {
      const s2 = linearAssignment(
        this._cost(remTracked, low, 0),
        this.secondMatchThresh, remTracked.length, low.length);
      for (const [ti, di] of s2.matches) {
        this._applyDetection(remTracked[ti], low[di], false);
        updated.add(remTracked[ti].uid);
      }
    }

    // 3. unconfirmed tracks <-> leftover high-score detections
    const remHigh = s1.unmatchedCols.map(i => high[i]);
    const s3 = linearAssignment(
      this._cost(unconfirmed, remHigh), this.unconfirmedMatchThresh, unconfirmed.length, remHigh.length);
    for (const [ti, di] of s3.matches) {
      this._applyDetection(unconfirmed[ti], remHigh[di], true);
      updated.add(unconfirmed[ti].uid);
      this._activate(unconfirmed[ti]);
    }

    // 4. lost transitions, expiry, removal of unmatched unconfirmed tracks
    const surviving = [];
    for (const tr of this.tracks) {
      if (updated.has(tr.uid)) { surviving.push(tr); continue; }
      if (!tr.isActivated) continue;
      if (tr.state === 'tracked') tr.state = 'lost';
      if (tr.tsu <= this.trackBuffer) surviving.push(tr);
    }
    this.tracks = surviving;

    // 5. spawn unconfirmed tracks from leftover high-score detections
    const occupied = this.tracks.filter(t => updated.has(t.uid));
    const leftovers = s3.unmatchedCols.map(i => remHigh[i]).sort((a, b) => b.score - a.score);
    for (const det of leftovers) {
      if (det.score < this.newTrackThresh) continue;
      if (occupied.some(t => t.classId === det.classId && boxIoA(det.box, t.kf.box) >= this.spawnOverlapThresh)) continue;
      const tr = {
        uid: this.nextUid++,
        id: 0,
        kf: new KalmanBoxFilter(det.box),
        score: det.score,
        label: det.label,
        classId: det.classId,
        startFrame: this.frameId,
        isActivated: false,
        age: 1,
        hits: 1,
        tsu: 0,
        state: 'tracked',
        recoveredNow: false,
      };
      this.tracks.push(tr);
      updated.add(tr.uid);
      occupied.push(tr);
      if (this.frameId === 1) this._activate(tr);
    }

    // 6. tracked/lost duplicates: the longer-lived track wins
    this._removeDuplicates();

    return this.tracks
      .filter(t => t.isActivated && updated.has(t.uid))
      .map(t => this._output(t, false))
      .sort((a, b) => a.id - b.id);
  }

  _removeDuplicates() {
    const tracked = this.tracks.filter(t => t.state === 'tracked');
    const lost = this.tracks.filter(t => t.state === 'lost');
    if (!tracked.length || !lost.length) return;
    const drop = new Set();
    for (const a of tracked) {
      for (const b of lost) {
        if (a.classId === b.classId && boxIoU(a.kf.box, b.kf.box) > this.duplicateIou) {
          drop.add(this.frameId - a.startFrame > this.frameId - b.startFrame ? b.uid : a.uid);
        }
      }
    }
    if (drop.size) this.tracks = this.tracks.filter(t => !drop.has(t.uid));
  }

  /** Confirmed tracks missed for <= maxFrames frames, at their predicted position (display only). */
  coastingTracks(maxFrames = 3) {
    return this.tracks
      .filter(t => t.isActivated && t.state === 'lost' && t.tsu <= maxFrames)
      .map(t => this._output(t, true))
      .sort((a, b) => a.id - b.id);
  }

  _output(t, coasting) {
    return {
      id: t.id,
      box: t.kf.box,
      score: t.score,
      label: t.label,
      classId: t.classId,
      age: t.age,
      hits: t.hits,
      recovered: t.recoveredNow && !coasting,
      coasting,
    };
  }
}
