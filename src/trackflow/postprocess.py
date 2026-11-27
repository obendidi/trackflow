"""Offline post-processing for multi-object tracking outputs.

Implements suspect-transition track splitting (internal gaps, crossing overlaps, and
birth/death overlap junctions) followed by two-stage global CMC-kinematic and
sequence-calibrated appearance re-stitching via exact bipartite assignment, then
support- and motion-gated CMC-compensated short-gap interpolation, and finally gated
image-space smoothing of box jitter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

# Stage 3 gap interpolation gates.
_INTERP_MAX_GAP_S = 20.0 / 30.0  # longest filled gap without strong Re-ID verification, seconds
_INTERP_REID_MAX_GAP_S = 36.0 / 30.0  # extended gap when sequence-calibrated Re-ID confirms identity
_INTERP_ENDPOINT_CONF = 0.25
_INTERP_SUPPORT_CONF = 0.40  # mean confidence of the support observations on each side
_INTERP_SUPPORT_WINDOW = 5
_INTERP_MIN_SUPPORT = 3
_INTERP_TRAVEL_BASE = 0.5  # camera-compensated centre travel, box heights
_INTERP_TRAVEL_PER_FRAME = 0.08
_INTERP_MAX_BIDIR = 0.80  # forward/backward extrapolation disagreement at midpoint, box heights
_INTERP_REID_EXT_BIDIR = 0.65  # tighter kinematic consistency required for extended Re-ID gaps
_INTERP_MAX_HEIGHT_RATIO = 1.6
_INTERP_MIN_COS = 0.5
_INTERP_CONF_DECAY = 0.9
# Stage 1 within-track appearance bisection: min clean observations per side, local window,
# recursion depth, and required calibrated appearance distance (global and local contrast).
_BIS_MIN = 15
_BIS_WIN = 15
_BIS_DEPTH = 2
_BIS_GLOBAL = 0.85
_BIS_LOCAL = 0.85
# Stage 2 temporal prototypes: clean head/tail window length and blend weight in stitching.
_HT_WIN = 30
_HT_BLEND = 0.5
# A trusted-Re-ID link must be within this cosine of the best candidate of its source and target.
_MUTUAL_COS_TOL = 0.04
# Stage 4 image-space smoothing: adaptive two-scale Gaussian (seconds), max accepted shift (box heights).
_SMOOTH_SIGMA_LO_S = 1.0 / 30.0
_SMOOTH_SIGMA_HI_S = 4.2 / 30.0
_SMOOTH_MIN_FPS = 6.0
_SMOOTH_V_SCALE = 0.65  # max box-edge speed (box-heights/s) for strong smoothing
_SMOOTH_D_SCALE = 0.0150  # 4-edge multi-scale discrepancy (box heights) for strong smoothing
_SMOOTH_GATE = 0.15


@dataclass
class _Tracklet:
    """Internal representation of a single contiguous tracklet segment."""

    idx: int
    raw_id: int
    part_id: int
    class_id: int
    row_indices: np.ndarray  # indices into input `rows` (sorted by frame_idx)
    frames: np.ndarray  # int64 (L,)
    boxes: np.ndarray  # float64 (L, 4) xyxy
    confs: np.ndarray  # float64 (L,)
    ioa: np.ndarray  # float64 (L,) max IoA with other tracks in same frame
    iou: np.ndarray  # float64 (L,) max IoU with other tracks in same frame
    start_frame: int
    end_frame: int
    # Precomputed boundary states in their own boundary frame coordinates (start_frame / end_frame)
    p_start: np.ndarray  # float64 (2,)
    v_start: np.ndarray  # float64 (2,) px/frame in start_frame coords
    h_start: float
    n_start: int
    p_end: np.ndarray  # float64 (2,)
    v_end: np.ndarray  # float64 (2,) px/frame in end_frame coords
    h_end: float
    n_end: int
    emb_full: np.ndarray | None = None
    n_emb: int = 0
    emb_head: np.ndarray | None = None  # clean prototype of the first observations
    emb_tail: np.ndarray | None = None  # clean prototype of the last observations
    # Continuity prior with the immediately following split sibling (if any)
    next_sibling_idx: int = -1
    next_sibling_bonus: float = 0.0
    next_sibling_soft: bool = False


def _l2_norm_vec(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-8:
        return None
    return (v / n).astype(np.float32, copy=False)


def _compute_cumulative_cmc(
    num_frames: int,
    cmc_warps: np.ndarray | None,
    cmc_valid: np.ndarray | None,
    frame_width: int | None,
    frame_height: int | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build cumulative 3x3 homogeneous transforms W[f] mapping frame 0 -> frame f."""
    T = max(num_frames, 1)
    W = np.broadcast_to(np.eye(3, dtype=np.float64), (T, 3, 3)).copy()
    if cmc_warps is None or len(cmc_warps) == 0:
        return W, W.copy(), False

    max_shift = float(max(frame_width or 1920, frame_height or 1080)) * 0.45
    n_w = min(T, len(cmc_warps))
    has_valid = cmc_valid is not None
    any_applied = False
    for f in range(1, T):
        if f < n_w and (not has_valid or bool(cmc_valid[f])):
            w = cmc_warps[f]
            if w.shape == (2, 3) and np.isfinite(w).all():
                det = float(w[0, 0] * w[1, 1] - w[0, 1] * w[1, 0])
                shift = float(np.hypot(w[0, 2], w[1, 2]))
                if 0.85 <= det <= 1.15 and shift <= max_shift:
                    A = np.eye(3, dtype=np.float64)
                    A[:2, :] = w
                    W[f] = A @ W[f - 1]
                    any_applied = True
                    continue
        W[f] = W[f - 1]

    if not any_applied:
        return W, W.copy(), False

    W_inv = np.broadcast_to(np.eye(3, dtype=np.float64), (T, 3, 3)).copy()
    a = W[:, 0, 0]
    b = W[:, 0, 1]
    c = W[:, 1, 0]
    d = W[:, 1, 1]
    tx = W[:, 0, 2]
    ty = W[:, 1, 2]
    det = a * d - b * c
    safe = np.abs(det) > 1e-9
    inv_det = np.where(safe, 1.0 / np.where(safe, det, 1.0), 0.0)
    ia = d * inv_det
    ib = -b * inv_det
    ic = -c * inv_det
    id_ = a * inv_det
    W_inv[safe, 0, 0] = ia[safe]
    W_inv[safe, 0, 1] = ib[safe]
    W_inv[safe, 1, 0] = ic[safe]
    W_inv[safe, 1, 1] = id_[safe]
    W_inv[safe, 0, 2] = -(ia[safe] * tx[safe] + ib[safe] * ty[safe])
    W_inv[safe, 1, 2] = -(ic[safe] * tx[safe] + id_[safe] * ty[safe])
    return W, W_inv, True


def _warp_points(
    cx: np.ndarray,
    cy: np.ndarray,
    frames: np.ndarray,
    f_ref: int,
    W: np.ndarray,
    W_inv: np.ndarray,
    use_cmc: bool,
) -> np.ndarray:
    """Warp 2D points observed at `frames` into the coordinate frame `f_ref`."""
    n = len(frames)
    if not use_cmc or n == 0:
        return np.column_stack((cx, cy))
    Mref = W[f_ref]
    Ms = Mref[None, :, :] @ W_inv[frames]
    px = Ms[:, 0, 0] * cx + Ms[:, 0, 1] * cy + Ms[:, 0, 2]
    py = Ms[:, 1, 0] * cx + Ms[:, 1, 1] * cy + Ms[:, 1, 2]
    return np.column_stack((px, py))


def _compute_frame_overlaps(
    frames: np.ndarray,
    tids: np.ndarray,
    boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], list[tuple[int, float]]]]:
    """Compute per-observation max IoA, max IoU, and pairwise overlap frames (IoU >= 0.20)."""
    n = len(frames)
    max_ioa = np.zeros(n, dtype=np.float64)
    max_iou = np.zeros(n, dtype=np.float64)
    pair_overlaps: dict[tuple[int, int], list[tuple[int, float]]] = {}

    order = np.argsort(frames, kind="mergesort")
    f_sorted = frames[order]
    u_frames, starts, counts = np.unique(f_sorted, return_index=True, return_counts=True)

    for fr, s, c in zip(u_frames, starts, counts, strict=True):
        if c < 2:
            continue
        idxs = order[s : s + c]
        b = boxes[idxs]
        x1 = np.maximum(b[:, None, 0], b[None, :, 0])
        y1 = np.maximum(b[:, None, 1], b[None, :, 1])
        x2 = np.minimum(b[:, None, 2], b[None, :, 2])
        y2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        areas = np.maximum(1e-9, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
        ioa_m = inter / areas[:, None]
        iou_m = inter / np.maximum(areas[:, None] + areas[None, :] - inter, 1e-9)
        np.fill_diagonal(ioa_m, 0.0)
        np.fill_diagonal(iou_m, 0.0)
        max_ioa[idxs] = ioa_m.max(axis=1)
        max_iou[idxs] = iou_m.max(axis=1)

        ii, jj = np.where(np.triu(iou_m, k=1) >= 0.20)
        fr_i = int(fr)
        for a, b_i in zip(ii.tolist(), jj.tolist(), strict=True):
            ta = int(tids[idxs[a]])
            tb = int(tids[idxs[b_i]])
            if ta == tb:
                continue
            key = (ta, tb) if ta < tb else (tb, ta)
            pair_overlaps.setdefault(key, []).append((fr_i, float(iou_m[a, b_i])))

    return max_ioa, max_iou, pair_overlaps


def _clean_prototype(
    row_indices: np.ndarray,
    confs: np.ndarray,
    ioa: np.ndarray,
    embeddings: np.ndarray | None,
    emb_valid: np.ndarray | None,
) -> tuple[np.ndarray | None, int]:
    """Compute occlusion-downweighted, confidence-weighted L2-normalized mean embedding."""
    if embeddings is None or emb_valid is None or len(row_indices) == 0:
        return None, 0
    valid_mask = emb_valid[row_indices]
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return None, 0
    sel_rows = row_indices[valid_mask]
    sel_ioa = ioa[valid_mask]
    sel_conf = confs[valid_mask]
    embs = embeddings[sel_rows].astype(np.float32, copy=False)

    unocc = sel_ioa < 0.15
    if int(unocc.sum()) >= 2:
        w = np.where(unocc, np.clip(sel_conf, 0.2, 1.0), 0.05 * np.clip(sel_conf, 0.2, 1.0))
    else:
        w = np.clip(sel_conf, 0.2, 1.0) * np.square(1.0 - np.clip(sel_ioa, 0.0, 0.90))
    mean_v = np.sum(embs * w[:, None].astype(np.float32), axis=0)
    return _l2_norm_vec(mean_v), n_valid


def _head_tail_prototypes(
    row_indices: np.ndarray,
    ioa: np.ndarray,
    embeddings: np.ndarray | None,
    emb_valid: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Mean of the first and last `_HT_WIN` clean (unoccluded) embeddings of a tracklet."""
    if embeddings is None or emb_valid is None or len(row_indices) == 0:
        return None, None
    v = emb_valid[row_indices]
    clean = v & (ioa < 0.20)
    sel = row_indices[clean] if int(clean.sum()) >= 3 else row_indices[v]
    if len(sel) < 3:
        return None, None
    protos = []
    for part in (sel[:_HT_WIN], sel[-_HT_WIN:]):
        E = embeddings[part].astype(np.float32)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        protos.append(_l2_norm_vec(E.mean(axis=0)))
    return protos[0], protos[1]


def _calibrate_reid(
    raw_tracks: dict[int, np.ndarray],
    frames: np.ndarray,
    confs: np.ndarray,
    max_ioa: np.ndarray,
    embeddings: np.ndarray | None,
    emb_valid: np.ndarray | None,
) -> tuple[float, float, float, float]:
    """Estimate (reid_trust, mu_same, p50_diff, p70_diff) from concurrent vs within-track cosines."""
    if embeddings is None or emb_valid is None or not emb_valid.any():
        return 0.0, 0.88, 0.88, 0.88

    spans: list[tuple[int, int]] = []
    protos: list[np.ndarray] = []
    same_cos: list[float] = []

    for idxs in raw_tracks.values():
        v_mask = emb_valid[idxs]
        if int(v_mask.sum()) < 3:
            continue
        proto, n_emb = _clean_prototype(idxs, confs[idxs], max_ioa[idxs], embeddings, emb_valid)
        if proto is None:
            continue
        spans.append((int(frames[idxs[0]]), int(frames[idxs[-1]])))
        protos.append(proto)

        if n_emb >= 10:
            v_idxs = idxs[v_mask]
            clean_v = v_idxs[max_ioa[v_idxs] < 0.20]
            if len(clean_v) < 8:
                clean_v = v_idxs
            q = max(2, len(clean_v) // 3)
            e1 = _l2_norm_vec(embeddings[clean_v[:q]].astype(np.float32).mean(axis=0))
            e2 = _l2_norm_vec(embeddings[clean_v[-q:]].astype(np.float32).mean(axis=0))
            if e1 is not None and e2 is not None:
                same_cos.append(float(np.dot(e1, e2)))

    if len(protos) < 2:
        return 0.0, 0.88, 0.88, 0.88

    P = np.stack(protos, axis=0)
    sim = P @ P.T
    diff_cos: list[float] = []
    n_t = len(protos)
    for i in range(n_t):
        s0, e0 = spans[i]
        for j in range(i + 1, n_t):
            s1, e1 = spans[j]
            if min(e0, e1) - max(s0, s1) + 1 >= 5:
                diff_cos.append(float(sim[i, j]))

    if len(diff_cos) < 3:
        tri = sim[np.triu_indices(n_t, k=1)]
        if len(tri) == 0:
            return 0.0, 0.88, 0.88, 0.88
        diff_arr = tri
    else:
        diff_arr = np.asarray(diff_cos, dtype=np.float64)

    p50_diff = float(np.percentile(diff_arr, 50))
    p70_diff = float(np.percentile(diff_arr, 70))
    mu_same = float(np.median(same_cos)) if len(same_cos) >= 3 else 0.88
    mu_same = max(mu_same, 0.78)

    # Calibrate against the look-alike upper tail (p70_diff) and median (p50_diff):
    # uniform-costume scenes drop cleanly to trust=0; multi-appearance & sports scenes -> 0.7..1.0.
    sep_tail = mu_same - p70_diff
    sep_med = mu_same - p50_diff
    trust_tail = float(np.clip((sep_tail - 0.03) / 0.11, 0.0, 1.0))
    trust_med = float(np.clip((sep_med - 0.09) / 0.14, 0.0, 1.0))
    reid_trust = min(trust_tail, trust_med)
    return reid_trust, mu_same, p50_diff, p70_diff


def _fit_edge_state(
    frames: np.ndarray,
    boxes: np.ndarray,
    ioa: np.ndarray,
    W: np.ndarray,
    W_inv: np.ndarray,
    use_cmc: bool,
    f_ref: int,
    *,
    from_end: bool,
    win: int = 12,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Fit weighted linear position & velocity at tracklet boundary in frame `f_ref`."""
    L = len(frames)
    if from_end:
        sel = slice(max(0, L - win), L)
        f_edge = int(frames[-1])
    else:
        sel = slice(0, min(L, win))
        f_edge = int(frames[0])

    fs = frames[sel]
    bs = boxes[sel]
    ios = ioa[sel]
    cx = 0.5 * (bs[:, 0] + bs[:, 2])
    cy = 0.5 * (bs[:, 1] + bs[:, 3])
    hs = bs[:, 3] - bs[:, 1]
    h = max(float(np.median(hs)) if len(hs) > 0 else 1.0, 8.0)

    pts = _warp_points(cx, cy, fs, f_ref, W, W_inv, use_cmc)
    edge_pt = pts[-1] if from_end else pts[0]
    if len(fs) < 2 or int(fs[-1]) == int(fs[0]):
        return edge_pt, np.zeros(2, dtype=np.float64), h, 1

    t = (fs - f_edge).astype(np.float64)
    w_time = np.exp(-np.abs(t) / 6.0)
    w_occ = np.where(ios < 0.25, 1.0, 0.20)
    w = w_time * w_occ
    sw = float(np.sum(w))
    if sw <= 1e-8:
        return edge_pt, np.zeros(2, dtype=np.float64), h, len(fs)

    mt = float(np.sum(w * t)) / sw
    dt = t - mt
    var_t = float(np.sum(w * dt * dt))
    if var_t < 0.25:
        return edge_pt, np.zeros(2, dtype=np.float64), h, len(fs)

    mp = np.sum(w[:, None] * pts, axis=0) / sw
    vel = np.sum((w * dt)[:, None] * (pts - mp), axis=0) / var_t

    damp = min(1.0, (len(fs) - 1) / 4.0)
    vel = vel * damp
    speed = float(np.hypot(vel[0], vel[1]))
    max_speed = 0.65 * h
    if speed > max_speed:
        vel = vel * (max_speed / speed)

    pos_fit = mp - vel * mt
    edge_ioa = float(ios[-1] if from_end else ios[0])
    if edge_ioa >= 0.25 and len(fs) >= 4:
        pos = 0.7 * pos_fit + 0.3 * edge_pt
    else:
        pos = 0.35 * pos_fit + 0.65 * edge_pt
    return pos, vel, h, len(fs)


def _make_tracklet(
    idx: int,
    raw_id: int,
    part_id: int,
    class_id: int,
    seg_idxs: np.ndarray,
    frames: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    max_ioa: np.ndarray,
    max_iou: np.ndarray,
    W: np.ndarray,
    W_inv: np.ndarray,
    use_cmc: bool,
    embeddings: np.ndarray | None,
    emb_valid: np.ndarray | None,
) -> _Tracklet:
    """Construct a _Tracklet with precomputed head/tail CMC edge states and clean prototype."""
    seg_f = frames[seg_idxs]
    seg_b = boxes[seg_idxs]
    seg_c = confs[seg_idxs]
    seg_ioa = max_ioa[seg_idxs]
    seg_iou = max_iou[seg_idxs]
    f_start = int(seg_f[0])
    f_end = int(seg_f[-1])
    p_s, v_s, h_s, n_s = _fit_edge_state(
        seg_f, seg_b, seg_ioa, W, W_inv, use_cmc, f_start, from_end=False, win=12
    )
    p_e, v_e, h_e, n_e = _fit_edge_state(
        seg_f, seg_b, seg_ioa, W, W_inv, use_cmc, f_end, from_end=True, win=12
    )
    emb, n_emb = _clean_prototype(seg_idxs, seg_c, seg_ioa, embeddings, emb_valid)
    e_head, e_tail = _head_tail_prototypes(seg_idxs, seg_ioa, embeddings, emb_valid)
    return _Tracklet(
        idx=idx,
        raw_id=raw_id,
        part_id=part_id,
        class_id=class_id,
        row_indices=seg_idxs,
        frames=seg_f,
        boxes=seg_b,
        confs=seg_c,
        ioa=seg_ioa,
        iou=seg_iou,
        start_frame=f_start,
        end_frame=f_end,
        p_start=p_s,
        v_start=v_s,
        h_start=h_s,
        n_start=n_s,
        p_end=p_e,
        v_end=v_e,
        h_end=h_e,
        n_end=n_e,
        emb_full=emb,
        n_emb=n_emb,
        emb_head=e_head,
        emb_tail=e_tail,
    )


def _appearance_bisect(
    idxs: np.ndarray,
    max_ioa: np.ndarray,
    max_iou: np.ndarray,
    embeddings: np.ndarray,
    emb_valid: np.ndarray,
    mu_same: float,
    sep_tail: float,
) -> list[int]:
    """Split points (observation positions in `idxs`) where clean appearance changes abruptly.

    The clean (unoccluded) embeddings of a track are cut at the point that minimises the cosine
    between the pre- and post-cut mean embeddings. A cut is kept when that global contrast and a
    local window contrast around it are both far below the sequence's same-identity similarity;
    each accepted side is bisected again up to `_BIS_DEPTH` levels.
    """
    clean = np.nonzero(emb_valid[idxs] & (max_ioa[idxs] < 0.20))[0]
    m = _BIS_MIN
    if len(clean) < 2 * m:
        return []
    E = embeddings[idxs[clean]].astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    cuts: list[int] = []

    def _rec(lo: int, hi: int, depth: int) -> None:
        n = hi - lo
        if depth >= _BIS_DEPTH or n < 2 * m:
            return
        c = np.arange(m - 1, n - m)
        pre = np.cumsum(E[lo:hi], axis=0)[c].astype(np.float64)
        total = E[lo:hi].sum(axis=0, dtype=np.float64)
        pre_sq = np.einsum("ij,ij->i", pre, pre)
        pre_tot = pre @ total
        post_sq = np.maximum(float(total @ total) - 2.0 * pre_tot + pre_sq, 1e-12)
        cos = (pre_tot - pre_sq) / np.sqrt(np.maximum(pre_sq, 1e-12) * post_sq)
        b = int(np.argmin(cos))
        if (mu_same - float(cos[b])) / sep_tail < _BIS_GLOBAL:
            return
        cut = lo + int(c[b]) + 1  # first clean position of the post side
        w = _BIS_WIN
        e_pre = E[max(lo, cut - w) : cut].mean(axis=0)
        e_post = E[cut : min(hi, cut + w)].mean(axis=0)
        cos_loc = float(e_pre @ e_post) / max(
            float(np.linalg.norm(e_pre) * np.linalg.norm(e_post)), 1e-12
        )
        if (mu_same - cos_loc) / sep_tail < _BIS_LOCAL:
            return
        # Place the cut at the most overlapped observation between the two clean samples.
        a = int(clean[cut - 1])
        k = int(clean[cut])
        if k > a + 1:
            k = a + 2 + int(np.argmax(max_iou[idxs[a + 1 : k]]))
        cuts.append(k)
        _rec(lo, cut, depth + 1)
        _rec(cut, hi, depth + 1)

    _rec(0, len(clean), 0)
    return cuts


def _detect_split_points(
    raw_tracks: dict[int, np.ndarray],
    frames: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    max_ioa: np.ndarray,
    max_iou: np.ndarray,
    pair_overlaps: dict[tuple[int, int], list[tuple[int, float]]],
    W: np.ndarray,
    W_inv: np.ndarray,
    use_cmc: bool,
    embeddings: np.ndarray | None,
    emb_valid: np.ndarray | None,
    reid_trust: float,
    mu_same: float,
    p70_diff: float,
    crossing_swaps: list[tuple[float, int, int]],
    frame_stride: int,
    fps: float,
) -> dict[int, list[tuple[int, float, bool]]]:
    """Return per-raw-track list of (split_k, continuity_bonus, is_soft_split)."""
    splits: dict[int, dict[int, tuple[float, bool]]] = {}
    sep_tail = max(mu_same - p70_diff, 0.08)
    stride = max(1, int(frame_stride))
    # Stronger continuity prior when appearance cannot verify junction/crossing re-wirings
    b_junc = 0.20 * (1.0 - reid_trust) + 0.12 * reid_trust
    b_cross = b_junc + 0.02

    def _add_split(
        tid: int, k: int, bonus: float, is_soft: bool, length: int, min_side: int = 3
    ) -> None:
        if k < min_side or length - k < min_side:
            return
        d = splits.setdefault(tid, {})
        prev = d.get(k)
        if prev is None:
            d[k] = (bonus, is_soft)
        else:
            prev_b, prev_soft = prev
            if (not is_soft and prev_soft) or (is_soft == prev_soft and bonus < prev_b):
                d[k] = (bonus, is_soft)

    spans: dict[int, tuple[int, int, int]] = {
        tid: (int(frames[idxs[0]]), int(frames[idxs[-1]]), len(idxs))
        for tid, idxs in raw_tracks.items()
    }

    # --- Trigger A: Suspect internal gaps ---
    for tid, idxs in raw_tracks.items():
        L = len(idxs)
        if L < 4:
            continue
        fs = frames[idxs]
        bs = boxes[idxs]
        ioas = max_ioa[idxs]
        ious = max_iou[idxs]
        cs = confs[idxs]

        diffs = np.diff(fs)
        gap_ks = np.where(diffs >= 2 * stride)[0] + 1
        for k_val in gap_ks.tolist():
            if k_val < 2 or L - k_val < 2:
                continue
            df = int(fs[k_val] - fs[k_val - 1])
            f0 = int(fs[k_val - 1])
            f1 = int(fs[k_val])
            f_mid = (f0 + f1) // 2

            p0, v0, h0, _ = _fit_edge_state(
                fs[:k_val], bs[:k_val], ioas[:k_val], W, W_inv, use_cmc, f_mid, from_end=True, win=10
            )
            p1, v1, h1, _ = _fit_edge_state(
                fs[k_val:], bs[k_val:], ioas[k_val:], W, W_inv, use_cmc, f_mid, from_end=False, win=10
            )
            h = max(12.0, 0.5 * (h0 + h1))
            d_ep = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) / h
            tau = (0.5 * float(df)) / (1.0 + (float(df) / max(2.2 * fps, 25.0)) ** 2)
            d_bidir = float(
                np.hypot(
                    (p1[0] - v1[0] * tau) - (p0[0] + v0[0] * tau),
                    (p1[1] - v1[1] * tau) - (p0[1] + v0[1] * tau),
                )
            ) / h
            ov_gap = max(float(ious[k_val - 1]), float(ious[k_val]))

            app_drop = False
            if reid_trust >= 0.45 and embeddings is not None and emb_valid is not None:
                pre_sl = slice(max(0, k_val - 14), k_val)
                post_sl = slice(k_val, min(L, k_val + 14))
                e_pre, n_pre = _clean_prototype(
                    idxs[pre_sl], cs[pre_sl], ioas[pre_sl], embeddings, emb_valid
                )
                e_post, n_post = _clean_prototype(
                    idxs[post_sl], cs[post_sl], ioas[post_sl], embeddings, emb_valid
                )
                if e_pre is not None and e_post is not None and n_pre >= 2 and n_post >= 2:
                    cos_g = float(np.dot(e_pre, e_post))
                    app_dist = (mu_same - cos_g) / sep_tail
                    if app_dist >= 0.85 and cos_g < mu_same - 0.08:
                        app_drop = True

            extreme_teleport = (d_bidir >= 0.85) and (d_ep >= 0.85)
            if app_drop or extreme_teleport:
                _add_split(tid, k_val, 0.0, False, L, min_side=2)
            elif (d_bidir >= 0.30) or (d_ep >= 0.35 + 0.03 * df):
                _add_split(tid, k_val, 0.04, True, L, min_side=2)
            elif ov_gap >= 0.20 or df >= 3:
                _add_split(tid, k_val, 0.09, True, L, min_side=2)

    # --- Trigger B & C: Overlap junctions (T/Y birth/death junctions & surviving crossings) ---
    for (ta, tb), evs in pair_overlaps.items():
        if ta not in raw_tracks or tb not in raw_tracks:
            continue
        sa0, sa1, la = spans[ta]
        sb0, sb1, lb = spans[tb]
        if la < 4 or lb < 4:
            continue

        evs.sort(key=lambda x: x[0])
        episodes: list[list[tuple[int, float]]] = [[evs[0]]]
        for item in evs[1:]:
            if item[0] - episodes[-1][-1][0] <= 6:
                episodes[-1].append(item)
            else:
                episodes.append([item])

        ia = raw_tracks[ta]
        ib = raw_tracks[tb]
        fa = frames[ia]
        fb = frames[ib]

        for ep in episodes:
            ts = ep[0][0]
            te = ep[-1][0]
            peak_fr, peak_iou = max(ep, key=lambda x: x[1])
            ep_len = te - ts + 1

            # Trigger B: Track died or started right at this overlap -> soft junction split
            if ts - 3 <= sb1 <= te + 4 and sa0 < sb1 < sa1:
                ka = int(np.searchsorted(fa, sb1, side="right"))
                _add_split(ta, ka, b_junc, True, la, min_side=3)
            if ts - 4 <= sb0 <= te + 3 and sa0 < sb0 < sa1:
                ka = int(np.searchsorted(fa, sb0, side="left"))
                _add_split(ta, ka, b_junc, True, la, min_side=3)
            if ts - 3 <= sa1 <= te + 4 and sb0 < sa1 < sb1:
                kb = int(np.searchsorted(fb, sa1, side="right"))
                _add_split(tb, kb, b_junc, True, lb, min_side=3)
            if ts - 4 <= sa0 <= te + 3 and sb0 < sa0 < sb1:
                kb = int(np.searchsorted(fb, sa0, side="left"))
                _add_split(tb, kb, b_junc, True, lb, min_side=3)

            # Trigger C: Surviving 2-track crossing -> soft split on both at peak IoU
            if (
                ep_len <= 30
                and peak_iou >= 0.30
                and sa0 < ts - 2
                and sa1 > te + 2
                and sb0 < ts - 2
                and sb1 > te + 2
            ):
                # Compare the two possible continuations using motion on either
                # side of the overlap. Both fits use the peak's camera coordinates.
                sides = []
                for ids, fs in ((ia, fa), (ib, fb)):
                    pre = ids[np.searchsorted(fs, ts - 12 * stride):np.searchsorted(fs, ts)]
                    post = ids[np.searchsorted(fs, te + 1):np.searchsorted(fs, te + 1 + 12 * stride)]
                    sides.append((pre[max_iou[pre] < 0.2], post[max_iou[post] < 0.2]))
                if min(len(ids) for pair in sides for ids in pair) < 4:
                    continue
                states = []
                for pair in sides:
                    track_states = []
                    for end, ids in zip((True, False), pair, strict=True):
                        pos, vel, h, _ = _fit_edge_state(
                            frames[ids], boxes[ids], max_ioa[ids], W, W_inv, use_cmc,
                            peak_fr, from_end=end,
                        )
                        edge = frames[ids[-1 if end else 0]]
                        emb, _ = _clean_prototype(ids, confs[ids], max_ioa[ids], embeddings, emb_valid)
                        track_states.append((pos, vel, h, edge, emb))
                    states.append(track_states)

                costs = np.empty((2, 2))
                for a in range(2):
                    p0, v0, h0, f0, e0 = states[a][0]
                    for b in range(2):
                        p1, v1, h1, f1, e1 = states[b][1]
                        gap = f1 - f0
                        delta = p1 - p0
                        h = 0.5 * (h0 + h1)
                        costs[a, b] = (
                            np.linalg.norm(delta - gap * v0) + np.linalg.norm(delta - gap * v1)
                        ) / (2 * h)
                        if e0 is not None and e1 is not None:
                            costs[a, b] += reid_trust * (1 - e0 @ e1) / sep_tail
                gain = min(costs[0, 0] - costs[0, 1], costs[1, 1] - costs[1, 0])
                if gain <= 0.08 or max(costs[0, 1], costs[1, 0]) > 1.2:
                    continue
                ka = int(np.searchsorted(fa, peak_fr, side="right"))
                kb = int(np.searchsorted(fb, peak_fr, side="right"))
                _add_split(ta, ka, b_cross, True, la, min_side=4)
                _add_split(tb, kb, b_cross, True, lb, min_side=4)
                crossing_swaps.append((gain, int(ia[ka - 1]), int(ib[kb - 1])))

    # --- Trigger D: Strong calibrated appearance shift across an overlap region ---
    if reid_trust >= 0.50 and embeddings is not None and emb_valid is not None:
        for tid, idxs in raw_tracks.items():
            L = len(idxs)
            if L < 14:
                continue
            ious = max_iou[idxs]
            ov_pos = np.where(ious >= 0.25)[0]
            if len(ov_pos) == 0:
                continue
            runs: list[tuple[int, int]] = []
            r_s = int(ov_pos[0])
            r_e = r_s
            for p in ov_pos[1:].tolist():
                if p <= r_e + 4:
                    r_e = p
                else:
                    runs.append((r_s, r_e))
                    r_s = p
                    r_e = p
            runs.append((r_s, r_e))

            ioas = max_ioa[idxs]
            cs = confs[idxs]
            for r_s, r_e in runs:
                if r_e - r_s > 30 or r_s < 5 or L - 1 - r_e < 5:
                    continue
                pre_sl = slice(max(0, r_s - 15), r_s)
                post_sl = slice(r_e + 1, min(L, r_e + 16))
                e_pre, n_pre = _clean_prototype(
                    idxs[pre_sl], cs[pre_sl], ioas[pre_sl], embeddings, emb_valid
                )
                e_post, n_post = _clean_prototype(
                    idxs[post_sl], cs[post_sl], ioas[post_sl], embeddings, emb_valid
                )
                if e_pre is not None and e_post is not None and n_pre >= 3 and n_post >= 3:
                    cos_ov = float(np.dot(e_pre, e_post))
                    app_dist = (mu_same - cos_ov) / sep_tail
                    if app_dist >= 0.95 and cos_ov < mu_same - 0.09:
                        peak_k = r_s + int(np.argmax(ious[r_s : r_e + 1])) + 1
                        _add_split(tid, peak_k, 0.0, False, L, min_side=4)

    # --- Trigger E: Within-track appearance bisection (undetected mid-track switches) ---
    if reid_trust >= 0.50 and embeddings is not None and emb_valid is not None:
        for tid, idxs in raw_tracks.items():
            for k in _appearance_bisect(
                idxs, max_ioa, max_iou, embeddings, emb_valid, mu_same, sep_tail
            ):
                _add_split(tid, k, 0.0, False, len(idxs), min_side=4)

    # Coalesce split points closer than 4 observations on the same track
    out: dict[int, list[tuple[int, float, bool]]] = {}
    for tid, kmap in splits.items():
        items = sorted(kmap.items(), key=lambda x: x[0])
        merged: list[tuple[int, float, bool]] = []
        for k, (b, is_soft) in items:
            if merged and k - merged[-1][0] < 4:
                prev_k, prev_b, prev_soft = merged[-1]
                if (not is_soft and prev_soft) or (is_soft == prev_soft and b < prev_b):
                    merged[-1] = (k, b, is_soft)
            else:
                merged.append((k, b, is_soft))
        if merged:
            out[tid] = merged
    return out


def _build_tracklets(
    raw_tracks: dict[int, np.ndarray],
    split_points: dict[int, list[tuple[int, float, bool]]],
    frames: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    max_ioa: np.ndarray,
    max_iou: np.ndarray,
    class_ids: np.ndarray | None,
    W: np.ndarray,
    W_inv: np.ndarray,
    use_cmc: bool,
    embeddings: np.ndarray | None,
    emb_valid: np.ndarray | None,
) -> list[_Tracklet]:
    """Slice raw tracks at split_points into initial tracklets with precomputed edge states."""
    tracklets: list[_Tracklet] = []
    for tid in sorted(raw_tracks):
        idxs = raw_tracks[tid]
        L = len(idxs)
        cls_id = int(class_ids[idxs[0]]) if class_ids is not None else 0
        spts = split_points.get(tid, [])
        bounds = [0] + [k for k, _, _ in spts] + [L]
        meta_s = [(b, soft) for _, b, soft in spts]

        first_tidx = len(tracklets)
        for p_i in range(len(bounds) - 1):
            lo, hi = bounds[p_i], bounds[p_i + 1]
            if hi <= lo:
                continue
            tracklets.append(
                _make_tracklet(
                    len(tracklets),
                    tid,
                    p_i,
                    cls_id,
                    idxs[lo:hi],
                    frames,
                    boxes,
                    confs,
                    max_ioa,
                    max_iou,
                    W,
                    W_inv,
                    use_cmc,
                    embeddings,
                    emb_valid,
                )
            )
        n_parts = len(tracklets) - first_tidx
        for p_i in range(n_parts - 1):
            t_cur = tracklets[first_tidx + p_i]
            t_nxt = tracklets[first_tidx + p_i + 1]
            t_cur.next_sibling_idx = t_nxt.idx
            t_cur.next_sibling_bonus = meta_s[p_i][0]
            t_cur.next_sibling_soft = meta_s[p_i][1]
    return tracklets


def _merge_chains(
    tracklets: list[_Tracklet],
    links: dict[int, int],
    frames: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    max_ioa: np.ndarray,
    max_iou: np.ndarray,
    W: np.ndarray,
    W_inv: np.ndarray,
    use_cmc: bool,
    embeddings: np.ndarray | None,
    emb_valid: np.ndarray | None,
) -> list[_Tracklet]:
    """Merge linked tracklets along directed time-ordered paths into new tracklets."""
    if not links:
        return tracklets

    has_pred = set(links.values())
    new_tracklets: list[_Tracklet] = []
    visited: set[int] = set()

    for t in tracklets:
        if t.idx in has_pred or t.idx in visited:
            continue
        chain: list[_Tracklet] = []
        cur: int | None = t.idx
        while cur is not None and cur not in visited:
            visited.add(cur)
            chain.append(tracklets[cur])
            cur = links.get(cur)

        if len(chain) == 1:
            nt = chain[0]
            nt.idx = len(new_tracklets)
            nt.next_sibling_idx = -1
            nt.next_sibling_bonus = 0.0
            nt.next_sibling_soft = False
            new_tracklets.append(nt)
            continue

        row_idxs = np.concatenate([c.row_indices for c in chain])
        new_tracklets.append(
            _make_tracklet(
                len(new_tracklets),
                chain[0].raw_id,
                chain[0].part_id,
                chain[0].class_id,
                row_idxs,
                frames,
                boxes,
                confs,
                max_ioa,
                max_iou,
                W,
                W_inv,
                use_cmc,
                embeddings,
                emb_valid,
            )
        )

    for t in tracklets:
        if t.idx not in visited:
            t.idx = len(new_tracklets)
            t.next_sibling_idx = -1
            t.next_sibling_bonus = 0.0
            t.next_sibling_soft = False
            new_tracklets.append(t)

    return new_tracklets


def _stitch_pass(
    tracklets: list[_Tracklet],
    W: np.ndarray,
    W_inv: np.ndarray,
    use_cmc: bool,
    fps: float,
    reid_trust: float,
    mu_same: float,
    p70_diff: float,
    *,
    max_gap_frames: int,
    max_cost: float,
    use_sibling_bonus: bool,
    margin_gate: float,
    margin_apply_above: float,
) -> dict[int, int]:
    """Solve global 1-to-1 forward-in-time tracklet linking using vectorized scoring + Hungarian."""
    n = len(tracklets)
    if n < 2:
        return {}

    sep_tail = max(mu_same - p70_diff, 0.08)
    cos_floor = max(0.68, p70_diff - 0.01)
    fps_safe = max(fps, 1.0)
    tau_mid_scale = max(2.2 * fps_safe, 25.0)
    tau_slow_cap = 0.30 * fps_safe

    start_f = np.array([t.start_frame for t in tracklets], dtype=np.int64)
    end_f = np.array([t.end_frame for t in tracklets], dtype=np.int64)
    cls_arr = np.array([t.class_id for t in tracklets], dtype=np.int64)
    p_starts = np.stack([t.p_start for t in tracklets], axis=0)
    v_starts = np.stack([t.v_start for t in tracklets], axis=0)
    h_starts = np.array([t.h_start for t in tracklets], dtype=np.float64)
    n_starts = np.array([t.n_start for t in tracklets], dtype=np.int64)
    p_ends = np.stack([t.p_end for t in tracklets], axis=0)
    v_ends = np.stack([t.v_end for t in tracklets], axis=0)
    h_ends = np.array([t.h_end for t in tracklets], dtype=np.float64)
    n_ends = np.array([t.n_end for t in tracklets], dtype=np.int64)

    has_emb = np.array([t.emb_full is not None for t in tracklets], dtype=bool)
    emb_dim = next((t.emb_full.shape[0] for t in tracklets if t.emb_full is not None), 0)
    emb_mat = np.zeros((n, emb_dim), dtype=np.float32) if emb_dim > 0 else None
    if emb_mat is not None:
        for i, t in enumerate(tracklets):
            if t.emb_full is not None:
                emb_mat[i] = t.emb_full

        head_mat = np.zeros((n, emb_dim), dtype=np.float32)
        tail_mat = np.zeros((n, emb_dim), dtype=np.float32)
        for i, t in enumerate(tracklets):
            if t.emb_head is not None:
                head_mat[i] = t.emb_head
                tail_mat[i] = t.emb_tail

    order_by_start = np.argsort(start_f, kind="mergesort")
    sorted_starts = start_f[order_by_start]

    edges: dict[tuple[int, int], float] = {}
    edge_cos: dict[tuple[int, int], float] = {}
    best_src_cos = np.full(n, -np.inf)
    best_dst_cos = np.full(n, -np.inf)
    active_src: set[int] = set()
    active_dst: set[int] = set()

    for i in range(n):
        t_a = tracklets[i]
        f0 = int(end_f[i])
        lo = int(np.searchsorted(sorted_starts, f0 + 1, side="left"))
        hi = int(np.searchsorted(sorted_starts, f0 + max_gap_frames, side="right"))
        if lo >= hi:
            continue

        cand_j = order_by_start[lo:hi]
        if cls_arr[i] != 0 or np.any(cls_arr[cand_j] != cls_arr[i]):
            cand_j = cand_j[cls_arr[cand_j] == cls_arr[i]]
            if len(cand_j) == 0:
                continue

        f1s = start_f[cand_j]
        dts = (f1s - f0).astype(np.float64)
        dt_s = dts / fps_safe
        f_mids = (f0 + f1s) // 2

        h0 = h_ends[i]
        h1s = h_starts[cand_j]
        h_ratio = np.maximum(h0 / h1s, h1s / h0)
        hs = np.maximum(12.0, 0.5 * (h0 + h1s))

        if use_cmc:
            W_mids = W[f_mids]
            M0 = W_mids @ W_inv[f0]
            p0 = np.empty((len(cand_j), 2), dtype=np.float64)
            p0[:, 0] = M0[:, 0, 0] * p_ends[i, 0] + M0[:, 0, 1] * p_ends[i, 1] + M0[:, 0, 2]
            p0[:, 1] = M0[:, 1, 0] * p_ends[i, 0] + M0[:, 1, 1] * p_ends[i, 1] + M0[:, 1, 2]
            v0 = np.empty((len(cand_j), 2), dtype=np.float64)
            v0[:, 0] = M0[:, 0, 0] * v_ends[i, 0] + M0[:, 0, 1] * v_ends[i, 1]
            v0[:, 1] = M0[:, 1, 0] * v_ends[i, 0] + M0[:, 1, 1] * v_ends[i, 1]

            M1 = W_mids @ W_inv[f1s]
            ps1 = p_starts[cand_j]
            vs1 = v_starts[cand_j]
            p1 = np.empty((len(cand_j), 2), dtype=np.float64)
            p1[:, 0] = M1[:, 0, 0] * ps1[:, 0] + M1[:, 0, 1] * ps1[:, 1] + M1[:, 0, 2]
            p1[:, 1] = M1[:, 1, 0] * ps1[:, 0] + M1[:, 1, 1] * ps1[:, 1] + M1[:, 1, 2]
            v1 = np.empty((len(cand_j), 2), dtype=np.float64)
            v1[:, 0] = M1[:, 0, 0] * vs1[:, 0] + M1[:, 0, 1] * vs1[:, 1]
            v1[:, 1] = M1[:, 1, 0] * vs1[:, 0] + M1[:, 1, 1] * vs1[:, 1]
        else:
            p0 = np.broadcast_to(p_ends[i], (len(cand_j), 2))
            v0 = np.broadcast_to(v_ends[i], (len(cand_j), 2))
            p1 = p_starts[cand_j]
            v1 = v_starts[cand_j]

        dp = p1 - p0
        d_ep = np.hypot(dp[:, 0], dp[:, 1]) / hs

        tau_mid = (0.5 * dts) / (1.0 + np.square(dts / tau_mid_scale))
        pred0 = p0 + v0 * tau_mid[:, None]
        pred1 = p1 - v1 * tau_mid[:, None]
        d_bidir = np.hypot(pred1[:, 0] - pred0[:, 0], pred1[:, 1] - pred0[:, 1]) / hs

        tau_slow = np.minimum(0.5 * dts, tau_slow_cap)
        ps0 = p0 + v0 * tau_slow[:, None]
        ps1_b = p1 - v1 * tau_slow[:, None]
        d_slow = np.hypot(ps1_b[:, 0] - ps0[:, 0], ps1_b[:, 1] - ps0[:, 1]) / hs
        d_geom = np.minimum(d_bidir, d_slow + 0.15)

        d_vel = np.zeros(len(cand_j), dtype=np.float64)
        n0 = int(n_ends[i])
        n1s = n_starts[cand_j]
        if n0 >= 3:
            m_ge2 = (n1s >= 3) & (dts >= 2.0)
            if np.any(m_ge2):
                v_avg = dp[m_ge2] / dts[m_ge2, None]
                sc = np.minimum(dts[m_ge2], 8.0) / hs[m_ge2]
                dv0 = np.hypot(v0[m_ge2, 0] - v_avg[:, 0], v0[m_ge2, 1] - v_avg[:, 1]) * sc
                dv1 = np.hypot(v1[m_ge2, 0] - v_avg[:, 0], v1[m_ge2, 1] - v_avg[:, 1]) * sc
                d_vel[m_ge2] = 0.5 * (dv0 + dv1)
            if n0 >= 4:
                m_eq1 = (n1s >= 4) & (dts == 1.0)
                if np.any(m_eq1):
                    d_vel[m_eq1] = (
                        np.hypot(v1[m_eq1, 0] - v0[m_eq1, 0], v1[m_eq1, 1] - v0[m_eq1, 1])
                        * 3.5
                        / hs[m_eq1]
                    )

        # Balanced kinematic cost: midpoint extrapolation + bounded endpoint anchor + velocity agreement
        kin_scale = 1.0 + (0.30 + 0.20 * reid_trust) * np.sqrt(dt_s)
        c_kin = (
            (0.82 * d_geom + 0.18 * np.minimum(d_ep, d_geom + 1.2)) / kin_scale
            + 0.22 * d_vel
            + 0.06 * np.abs(np.log(h0 / h1s))
            + 0.06 * np.log1p(dt_s)
        )

        max_ep = 1.50 + (3.6 + 1.4 * reid_trust) * dt_s
        max_geom = 1.05 + (1.25 + 1.0 * reid_trust) * dt_s
        valid = (h_ratio <= 2.3) & (d_ep <= max_ep) & (d_geom <= max_geom)

        costs = c_kin.copy()
        cos_all = np.full(len(cand_j), np.nan)
        if reid_trust >= 0.08 and emb_mat is not None and has_emb[i]:
            both_emb = has_emb[cand_j]
            if np.any(both_emb):
                cos = (emb_mat[cand_j[both_emb]] @ emb_mat[i]).astype(np.float64)
                # Credit a clean tail-of-i to head-of-j match that the full-track mean blurs.
                cos_th = (head_mat[cand_j[both_emb]] @ tail_mat[i]).astype(np.float64)
                cos += _HT_BLEND * np.maximum(cos_th - cos, 0.0)
                app_dist = (mu_same - cos) / sep_tail
                w_kin = 1.0 - 0.52 * reid_trust
                w_app = 0.55 * reid_trust
                app_term = np.maximum(-0.18, app_dist)
                costs[both_emb] = w_kin * c_kin[both_emb] + w_app * app_term
                cos_all[both_emb] = cos
                if reid_trust >= 0.45:
                    veto = (app_dist > 1.00) | (cos < cos_floor)
                    idx_b = np.where(both_emb)[0]
                    valid[idx_b[veto]] = False

        # Continuity prior for split siblings (soft splits are never orphaned)
        sib_j = t_a.next_sibling_idx if use_sibling_bonus else -1
        if sib_j >= 0:
            sib_pos = np.where(cand_j == sib_j)[0]
            if len(sib_pos) == 1:
                sp = int(sib_pos[0])
                costs[sp] -= t_a.next_sibling_bonus
                if t_a.next_sibling_soft:
                    valid[sp] = True
                    if costs[sp] > max_cost - 0.02:
                        costs[sp] = max_cost - 0.02

        keep = np.where(valid & (costs < max_cost))[0]
        if len(keep) == 0:
            continue
        active_src.add(i)
        for k_pos in keep.tolist():
            j = int(cand_j[k_pos])
            edges[(i, j)] = float(costs[k_pos])
            active_dst.add(j)
            c_ij = float(cos_all[k_pos])
            if np.isfinite(c_ij):
                edge_cos[(i, j)] = c_ij
                best_src_cos[i] = max(best_src_cos[i], c_ij)
                best_dst_cos[j] = max(best_dst_cos[j], c_ij)

    if not edges:
        return {}

    src_list = sorted(active_src)
    dst_list = sorted(active_dst)
    src_pos = {u: idx for idx, u in enumerate(src_list)}
    dst_pos = {v: idx for idx, v in enumerate(dst_list)}
    nr = len(src_list)
    nc = len(dst_list)

    big = max_cost + 100.0
    mat = np.full((nr, nc + nr), big, dtype=np.float64)
    for (u, v), c in edges.items():
        mat[src_pos[u], dst_pos[v]] = c
    mat[np.arange(nr), nc + np.arange(nr)] = max_cost

    row_ind, col_ind = linear_sum_assignment(mat)
    links: dict[int, int] = {}

    for r_i, c_i in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
        if c_i >= nc:
            continue
        cost = float(mat[r_i, c_i])
        if cost >= max_cost:
            continue
        u = src_list[r_i]
        v = dst_list[c_i]

        is_sib = use_sibling_bonus and tracklets[u].next_sibling_idx == v
        if margin_gate > 0.0 and not is_sib and cost > margin_apply_above:
            row_vals = mat[r_i, :nc]
            col_vals = mat[:, c_i]
            alt_row = float(np.min(np.delete(row_vals, c_i))) if nc > 1 else big
            alt_col = float(np.min(np.delete(col_vals, r_i))) if nr > 1 else big
            if min(alt_row, alt_col) - cost < margin_gate:
                continue

        # Mutual-best appearance: reject a link when either end has a clearly better-looking
        # valid candidate that the assignment gave elsewhere.
        c_uv = edge_cos.get((u, v))
        if (
            reid_trust >= 0.50
            and c_uv is not None
            and c_uv < max(best_src_cos[u], best_dst_cos[v]) - _MUTUAL_COS_TOL
        ):
            continue
        links[u] = v

    if use_sibling_bonus:
        used_dst = set(links.values())
        for t in tracklets:
            if t.next_sibling_soft and t.next_sibling_idx >= 0:
                u = t.idx
                v = t.next_sibling_idx
                if u not in links and v not in used_dst:
                    links[u] = v
                    used_dst.add(v)

    return links


def _side_velocity(frames: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Least-squares velocity (px/frame) of a short run of points, zero if degenerate."""
    if len(frames) < 2:
        return np.zeros(2, dtype=np.float64)
    t = frames - frames.mean()
    denom = float(np.dot(t, t))
    if denom <= 0.0:
        return np.zeros(2, dtype=np.float64)
    return (t[:, None] * (pts - pts.mean(axis=0))).sum(axis=0) / denom


def _gated_cmc_interpolate(
    tracklets: list[_Tracklet],
    W: np.ndarray,
    W_inv: np.ndarray,
    use_cmc: bool,
    fps: float,
    frame_stride: int,
    frame_width: int | None,
    frame_height: int | None,
    embeddings: np.ndarray | None,
    emb_valid: np.ndarray | None,
    reid_trust: float = 0.0,
    mu_same: float = 0.88,
    p70_diff: float = 0.88,
) -> np.ndarray:
    """Fill short gaps whose both sides are well supported and kinematically consistent.

    A gap is filled when each side has at least `_INTERP_MIN_SUPPORT` nearby observations with
    mean confidence `>= _INTERP_SUPPORT_CONF`, both endpoints have confidence
    `>= _INTERP_ENDPOINT_CONF`, the camera-compensated travel and bidirectional velocity
    extrapolations agree, and (when embeddings exist) both sides look alike. When calibrated
    Re-ID is trustworthy and strongly confirms identity across both sides of a gap, gaps up to
    `_INTERP_REID_MAX_GAP_S` are eligible under tighter kinematic checks. Centres are
    interpolated in the coordinates of the gap's last frame and mapped back to each filled frame.
    """
    stride = max(1, int(frame_stride))
    max_gap_base = max(2 * stride, int(round(_INTERP_MAX_GAP_S * fps)))
    can_extend = reid_trust >= 0.45 and embeddings is not None
    max_gap = (
        max(max_gap_base, int(round(_INTERP_REID_MAX_GAP_S * fps)))
        if can_extend
        else max_gap_base
    )
    reid_confirm_cos = max(p70_diff + 0.08, mu_same - 0.10)
    reid_veto_cos = max(_INTERP_MIN_COS, p70_diff - 0.04) if reid_trust >= 0.50 else _INTERP_MIN_COS
    win = _INTERP_SUPPORT_WINDOW
    out: list[np.ndarray] = []

    for assigned_id, t in enumerate(tracklets, start=1):
        L = len(t.frames)
        if L < 2 * _INTERP_MIN_SUPPORT:
            continue
        fr = t.frames
        diffs = np.diff(fr)
        gap_idxs = np.nonzero((diffs > stride) & (diffs - 1 <= max_gap))[0]
        for g in gap_idxs.tolist():
            f0 = int(fr[g])
            f1 = int(fr[g + 1])
            span = f1 - f0
            is_extended = (span - 1) > max_gap_base
            c0 = float(t.confs[g])
            c1 = float(t.confs[g + 1])
            if min(c0, c1) < (_INTERP_ENDPOINT_CONF + 0.05 if is_extended else _INTERP_ENDPOINT_CONF):
                continue
            before = np.arange(max(0, g + 1 - win), g + 1)
            after = np.arange(g + 1, min(L, g + 1 + win))
            before = before[fr[before] >= f0 - 2 * stride * win]
            after = after[fr[after] <= f1 + 2 * stride * win]
            if len(before) < _INTERP_MIN_SUPPORT or len(after) < _INTERP_MIN_SUPPORT:
                continue
            min_sup_conf = _INTERP_SUPPORT_CONF + 0.05 if is_extended else _INTERP_SUPPORT_CONF
            if (
                float(t.confs[before].mean()) < min_sup_conf
                or float(t.confs[after].mean()) < min_sup_conf
            ):
                continue
            b0 = t.boxes[g]
            b1 = t.boxes[g + 1]
            h0 = float(b0[3] - b0[1])
            h1 = float(b1[3] - b1[1])
            if h0 <= 1.0 or h1 <= 1.0 or max(h0, h1) / min(h0, h1) > _INTERP_MAX_HEIGHT_RATIO:
                continue
            hm = 0.5 * (h0 + h1)

            side = np.concatenate((before, after))
            sb = t.boxes[side]
            pts = _warp_points(
                0.5 * (sb[:, 0] + sb[:, 2]),
                0.5 * (sb[:, 1] + sb[:, 3]),
                fr[side],
                f1,
                W,
                W_inv,
                use_cmc,
            )
            pb = pts[: len(before)]
            pa = pts[len(before) :]
            p0 = pb[-1]
            p1 = pa[0]
            travel = float(np.hypot(*(p1 - p0))) / hm
            travel_rate = 0.065 if is_extended else _INTERP_TRAVEL_PER_FRAME
            if travel > _INTERP_TRAVEL_BASE + travel_rate * span:
                continue
            v0 = _side_velocity(fr[before].astype(np.float64), pb)
            v1 = _side_velocity(fr[after].astype(np.float64), pa)
            half = 0.5 * span
            bidir = float(np.hypot(*((p0 + v0 * half) - (p1 - v1 * half)))) / hm
            max_bidir = _INTERP_REID_EXT_BIDIR if is_extended else _INTERP_MAX_BIDIR
            if bidir > max_bidir:
                continue

            # Use a slightly wider occlusion-filtered window for Re-ID verification across gaps
            emb_b = np.arange(max(0, g + 1 - 10), g + 1)
            emb_a = np.arange(g + 1, min(L, g + 1 + 10))
            e0, _ = _clean_prototype(
                t.row_indices[emb_b], t.confs[emb_b], t.ioa[emb_b], embeddings, emb_valid
            )
            e1, _ = _clean_prototype(
                t.row_indices[emb_a], t.confs[emb_a], t.ioa[emb_a], embeddings, emb_valid
            )
            if is_extended:
                if e0 is None or e1 is None or float(e0 @ e1) < reid_confirm_cos:
                    continue
            elif e0 is not None and e1 is not None and float(e0 @ e1) < reid_veto_cos:
                continue

            fill = np.arange(f0 + stride, f1, stride, dtype=np.int64)
            if len(fill) == 0:
                continue
            s = ((fill - f0) / float(span))[:, None]
            centers = p0[None] + s * (p1 - p0)[None]
            if use_cmc:
                M = W[fill] @ W_inv[f1][None, :, :]
                centers = np.einsum("nij,nj->ni", M[:, :2, :2], centers) + M[:, :2, 2]
            wh0 = b0[2:4] - b0[0:2]
            wh1 = b1[2:4] - b1[0:2]
            wh = wh0[None] + s * (wh1 - wh0)[None]
            bx = np.concatenate((centers - 0.5 * wh, centers + 0.5 * wh), axis=1)
            if frame_width is not None:
                bx[:, [0, 2]] = np.clip(bx[:, [0, 2]], 0.0, float(frame_width))
            if frame_height is not None:
                bx[:, [1, 3]] = np.clip(bx[:, [1, 3]], 0.0, float(frame_height))
            keep = ((bx[:, 2] - bx[:, 0]) >= 1.0) & ((bx[:, 3] - bx[:, 1]) >= 1.0)
            keep &= np.isfinite(bx).all(axis=1)
            if not keep.any():
                continue
            rows = np.empty((int(keep.sum()), 7), dtype=np.float64)
            rows[:, 0] = fill[keep]
            rows[:, 1] = float(assigned_id)
            rows[:, 2:6] = bx[keep]
            rows[:, 6] = min(c0, c1) * _INTERP_CONF_DECAY
            out.append(rows)

    if not out:
        return np.zeros((0, 7), dtype=np.float64)
    return np.concatenate(out, axis=0)


def _smooth_tracks(
    out: np.ndarray,
    fps: float,
    frame_stride: int,
    is_interp: np.ndarray | None = None,
) -> np.ndarray:
    """Motion- and curvature-adaptive two-scale Gaussian smoothing in image coordinates.

    Every contiguous run starts from light Gaussian smoothing (`_SMOOTH_SIGMA_LO_S`, with a
    low-fps frame-space floor) and smoothly blends toward stronger smoothing
    (`_SMOOTH_SIGMA_HI_S`) on low-speed, low-curvature frames measured across all four box
    edges (`x1, y1, x2, y2`), so steady targets lose jitter while fast articulated motion stays
    sharp. A smoothed value is only accepted if it stays within `_SMOOTH_GATE` box heights.
    """
    stride = max(1, int(frame_stride))
    fps_eff = float(fps) / stride
    if fps_eff < _SMOOTH_MIN_FPS or len(out) == 0:
        return out
    sigma_lo = max(0.55, _SMOOTH_SIGMA_LO_S * fps_eff)
    sigma_hi = max(1.80, _SMOOTH_SIGMA_HI_S * fps_eff)

    order = np.lexsort((out[:, 0], out[:, 1]))
    srt = out[order]
    interp_srt = is_interp[order] if is_interp is not None else np.zeros(len(srt), dtype=bool)
    tid = srt[:, 1]
    fr = np.rint(srt[:, 0]).astype(np.int64)
    brk = np.nonzero((np.diff(tid) != 0) | (np.diff(fr) != stride))[0] + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [len(srt)]))

    b = srt[:, 2:6]
    sm = b.copy()

    for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
        L = e - s
        if L < 5:
            continue
        seg = b[s:e]
        s_lo = gaussian_filter1d(seg, sigma_lo, axis=0, mode="nearest")
        s_hi = gaussian_filter1d(seg, sigma_hi, axis=0, mode="nearest")

        h_ref = np.maximum(s_lo[:, 3] - s_lo[:, 1], 1.0)
        edge_v = (np.abs(np.gradient(s_lo, axis=0)).max(axis=1) / h_ref) * fps_eff
        d_hl = np.abs(s_hi - s_lo).max(axis=1) / h_ref
        # Local neighborhood smoothing prevents turning on strong smoothing at inflection points
        edge_v_loc = gaussian_filter1d(edge_v, sigma_lo, mode="nearest")
        d_hl_loc = gaussian_filter1d(d_hl, sigma_lo, mode="nearest")

        w_v = 1.0 / (1.0 + np.power(edge_v_loc / _SMOOTH_V_SCALE, 4))
        w_d = 1.0 / (1.0 + np.power(d_hl_loc / _SMOOTH_D_SCALE, 4))
        w_hi = w_v * w_d

        seg_interp = interp_srt[s:e]
        if seg_interp.any():
            seam = seg_interp.copy()
            seam[:-1] |= seg_interp[1:]
            seam[1:] |= seg_interp[:-1]
            w_seam = (
                0.50
                / (1.0 + np.power(edge_v_loc / (1.3 * _SMOOTH_V_SCALE), 4))
                / (1.0 + np.power(d_hl_loc / (1.5 * _SMOOTH_D_SCALE), 4))
            )
            w_hi = np.where(seam, np.maximum(w_hi, w_seam), w_hi)

        sm[s:e] = (1.0 - w_hi[:, None]) * s_lo + w_hi[:, None] * s_hi

    h = np.maximum(b[:, 3] - b[:, 1], 1.0)
    # Match baseline gate on [cx, cy, w, h]
    st_raw = np.column_stack(
        (0.5 * (b[:, 0] + b[:, 2]), 0.5 * (b[:, 1] + b[:, 3]), b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    )
    st_sm = np.column_stack(
        (0.5 * (sm[:, 0] + sm[:, 2]), 0.5 * (sm[:, 1] + sm[:, 3]), sm[:, 2] - sm[:, 0], sm[:, 3] - sm[:, 1])
    )
    dev = np.abs(st_sm - st_raw).max(axis=1) / h
    use = dev <= _SMOOTH_GATE
    nb = np.where(use[:, None], sm, b)
    ok = np.isfinite(nb).all(axis=1) & (nb[:, 2] - nb[:, 0] >= 1.0) & (nb[:, 3] - nb[:, 1] >= 1.0)
    srt[ok, 2:6] = nb[ok]
    res = out.copy()
    res[order] = srt
    return res


def postprocess_tracks_array(
    rows: np.ndarray,
    *,
    fps: float = 30.0,
    frame_width: int | None = None,
    frame_height: int | None = None,
    embeddings: np.ndarray | None = None,
    emb_valid: np.ndarray | None = None,
    cmc_warps: np.ndarray | None = None,
    cmc_valid: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    frame_stride: int = 1,
) -> np.ndarray:
    """Offline post-processing: suspect-transition splitting + global CMC/Re-ID re-stitching.

    `rows` has shape (N, 7) float64: [frame_idx (0-based), track_id, x1, y1, x2, y2, conf].
    Returns float64 array of shape (M, 7) with valid boxes and unique (frame_idx, track_id).
    """
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    if len(arr) == 0:
        return arr.copy()

    frames = np.rint(arr[:, 0]).astype(np.int64)
    tids = np.rint(arr[:, 1]).astype(np.int64)
    boxes = arr[:, 2:6]
    confs = arr[:, 6]

    num_frames = int(frames.max()) + 1
    if cmc_warps is not None:
        num_frames = max(num_frames, len(cmc_warps))

    # Stage 0: Cumulative CMC, per-frame overlaps, and sequence-level Re-ID calibration
    W, W_inv, use_cmc = _compute_cumulative_cmc(
        num_frames, cmc_warps, cmc_valid, frame_width, frame_height
    )
    max_ioa, max_iou, pair_overlaps = _compute_frame_overlaps(frames, tids, boxes)

    order = np.lexsort((frames, tids))
    t_sorted = tids[order]
    u_tids, t_starts, t_counts = np.unique(t_sorted, return_index=True, return_counts=True)
    raw_tracks: dict[int, np.ndarray] = {
        int(tid): order[s : s + c]
        for tid, s, c in zip(u_tids, t_starts, t_counts, strict=True)
    }

    reid_trust, mu_same, _p50_diff, p70_diff = _calibrate_reid(
        raw_tracks, frames, confs, max_ioa, embeddings, emb_valid
    )

    # Stage 1: Suspect-transition track splitting
    crossing_swaps: list[tuple[float, int, int]] = []
    split_points = _detect_split_points(
        raw_tracks,
        frames,
        boxes,
        confs,
        max_ioa,
        max_iou,
        pair_overlaps,
        W,
        W_inv,
        use_cmc,
        embeddings,
        emb_valid,
        reid_trust,
        mu_same,
        p70_diff,
        crossing_swaps,
        frame_stride=frame_stride,
        fps=fps,
    )
    tracklets = _build_tracklets(
        raw_tracks,
        split_points,
        frames,
        boxes,
        confs,
        max_ioa,
        max_iou,
        class_ids,
        W,
        W_inv,
        use_cmc,
        embeddings,
        emb_valid,
    )

    # A cut may have coalesced with another split. Rewire only intact
    # crossing pairs, resolving simultaneous proposals by their cost margin.
    tails = {int(t.row_indices[-1]): t for t in tracklets}
    paired = set()
    for _, ra, rb in sorted(crossing_swaps, reverse=True):
        if ra not in tails or rb not in tails or ra in paired or rb in paired:
            continue
        a, b = tails[ra], tails[rb]
        if a.class_id != b.class_id or min(a.next_sibling_idx, b.next_sibling_idx) < 0:
            continue
        a.next_sibling_idx, b.next_sibling_idx = b.next_sibling_idx, a.next_sibling_idx
        a.next_sibling_bonus = b.next_sibling_bonus = 1.0
        a.next_sibling_soft = b.next_sibling_soft = True
        paired.update((ra, rb))

    # Stage 2, Pass 1 (Local Repair): resolve crossing swaps, Y/T-junction hijacks & restore soft splits
    p1_gap = max(20, int(round(0.9 * fps)))
    p1_max_cost = 0.54 + 0.18 * reid_trust
    links_p1 = _stitch_pass(
        tracklets,
        W,
        W_inv,
        use_cmc,
        fps,
        reid_trust,
        mu_same,
        p70_diff,
        max_gap_frames=p1_gap,
        max_cost=p1_max_cost,
        use_sibling_bonus=True,
        margin_gate=0.06,
        margin_apply_above=0.22,
    )
    tracklets = _merge_chains(
        tracklets,
        links_p1,
        frames,
        boxes,
        confs,
        max_ioa,
        max_iou,
        W,
        W_inv,
        use_cmc,
        embeddings,
        emb_valid,
    )

    # Stage 2, Pass 2 & 3 (Global Long-Range Stitching): operate on repaired full-length tracklets
    long_gap_s = 6.0 if reid_trust >= 0.50 else 3.2
    p2_max_cost = 0.54 + 0.28 * reid_trust
    # With trusted Re-ID, commit medium-range links before the long-range pass.
    horizons = [2.5, long_gap_s] if reid_trust >= 0.50 else [long_gap_s, long_gap_s]
    for horizon_s in horizons:
        long_gap = max(75, int(round(horizon_s * fps)))
        links_p2 = _stitch_pass(
            tracklets,
            W,
            W_inv,
            use_cmc,
            fps,
            reid_trust,
            mu_same,
            p70_diff,
            max_gap_frames=long_gap,
            max_cost=p2_max_cost,
            use_sibling_bonus=False,
            margin_gate=0.09 if reid_trust < 0.35 else 0.07,
            margin_apply_above=0.22,
        )
        if not links_p2:
            break
        tracklets = _merge_chains(
            tracklets,
            links_p2,
            frames,
            boxes,
            confs,
            max_ioa,
            max_iou,
            W,
            W_inv,
            use_cmc,
            embeddings,
            emb_valid,
        )

    # Reconstruct output rows with clean contiguous 1-based track IDs
    tracklets.sort(key=lambda t: (t.start_frame, t.raw_id, t.part_id))
    out = arr.copy()
    new_tid = np.empty(len(arr), dtype=np.int64)
    for assigned_id, t in enumerate(tracklets, start=1):
        new_tid[t.row_indices] = assigned_id
    out[:, 1] = new_tid.astype(np.float64)

    # Stage 3: Strictly gated short-gap CMC interpolation
    interp_rows = _gated_cmc_interpolate(
        tracklets,
        W,
        W_inv,
        use_cmc,
        fps,
        frame_stride,
        frame_width,
        frame_height,
        embeddings,
        emb_valid,
        reid_trust=reid_trust,
        mu_same=mu_same,
        p70_diff=p70_diff,
    )
    is_interp = np.zeros(len(out), dtype=bool)
    if len(interp_rows) > 0:
        out = np.vstack((out, interp_rows))
        is_interp = np.concatenate((is_interp, np.ones(len(interp_rows), dtype=bool)))

    # Stage 4: Motion- and curvature-adaptive image-space smoothing
    out = _smooth_tracks(out, fps, frame_stride, is_interp=is_interp)

    out_order = np.lexsort((out[:, 1], out[:, 0]))
    return out[out_order]


__all__ = ["postprocess_tracks_array"]
