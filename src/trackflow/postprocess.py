"""Offline post-processing for multi-object tracking outputs.

Implements suspect-transition track splitting (internal gaps, crossing overlaps, and
birth/death overlap junctions) followed by two-stage global CMC-kinematic and
sequence-calibrated appearance re-stitching via exact bipartite assignment, plus
strictly gated CMC-compensated short-gap interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


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
    )


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
                ka = int(np.searchsorted(fa, peak_fr, side="right"))
                kb = int(np.searchsorted(fb, peak_fr, side="right"))
                _add_split(ta, ka, b_cross, True, la, min_side=4)
                _add_split(tb, kb, b_cross, True, lb, min_side=4)

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

    order_by_start = np.argsort(start_f, kind="mergesort")
    sorted_starts = start_f[order_by_start]

    edges: dict[tuple[int, int], float] = {}
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
        if reid_trust >= 0.08 and emb_mat is not None and has_emb[i]:
            both_emb = has_emb[cand_j]
            if np.any(both_emb):
                cos = (emb_mat[cand_j[both_emb]] @ emb_mat[i]).astype(np.float64)
                app_dist = (mu_same - cos) / sep_tail
                w_kin = 1.0 - 0.52 * reid_trust
                w_app = 0.55 * reid_trust
                app_term = np.maximum(-0.18, app_dist)
                costs[both_emb] = w_kin * c_kin[both_emb] + w_app * app_term
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


def _gated_cmc_interpolate(
    tracklets: list[_Tracklet],
    W: np.ndarray,
    W_inv: np.ndarray,
    use_cmc: bool,
    fps: float,
    frame_stride: int,
    frame_width: int | None,
    frame_height: int | None,
) -> np.ndarray:
    """Fill only clean, high-confidence, unoccluded short gaps in CMC-compensated coordinates."""
    stride = max(1, int(frame_stride))
    max_gap = max(8, int(round(0.50 * fps)))
    extra: list[tuple[float, float, float, float, float, float, float]] = []

    for assigned_id, t in enumerate(tracklets, start=1):
        if len(t.frames) < 8:
            continue
        diffs = np.diff(t.frames)
        gap_idxs = np.where((diffs >= 2 * stride) & (diffs <= max_gap))[0]
        for g in gap_idxs.tolist():
            c0 = float(t.confs[g])
            c1 = float(t.confs[g + 1])
            if c0 < 0.60 or c1 < 0.60:
                continue
            if float(t.ioa[g]) >= 0.20 or float(t.ioa[g + 1]) >= 0.20:
                continue

            f0 = int(t.frames[g])
            f1 = int(t.frames[g + 1])
            b0 = t.boxes[g]
            b1 = t.boxes[g + 1]
            h0 = float(b0[3] - b0[1])
            h1 = float(b1[3] - b1[1])
            w0 = float(b0[2] - b0[0])
            w1 = float(b1[2] - b1[0])
            if h0 <= 1.0 or h1 <= 1.0 or max(h0 / h1, h1 / h0) > 1.35:
                continue

            h = max(12.0, 0.5 * (h0 + h1))
            c0p = np.array([0.5 * (b0[0] + b0[2]), 0.5 * (b0[1] + b0[3])], dtype=np.float64)
            c1p = np.array([0.5 * (b1[0] + b1[2]), 0.5 * (b1[1] + b1[3])], dtype=np.float64)
            if use_cmc:
                M01 = W[f1] @ W_inv[f0]
                c0w = M01[:2, :2] @ c0p + M01[:2, 2]
            else:
                c0w = c0p
            if float(np.hypot(c1p[0] - c0w[0], c1p[1] - c0w[1])) / h > 1.15:
                continue

            base_conf = min(c0, c1) * 0.85
            inv_span = 1.0 / float(f1 - f0)
            for f in range(f0 + stride, f1, stride):
                alpha = float(f - f0) * inv_span
                if use_cmc:
                    M0f = W[f] @ W_inv[f0]
                    M1f = W[f] @ W_inv[f1]
                    p_a = M0f[:2, :2] @ c0p + M0f[:2, 2]
                    p_b = M1f[:2, :2] @ c1p + M1f[:2, 2]
                    p_f = (1.0 - alpha) * p_a + alpha * p_b
                else:
                    p_f = (1.0 - alpha) * c0p + alpha * c1p
                wf = (1.0 - alpha) * w0 + alpha * w1
                hf = (1.0 - alpha) * h0 + alpha * h1
                x1 = float(p_f[0] - 0.5 * wf)
                y1 = float(p_f[1] - 0.5 * hf)
                x2 = float(p_f[0] + 0.5 * wf)
                y2 = float(p_f[1] + 0.5 * hf)
                if frame_width is not None:
                    x1 = max(0.0, min(float(frame_width) - 1.0, x1))
                    x2 = max(x1 + 1.0, min(float(frame_width), x2))
                if frame_height is not None:
                    y1 = max(0.0, min(float(frame_height) - 1.0, y1))
                    y2 = max(y1 + 1.0, min(float(frame_height), y2))
                if x2 > x1 and y2 > y1 and np.isfinite((x1, y1, x2, y2)).all():
                    extra.append((float(f), float(assigned_id), x1, y1, x2, y2, base_conf))

    if not extra:
        return np.zeros((0, 7), dtype=np.float64)
    return np.asarray(extra, dtype=np.float64).reshape(-1, 7)


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
    long_gap = max(75, int(round(long_gap_s * fps)))
    p2_max_cost = 0.54 + 0.28 * reid_trust
    for _ in range(2):
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
        tracklets, W, W_inv, use_cmc, fps, frame_stride, frame_width, frame_height
    )
    if len(interp_rows) > 0:
        out = np.vstack((out, interp_rows))

    out_order = np.lexsort((out[:, 1], out[:, 0]))
    return out[out_order]


__all__ = ["postprocess_tracks_array"]
