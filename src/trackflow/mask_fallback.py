"""Ambiguity-triggered selective mask fallback (pure NumPy/SciPy/OpenCV core).

Monitors a box tracker online using a bounded in-RAM frame ring buffer, detects
ambiguous occlusion/collision windows with backward clean-seed lookup, dispatches
only high-value windows to a pluggable :class:`MaskPropagator` (such as Apache-2.0
SAM 2.1 / EdgeTAM in :mod:`trackflow.sam2_adapter` or an external SAM 3 predictor
wrapped via :class:`Sam3MaskPropagator`), and reconciles identities using a
collision-free bijective permutation.

Core ``MaskFallbackResolver`` never writes frames to disk and has zero PyTorch,
CUDA, Transformers, or Meta-licensed dependencies so core ``trackflow-python``
stays lightweight and 100% Apache-2.0.
"""

from __future__ import annotations

import bisect
import contextlib
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from trackflow.tracker import compute_ioa_matrix, compute_iou_matrix
from trackflow.types import Track

BBox = tuple[float, float, float, float]
FRESH_ID_OFFSET = 1_000_000_000


class MaskPropagator(Protocol):
    """Pluggable mask propagation backend for short ambiguity windows.

    ``frames_rgb`` holds consecutive RGB uint8 frames for ``[window.start_frame .. window.end_frame]``.
    ``seeds`` maps ``seed_track_id -> (local_frame_idx, bbox_xyxy_orig_pixels)``.
    Returns ``{local_frame_idx: {seed_track_id: mask_bool}}``, where each ``mask_bool`` is a 2D
    boolean array at either full frame resolution or a smaller native mask grid (e.g. 256x256);
    box coordinates are automatically scaled to match ``mask_bool.shape``.
    """

    def propagate_window(
        self,
        frames_rgb: Sequence[np.ndarray],
        seeds: dict[int, tuple[int, BBox]],
        frame_size_wh: tuple[int, int],
    ) -> dict[int, dict[int, np.ndarray]]: ...


@dataclass(frozen=True, slots=True)
class MaskFallbackConfig:
    """Configuration for online ambiguity detection and two-tier safe reconciliation."""

    # Stage A: collision & occlusion window detection
    iou_overlap_thresh: float = 0.60
    occlusion_ioa_thresh: float = 0.65
    min_track_age: int = 6
    min_track_conf: float = 0.50
    lookback_frames: int = 4
    clean_exit_frames: int = 2
    max_window_len: int = 16
    max_cluster_tracks: int = 3
    cooldown_frames: int = 5
    clean_seed_max_iou: float = 0.20

    # Stage B: zero-cost causal CPU gate & optional frame budget
    # Conservative default: pure transient-occlusion recovery (zero unseen regressions);
    # set include_fragmentation=True to also relink occlusion-split newborn tracks.
    require_lifecycle_event: bool = True
    include_fragmentation: bool = False
    fragmentation_newborn_iou: float = 0.20
    max_frame_rate: float | None = None
    bucket_cap_frames: float = 20.0

    # Two-Tier Safe Bijective Reconciliation
    min_seed_score: float = 0.60
    min_mask_area_ratio: float = 0.12
    confirm_frames: int = 2
    # Tier 1: mutual swap cycles (A <-> B)
    mutual_min_score: float = 0.45
    mutual_min_containment: float = 0.55
    mutual_min_margin: float = 0.05
    # Tier 2: area-guarded unilateral overrides / relinks
    unilateral_min_score: float = 0.72
    unilateral_min_containment: float = 0.85
    unilateral_min_margin: float = 0.28
    unilateral_min_area_ratio: float = 0.45
    unilateral_max_area_ratio: float = 1.85
    apply_in_window: bool = True

    def __post_init__(self) -> None:
        if self.max_window_len < 4:
            raise ValueError("max_window_len must be at least 4.")
        if self.max_cluster_tracks < 2:
            raise ValueError("max_cluster_tracks must be at least 2.")
        if self.lookback_frames < 0:
            raise ValueError("lookback_frames must be nonnegative.")
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1.")
        if self.clean_exit_frames < 1:
            raise ValueError("clean_exit_frames must be at least 1.")
        if self.max_frame_rate is not None and not (0.0 < self.max_frame_rate <= 1.0):
            raise ValueError("max_frame_rate must be in (0, 1] when set.")


@dataclass(slots=True)
class AmbiguityWindow:
    """Closed ambiguity window eligible for selective mask propagation."""

    start_frame: int
    trigger_frame: int
    end_frame: int
    seeds: dict[int, tuple[int, BBox]]
    reason: str


@dataclass(slots=True)
class MaskFallbackStats:
    """Cumulative dispatch and reconciliation counters for a run."""

    processed_frames: int = 0
    windows_detected: int = 0
    windows_dispatched: int = 0
    windows_skipped_gate: int = 0
    windows_skipped_budget: int = 0
    dispatched_frames: int = 0
    overrides: int = 0
    mutual_links: int = 0
    unilateral_links: int = 0
    swaps: int = 0
    relinks: int = 0
    abstained: int = 0
    propagation_ms: float = 0.0

    @property
    def pct_frames(self) -> float:
        if self.processed_frames <= 0:
            return 0.0
        return 100.0 * self.dispatched_frames / self.processed_frames

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_frames": self.processed_frames,
            "windows_detected": self.windows_detected,
            "windows_dispatched": self.windows_dispatched,
            "windows_skipped_gate": self.windows_skipped_gate,
            "windows_skipped_budget": self.windows_skipped_budget,
            "dispatched_frames": self.dispatched_frames,
            "pct_frames": round(self.pct_frames, 2),
            "overrides": self.overrides,
            "mutual_links": self.mutual_links,
            "unilateral_links": self.unilateral_links,
            "swaps": self.swaps,
            "relinks": self.relinks,
            "abstained": self.abstained,
            "propagation_ms": round(self.propagation_ms, 2),
        }


def compute_mask_box_affinities(
    mask_bool: np.ndarray,
    boxes_xyxy: Sequence[BBox],
    orig_w: int,
    orig_h: int,
) -> tuple[int, list[tuple[float, float, float]]]:
    """Score a 2D boolean mask against candidate boxes in original pixel coordinates.

    Works on either native SAM 256x256 masks or full-resolution masks by scaling boxes to
    ``mask_bool.shape``. Returns ``(mask_area_pixels, [(combined, containment, bbox_iou), ...])``.
    """
    mask = np.asarray(mask_bool, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D boolean mask, got shape {mask.shape}")
    mh, mw = mask.shape
    ys, xs = np.nonzero(mask)
    area = int(ys.size)
    min_px = 6 if max(mh, mw) <= 320 else 64
    if area < min_px:
        return area, [(0.0, 0.0, 0.0) for _ in boxes_xyxy]

    if area >= 40:
        xs_s = np.sort(xs)
        ys_s = np.sort(ys)
        k = int(0.02 * area)
        mx1, mx2 = float(xs_s[k]), float(xs_s[area - 1 - k])
        my1, my2 = float(ys_s[k]), float(ys_s[area - 1 - k])
    else:
        mx1, mx2 = float(xs.min()), float(xs.max())
        my1, my2 = float(ys.min()), float(ys.max())
    m_area_box = max(1.0, (mx2 - mx1 + 1.0) * (my2 - my1 + 1.0))

    sx = mw / max(float(orig_w), 1.0)
    sy = mh / max(float(orig_h), 1.0)

    scores: list[tuple[float, float, float]] = []
    for bx1, by1, bx2, by2 in boxes_xyxy:
        sbx1, sby1, sbx2, sby2 = bx1 * sx, by1 * sy, bx2 * sx, by2 * sy
        ix1 = max(0, int(np.floor(sbx1)))
        iy1 = max(0, int(np.floor(sby1)))
        ix2 = min(mw, int(np.ceil(sbx2)))
        iy2 = min(mh, int(np.ceil(sby2)))
        if ix2 <= ix1 or iy2 <= iy1:
            scores.append((0.0, 0.0, 0.0))
            continue
        in_box = int(mask[iy1:iy2, ix1:ix2].sum())
        containment = in_box / max(area, 1)

        ox1, oy1 = max(mx1, sbx1), max(my1, sby1)
        ox2, oy2 = min(mx2, sbx2), min(my2, sby2)
        inter = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
        b_area = max(1.0, (sbx2 - sbx1) * (sby2 - sby1))
        biou = inter / max(m_area_box + b_area - inter, 1e-6)
        comb = 0.55 * containment + 0.45 * biou
        scores.append((comb, containment, biou))
    return area, scores


@dataclass(slots=True)
class _CommittedOverride:
    start_frame: int
    end_frame: int
    in_window_maps: dict[int, dict[int, int]]
    suffix_perm: dict[int, int]


def _close_bijective_frame_map(
    cur_to_tgt: dict[int, int],
    perm: dict[int, int],
) -> dict[int, int]:
    """Close ``cur_to_tgt`` (a subset of bijective ``perm``) along full ``perm`` orbits.

    Because ``perm`` is a bijection (closed cycles and chains ending in fresh IDs outside
    existing active IDs), closing every orbit touched by ``cur_to_tgt`` guarantees that the
    resulting per-frame map is strictly bijective across both active and coasting tracks.
    """
    fmap: dict[int, int] = {}
    for start_id in cur_to_tgt:
        if start_id in fmap:
            continue
        cur = start_id
        while cur in perm and cur not in fmap:
            nxt = perm[cur]
            fmap[cur] = nxt
            cur = nxt
    return fmap


def _compose_perm(base: dict[int, int], update: dict[int, int]) -> dict[int, int]:
    """Return ``lambda x: update.get(base.get(x, x), base.get(x, x))`` in dict form."""
    if not update:
        return base
    out: dict[int, int] = {}
    for k, v in base.items():
        nv = update.get(v, v)
        if nv != k:
            out[k] = nv
    for k, v in update.items():
        if k not in base and v != k:
            out[k] = v
    return out


class MaskFallbackResolver:
    """Online ambiguity monitor and bijective ID reconciler with bounded frame memory.

    Feed raw box-tracker outputs one frame at a time via :meth:`observe`. The resolver mutates
    each frame's :class:`Track` objects to apply already-committed identity fixes in real time,
    buffers at most ``max_window_len + lookback_frames + 4`` RGB frames in RAM (never writing
    frames to disk), and supports retrospective relabeling via :meth:`resolve_id` when flushing
    spooled exports.
    """

    def __init__(
        self,
        propagator: MaskPropagator,
        config: MaskFallbackConfig | None = None,
    ) -> None:
        self.propagator = propagator
        self.config = config or MaskFallbackConfig()
        self.stats = MaskFallbackStats()

        # Bounded per-frame RAM buffers keyed by frame_idx (pruned every step)
        self._frames_rgb: dict[int, np.ndarray] = {}
        self._raw_tracks_by_f: dict[int, list[tuple[int, BBox, float]]] = {}
        self._frame_wh: tuple[int, int] = (0, 0)
        self._seen_frame_order: list[int] = []

        # Online track lifecycle metadata (keyed by raw box-tracker track_id; stale keys pruned)
        self._track_age: dict[int, int] = defaultdict(int)
        self._track_max_conf: dict[int, float] = defaultdict(float)
        self._track_last_step: dict[int, int] = {}
        self._last_clean: dict[int, tuple[int, BBox]] = {}
        self._last_seen: dict[int, tuple[int, BBox]] = {}
        self._prev_active: dict[int, BBox] = {}
        self._prev2_active: dict[int, BBox] = {}
        self._pair_last_closed: dict[tuple[int, int], int] = {}
        self._open_windows: list[dict[str, Any]] = []

        # Causal leaky-bucket frame budget state (recent dispatched frames kept for dedup only)
        self._recent_dispatched_frames: set[int] = set()
        self._budget_tokens: float = self.config.bucket_cap_frames
        self._budget_last_step: int = 0

        # Committed bijective reconciliation history + O(1) live permutation for current/future frames
        self._commits: list[_CommittedOverride] = []
        self._commit_end_frames: list[int] = []
        self._live_perm: dict[int, int] = {}
        self._next_fresh_id: int = FRESH_ID_OFFSET

    def resolve_id(self, frame_idx: int, raw_track_id: int) -> int:
        """Map a raw tracker ID at ``frame_idx`` through all committed bijective fixes."""
        cid = int(raw_track_id)
        if not self._commits:
            return cid
        if frame_idx >= self._commit_end_frames[-1]:
            return self._live_perm.get(cid, cid)
        # Skip commits whose entire window lies strictly after frame_idx? Not needed if we scan
        # from the earliest commit with start_frame <= frame_idx, or apply only relevant commits.
        # Since commits are appended in non-decreasing end_frame order, suffix_perm applies to
        # frame_idx >= end_frame and in_window_maps applies to start_frame <= frame_idx < end_frame.
        for commit in self._commits:
            if frame_idx >= commit.end_frame:
                cid = commit.suffix_perm.get(cid, cid)
            elif commit.start_frame <= frame_idx:
                fmap = commit.in_window_maps.get(frame_idx)
                if fmap is not None:
                    cid = fmap.get(cid, cid)
        return cid

    def _eligible(self, raw_tid: int) -> bool:
        return (
            self._track_age[raw_tid] >= self.config.min_track_age
            and self._track_max_conf[raw_tid] >= self.config.min_track_conf
        )

    def _make_seed(self, raw_tid: int) -> tuple[int, BBox] | None:
        min_keep = (
            self._seen_frame_order[
                max(0, len(self._seen_frame_order) - 1 - self.config.lookback_frames)
            ]
            if self._seen_frame_order
            else 0
        )
        lc = self._last_clean.get(raw_tid)
        if lc is not None and lc[0] >= min_keep and lc[0] in self._frames_rgb:
            return lc
        ls = self._last_seen.get(raw_tid)
        if ls is not None and ls[0] >= min_keep and ls[0] in self._frames_rgb:
            return ls
        return None

    def _passes_stage_b(self, win: dict[str, Any], end_frame: int) -> bool:
        cfg = self.config
        if not cfg.require_lifecycle_event:
            return True

        start_f = int(win["start"])
        seeds: dict[int, tuple[int, BBox]] = win["seeds"]
        window_frames = [f for f in self._seen_frame_order if start_f <= f <= end_frame]
        if len(window_frames) < 2:
            return False

        exit_frames = window_frames[-2:]
        alive_exit: dict[int, BBox] = {}
        for ef in exit_frames:
            for tid, bbox, _ in self._raw_tracks_by_f.get(ef, ()):
                alive_exit[tid] = bbox

        before_tids: set[int] = set()
        for f in self._seen_frame_order:
            if f <= start_f:
                for tid, _, _ in self._raw_tracks_by_f.get(f, ()):
                    before_tids.add(tid)

        any_recovered = False
        any_died = False
        for st, (sf, _sb) in seeds.items():
            missing_mid = False
            for f in window_frames:
                if f <= sf:
                    continue
                present = any(tid == st for tid, _, _ in self._raw_tracks_by_f.get(f, ()))
                if not present:
                    missing_mid = True
                elif missing_mid:
                    any_recovered = True
            if st not in alive_exit:
                any_died = True

        if any_recovered:
            return True
        if cfg.include_fragmentation and any_died:
            nb_first: dict[int, BBox] = {}
            for f in window_frames:
                for tid, bbox, _ in self._raw_tracks_by_f.get(f, ()):
                    if tid not in before_tids and tid not in seeds and tid not in nb_first:
                        nb_first[tid] = bbox
            nb_boxes = list(nb_first.values()) + [
                b for tid, b in alive_exit.items() if tid not in before_tids and tid not in seeds
            ]
            ref_boxes = [sb for _, sb in seeds.values()] + [
                self._last_seen[st][1] for st in seeds if st in self._last_seen
            ]
            if nb_boxes and ref_boxes:
                max_iou = float(
                    compute_iou_matrix(
                        np.asarray(ref_boxes, dtype=np.float64),
                        np.asarray(nb_boxes, dtype=np.float64),
                    ).max()
                )
                if max_iou >= cfg.fragmentation_newborn_iou:
                    return True
        return False

    def _consume_budget(self, window_frames: list[int], step_idx: int) -> bool:
        new_frames = [f for f in window_frames if f not in self._recent_dispatched_frames]
        rate = self.config.max_frame_rate
        if rate is not None:
            dt = max(0, step_idx - self._budget_last_step)
            self._budget_tokens = min(
                self.config.bucket_cap_frames, self._budget_tokens + dt * rate
            )
            self._budget_last_step = step_idx
            if len(new_frames) > self._budget_tokens:
                return False
            self._budget_tokens -= len(new_frames)
        self._recent_dispatched_frames.update(new_frames)
        self.stats.dispatched_frames += len(new_frames)
        return True

    def _dispatch_and_reconcile(self, win: AmbiguityWindow) -> dict[str, Any] | None:
        cfg = self.config
        window_frames = [
            f
            for f in self._seen_frame_order
            if win.start_frame <= f <= win.end_frame and f in self._frames_rgb
        ]
        if len(window_frames) < 2:
            self.stats.abstained += 1
            return None

        f_to_local = {f: i for i, f in enumerate(window_frames)}
        frames_rgb = [self._frames_rgb[f] for f in window_frames]
        local_seeds: dict[int, tuple[int, BBox]] = {}
        for st, (sf, sb) in win.seeds.items():
            lf = f_to_local.get(sf)
            if lf is not None:
                local_seeds[st] = (lf, sb)
        if len(local_seeds) < 2:
            self.stats.abstained += 1
            return None

        t0 = time.perf_counter()
        masks_by_lf = self.propagator.propagate_window(frames_rgb, local_seeds, self._frame_wh)
        self.stats.propagation_ms += (time.perf_counter() - t0) * 1000.0
        self.stats.windows_dispatched += 1

        orig_w, orig_h = self._frame_wh
        seed_areas: dict[int, int] = {}
        frame_scores: dict[int, dict[int, tuple[int, list[tuple[int, float, float, float]]]]] = {}

        for lf, f_masks in masks_by_lf.items():
            if not (0 <= lf < len(window_frames)):
                continue
            vid_f = window_frames[lf]
            rows_f = self._raw_tracks_by_f.get(vid_f, [])
            boxes_f = [r[1] for r in rows_f]
            f_entry: dict[int, tuple[int, list[tuple[int, float, float, float]]]] = {}
            for st, mbool in f_masks.items():
                if st not in local_seeds:
                    continue
                area, affs = compute_mask_box_affinities(mbool, boxes_f, orig_w, orig_h)
                if lf == local_seeds[st][0] or (st not in seed_areas and area > 0):
                    seed_areas[st] = area
                matches = [
                    (rows_f[di][0], comb, cont, biou)
                    for di, (comb, cont, biou) in enumerate(affs)
                    if comb >= 0.05
                ]
                f_entry[st] = (area, matches)
            frame_scores[vid_f] = f_entry

        # Map each seeded raw_tid -> unique canonical ID at its seed frame, filtering poor seeds
        seed_raw_to_cur: dict[int, int] = {}
        seen_seed_curs: set[int] = set()
        for st, (sf, _sb) in sorted(win.seeds.items()):
            if st not in seed_areas or seed_areas[st] <= 0:
                continue
            if cfg.min_seed_score > 0.0:
                srec = frame_scores.get(sf, {}).get(st)
                if srec is None:
                    continue
                self_sc = next((comb for ot, comb, _c, _b in srec[1] if ot == st), 0.0)
                if self_sc < cfg.min_seed_score:
                    continue
            scur = self.resolve_id(sf, st)
            if scur in seen_seed_curs:
                continue
            seen_seed_curs.add(scur)
            seed_raw_to_cur[st] = scur

        if not seed_raw_to_cur:
            self.stats.abstained += 1
            return None

        # 1-to-1 Hungarian mask-to-track matching per window frame
        per_f_match: dict[int, dict[int, tuple[int, float, float, float, float]]] = {}
        for vid_f in window_frames:
            f_rec = frame_scores.get(vid_f)
            rows_f = self._raw_tracks_by_f.get(vid_f, [])
            if not f_rec or not rows_f:
                continue
            valid_seeds = [
                st
                for st in seed_raw_to_cur
                if st in f_rec
                and f_rec[st][0] >= max(6, int(cfg.min_mask_area_ratio * max(seed_areas[st], 1)))
            ]
            if not valid_seeds:
                continue
            raw_tids_f = [r[0] for r in rows_f]
            raw_to_col = {rt: ci for ci, rt in enumerate(raw_tids_f)}
            raw_to_cur = {rt: self.resolve_id(vid_f, rt) for rt in raw_tids_f}

            score_mat = np.zeros((len(valid_seeds), len(rows_f)), dtype=np.float64)
            cont_mat = np.zeros((len(valid_seeds), len(rows_f)), dtype=np.float64)
            for ri, st in enumerate(valid_seeds):
                for ot, comb, cont, _biou in f_rec[st][1]:
                    ci = raw_to_col.get(ot)
                    if ci is not None:
                        score_mat[ri, ci] = comb
                        cont_mat[ri, ci] = cont

            r_ind, c_ind = linear_sum_assignment(-score_mat)
            f_matches: dict[int, tuple[int, float, float, float, float]] = {}
            for ri, ci in zip(r_ind.tolist(), c_ind.tolist(), strict=True):
                sc = float(score_mat[ri, ci])
                ct = float(cont_mat[ri, ci])
                if sc < 0.35 or ct < 0.45:
                    continue
                row_other = np.delete(score_mat[ri], ci)
                col_other = np.delete(score_mat[:, ci], ri)
                s2 = max(
                    float(row_other.max()) if row_other.size else 0.0,
                    float(col_other.max()) if col_other.size else 0.0,
                )
                margin = sc - s2
                st = valid_seeds[ri]
                area_ratio = float(f_rec[st][0]) / max(float(seed_areas[st]), 1.0)
                f_matches[st] = (raw_to_cur[raw_tids_f[ci]], sc, ct, margin, area_ratio)
            if f_matches:
                per_f_match[vid_f] = f_matches

        exit_frames = [f for f in window_frames if f >= win.trigger_frame][
            -max(cfg.confirm_frames + 1, 3) :
        ]
        if not exit_frames:
            exit_frames = window_frames[-max(cfg.confirm_frames, 2) :]

        seeded_cur_set = set(seed_raw_to_cur.values())

        # Tier 1: Mutual swap cycles (A <-> B)
        cand_mut: dict[int, int] = {}
        for st, seed_cur in seed_raw_to_cur.items():
            obs: list[int] = []
            for ef in exit_frames:
                m = per_f_match.get(ef, {}).get(st)
                if m is None:
                    continue
                ex_cur, sc, ct, marg, _ar = m
                if (
                    sc >= cfg.mutual_min_score
                    and ct >= cfg.mutual_min_containment
                    and marg >= cfg.mutual_min_margin
                ):
                    obs.append(ex_cur)
            if len(obs) >= cfg.confirm_frames and len(set(obs[-cfg.confirm_frames :])) == 1:
                ex_cur = obs[-1]
                if ex_cur != seed_cur and ex_cur in seeded_cur_set and ex_cur not in cand_mut:
                    cand_mut[ex_cur] = seed_cur

        desired: dict[int, int] = {}
        n_mut = 0
        for ex_cur, seed_cur in cand_mut.items():
            if seed_cur in cand_mut and cand_mut[seed_cur] == ex_cur:
                desired[ex_cur] = seed_cur
                n_mut += 1

        # Tier 2: High-confidence area-guarded unilateral overrides / relinks
        n_uni = 0
        for st, seed_cur in seed_raw_to_cur.items():
            if seed_cur in desired.values():
                continue
            obs_u: list[int] = []
            for ef in exit_frames:
                m = per_f_match.get(ef, {}).get(st)
                if m is None:
                    continue
                ex_cur, sc, ct, marg, ar = m
                if (
                    sc >= cfg.unilateral_min_score
                    and ct >= cfg.unilateral_min_containment
                    and marg >= cfg.unilateral_min_margin
                    and cfg.unilateral_min_area_ratio <= ar <= cfg.unilateral_max_area_ratio
                ):
                    obs_u.append(ex_cur)
            if len(obs_u) >= cfg.confirm_frames and len(set(obs_u[-cfg.confirm_frames :])) == 1:
                ex_cur = obs_u[-1]
                if ex_cur != seed_cur and ex_cur not in desired:
                    desired[ex_cur] = seed_cur
                    n_uni += 1

        if not desired or len(desired.values()) != len(set(desired.values())):
            self.stats.abstained += 1
            return None

        # Strictly bijective suffix permutation (prevents future ID collisions)
        perm = dict(desired)
        n_swaps = sum(1 for k in desired if k in seeded_cur_set)
        n_relinks = len(desired) - n_swaps
        relink_pairs: list[tuple[int, int]] = []
        for k, v in desired.items():
            if k not in seeded_cur_set:
                relink_pairs.append((k, v))
        for tgt in sorted(set(desired.values())):
            if tgt not in perm:
                perm[tgt] = self._next_fresh_id
                self._next_fresh_id += 1

        # In-window bijective maps for frames strictly before win.end_frame;
        # win.end_frame and all later frames use suffix_perm directly (zero end_frame flicker).
        in_window_maps: dict[int, dict[int, int]] = {}
        if cfg.apply_in_window:
            for vid_f in window_frames:
                if vid_f >= win.end_frame:
                    continue
                fm = per_f_match.get(vid_f, {})
                cur_to_tgt: dict[int, int] = {}
                for st, (ex_cur, sc, ct, _marg, _ar) in fm.items():
                    if sc >= 0.45 and ct >= 0.55:
                        tgt = seed_raw_to_cur[st]
                        if perm.get(ex_cur) == tgt:
                            cur_to_tgt[ex_cur] = tgt
                if cur_to_tgt:
                    in_window_maps[vid_f] = _close_bijective_frame_map(cur_to_tgt, perm)

        commit = _CommittedOverride(
            start_frame=win.start_frame,
            end_frame=win.end_frame,
            in_window_maps=in_window_maps,
            suffix_perm=perm,
        )
        self._commits.append(commit)
        bisect.insort(self._commit_end_frames, win.end_frame)
        self._live_perm = _compose_perm(self._live_perm, perm)

        self.stats.overrides += 1
        self.stats.mutual_links += n_mut
        self.stats.unilateral_links += n_uni
        self.stats.swaps += n_swaps
        self.stats.relinks += n_relinks
        return {
            "type": "mask_fallback_resolved",
            "window_start": win.start_frame,
            "window_end": win.end_frame,
            "reason": win.reason,
            "swaps": n_swaps,
            "relinks": n_relinks,
            "relink_pairs": relink_pairs,
            "permutation": {str(k): v for k, v in perm.items()},
        }

    def observe(
        self, frame_idx: int, frame_bgr: np.ndarray, tracks: list[Track]
    ) -> list[dict[str, Any]]:
        """Ingest one tracked frame, dispatch any closed qualifying ambiguity windows, and relabel ``tracks``."""
        cfg = self.config
        self.stats.processed_frames += 1
        step_idx = self.stats.processed_frames
        h, w = frame_bgr.shape[:2]
        self._frame_wh = (w, h)
        self._frames_rgb[frame_idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._seen_frame_order.append(frame_idx)

        raw_rows: list[tuple[int, BBox, float]] = [
            (int(t.track_id), tuple(float(v) for v in t.bbox), float(t.confidence)) for t in tracks
        ]
        self._raw_tracks_by_f[frame_idx] = raw_rows
        if raw_rows:
            max_raw = max(r[0] for r in raw_rows)
            if max_raw >= self._next_fresh_id - 10_000:
                self._next_fresh_id = max(self._next_fresh_id, max_raw + FRESH_ID_OFFSET)

        active_boxes = {tid: box for tid, box, _ in raw_rows}
        tids = [tid for tid, _, _ in raw_rows]
        active_set = set(tids)

        # Pairwise active IoU on current frame
        track_max_iou: dict[int, float] = {t: 0.0 for t in tids}
        trig_pairs: list[tuple[int, int, str, float]] = []
        if len(tids) >= 2:
            arr = np.asarray([active_boxes[t] for t in tids], dtype=np.float64)
            iou_m = compute_iou_matrix(arr, arr)
            np.fill_diagonal(iou_m, 0.0)
            for i, tid in enumerate(tids):
                track_max_iou[tid] = float(iou_m[i].max())
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    v = float(iou_m[i, j])
                    if (
                        v >= cfg.iou_overlap_thresh
                        and self._eligible(tids[i])
                        and self._eligible(tids[j])
                    ):
                        trig_pairs.append((tids[i], tids[j], "overlap", v))

        # Update age, confidence, and clean-seed snapshots
        for tid, box, conf in raw_rows:
            self._track_age[tid] += 1
            self._track_last_step[tid] = step_idx
            if conf > self._track_max_conf[tid]:
                self._track_max_conf[tid] = conf
            if track_max_iou[tid] < cfg.clean_seed_max_iou:
                self._last_clean[tid] = (frame_idx, box)
            self._last_seen[tid] = (frame_idx, box)

        # Detect tracks that just vanished while covered by an overlapping active track (lost_occl).
        # Use 1-step constant-velocity extrapolation when available so fast crossings match at frame t.
        just_lost = [t for t in self._prev_active if t not in active_set]
        if just_lost and tids:
            pred_lost: list[BBox] = []
            for t in just_lost:
                b1 = self._prev_active[t]
                b0 = self._prev2_active.get(t)
                if b0 is not None:
                    pred_lost.append(
                        (
                            b1[0] + (b1[0] - b0[0]),
                            b1[1] + (b1[1] - b0[1]),
                            b1[2] + (b1[2] - b0[2]),
                            b1[3] + (b1[3] - b0[3]),
                        )
                    )
                else:
                    pred_lost.append(b1)
            lost_arr = np.asarray(pred_lost, dtype=np.float64)
            act_arr = np.asarray([active_boxes[t] for t in tids], dtype=np.float64)
            ioa_m = compute_ioa_matrix(lost_arr, act_arr)
            for li, ltid in enumerate(just_lost):
                if not self._eligible(ltid):
                    continue
                aj = int(np.argmax(ioa_m[li]))
                val = float(ioa_m[li, aj])
                atid = tids[aj]
                if val >= cfg.occlusion_ioa_thresh and self._eligible(atid):
                    trig_pairs.append((ltid, atid, "lost_occl", val + 0.3))

        # Open or extend pair-scoped ambiguity windows
        matched_wins: set[int] = set()
        trig_pairs.sort(key=lambda x: x[3], reverse=True)
        for ta, tb, rsn, sev in trig_pairs:
            pkey = (min(ta, tb), max(ta, tb))
            if step_idx - self._pair_last_closed.get(pkey, -999) < cfg.cooldown_frames:
                continue
            target_wi = None
            for wi, wobj in enumerate(self._open_windows):
                if ta in wobj["tracks"] and tb in wobj["tracks"]:
                    target_wi = wi
                    break
            if target_wi is None:
                for wi, wobj in enumerate(self._open_windows):
                    if (ta in wobj["tracks"] or tb in wobj["tracks"]) and len(
                        wobj["tracks"] | {ta, tb}
                    ) <= cfg.max_cluster_tracks:
                        target_wi = wi
                        break
            if target_wi is not None:
                wobj = self._open_windows[target_wi]
                wobj["tracks"].update((ta, tb))
                wobj["clean_streak"] = 0
                if sev > wobj["severity"]:
                    wobj["severity"] = sev
                    wobj["reason"] = rsn
                for t in (ta, tb):
                    if t not in wobj["seeds"]:
                        sd = self._make_seed(t)
                        if sd is not None and sd[0] >= wobj["start"]:
                            wobj["seeds"][t] = sd
                matched_wins.add(target_wi)
            else:
                seeds: dict[int, tuple[int, BBox]] = {}
                for t in (ta, tb):
                    sd = self._make_seed(t)
                    if sd is not None:
                        seeds[t] = sd
                if len(seeds) < 2:
                    continue
                start_f = min(sf for sf, _ in seeds.values())
                self._open_windows.append(
                    {
                        "start": start_f,
                        "start_step": step_idx
                        - (len(self._seen_frame_order) - 1 - self._seen_frame_order.index(start_f)),
                        "trig": frame_idx,
                        "tracks": {ta, tb},
                        "seeds": seeds,
                        "clean_streak": 0,
                        "reason": rsn,
                        "severity": sev,
                    }
                )
                matched_wins.add(len(self._open_windows) - 1)

        # Close resolved or max-length windows and dispatch qualifying ones
        events: list[dict[str, Any]] = []
        still_open: list[dict[str, Any]] = []
        for wi, wobj in enumerate(self._open_windows):
            win_steps = step_idx - wobj["start_step"] + 1
            if wi not in matched_wins:
                active_seeded = [t for t in wobj["seeds"] if t in active_set]
                still_tangled = False
                if len(active_seeded) >= 2:
                    sarr = np.asarray([active_boxes[t] for t in active_seeded], dtype=np.float64)
                    siou = compute_iou_matrix(sarr, sarr)
                    np.fill_diagonal(siou, 0.0)
                    if float(siou.max()) >= 0.25:
                        still_tangled = True
                if not still_tangled:
                    for stid in wobj["seeds"]:
                        if stid not in active_set:
                            ls = self._last_seen.get(stid)
                            if ls is not None and ls[0] in self._seen_frame_order[-6:]:
                                still_tangled = True
                                break
                if still_tangled:
                    wobj["clean_streak"] = 0
                else:
                    wobj["clean_streak"] += 1

            if wobj["clean_streak"] >= cfg.clean_exit_frames or win_steps >= cfg.max_window_len:
                self.stats.windows_detected += 1
                if not self._passes_stage_b(wobj, frame_idx):
                    self.stats.windows_skipped_gate += 1
                else:
                    w_frames = [
                        f for f in self._seen_frame_order if wobj["start"] <= f <= frame_idx
                    ]
                    if not self._consume_budget(w_frames, step_idx):
                        self.stats.windows_skipped_budget += 1
                    else:
                        ev = self._dispatch_and_reconcile(
                            AmbiguityWindow(
                                start_frame=wobj["start"],
                                trigger_frame=wobj["trig"],
                                end_frame=frame_idx,
                                seeds=dict(wobj["seeds"]),
                                reason=wobj["reason"],
                            )
                        )
                        if ev is not None:
                            events.append(ev)
                tlist = sorted(wobj["tracks"])
                for i in range(len(tlist)):
                    for j in range(i + 1, len(tlist)):
                        self._pair_last_closed[(tlist[i], tlist[j])] = step_idx
            else:
                still_open.append(wobj)
        self._open_windows = still_open
        self._prev2_active = self._prev_active
        self._prev_active = active_boxes

        # Prune RAM ring buffer & stale per-track metadata so memory stays O(1) on long streams
        max_keep = cfg.max_window_len + cfg.lookback_frames + 4
        if len(self._seen_frame_order) > max_keep:
            oldest_open = min((w["start"] for w in self._open_windows), default=frame_idx)
            cutoff = min(oldest_open, self._seen_frame_order[-max_keep])
            while self._seen_frame_order and self._seen_frame_order[0] < cutoff:
                old_f = self._seen_frame_order.pop(0)
                self._frames_rgb.pop(old_f, None)
                self._raw_tracks_by_f.pop(old_f, None)
                self._recent_dispatched_frames.discard(old_f)

        if step_idx % 256 == 0:
            stale_step = step_idx - max(max_keep * 4, 256)
            open_tids = {t for w in self._open_windows for t in w["tracks"]} | active_set
            for tid in [t for t, ls in self._track_last_step.items() if ls < stale_step and t not in open_tids]:
                self._track_last_step.pop(tid, None)
                self._track_age.pop(tid, None)
                self._track_max_conf.pop(tid, None)
                self._last_clean.pop(tid, None)
                self._last_seen.pop(tid, None)
            cd_cutoff = step_idx - max(cfg.cooldown_frames * 2, 32)
            for pkey in [p for p, cs in self._pair_last_closed.items() if cs < cd_cutoff]:
                self._pair_last_closed.pop(pkey, None)

        # Apply committed canonical IDs to live tracks for this frame in O(1)
        for tr, (raw_tid, _, _) in zip(tracks, raw_rows, strict=True):
            tr.track_id = self.resolve_id(frame_idx, raw_tid)

        return events


def _to_bool_mask(arr: Any) -> np.ndarray:
    """Convert a NumPy array or PyTorch CPU/CUDA tensor to a 2D boolean NumPy mask without importing torch."""
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    m = np.asarray(arr)
    while m.ndim > 2:
        m = m[0]
    return m > 0


class Sam3MaskPropagator:
    """External zero-dependency adapter wrapping a user-supplied SAM 3 video predictor.

    Keeps Meta's SAM 3 code, weights, and non-OSI Meta SAM License completely outside
    Trackflow. Compatible with ``sam3.model_builder.build_sam3_video_predictor()``
    (``handle_request`` + ``handle_stream_request`` with ``out_obj_ids``/``out_binary_masks``)
    as well as custom predictors returning ``{"outputs": [...]}``.

    Because SAM 3's ``start_session`` reads a directory of JPEG frames, this adapter writes
    short temporary window JPEGs to disk (``writes_frames_to_disk = True``) and is rejected
    by :class:`TrackflowPipeline` when ``anonymize != "none"``.
    """

    writes_frames_to_disk: bool = True

    def __init__(self, predictor: Any) -> None:
        if not hasattr(predictor, "handle_request"):
            raise TypeError(
                "Sam3MaskPropagator expects a SAM 3 video predictor with a handle_request() method."
            )
        self.predictor = predictor

    def propagate_window(
        self,
        frames_rgb: Sequence[np.ndarray],
        seeds: dict[int, tuple[int, BBox]],
        frame_size_wh: tuple[int, int],
    ) -> dict[int, dict[int, np.ndarray]]:
        if not frames_rgb or not seeds:
            return {}
        orig_w, orig_h = max(1, int(frame_size_wh[0])), max(1, int(frame_size_wh[1]))
        with tempfile.TemporaryDirectory(prefix="trackflow_sam3_") as tmp_dir:
            clip_dir = Path(tmp_dir)
            for idx, rgb in enumerate(frames_rgb):
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(clip_dir / f"{idx:06d}.jpg"), bgr)

            start_resp = self.predictor.handle_request(
                {"type": "start_session", "resource_path": str(clip_dir)}
            )
            session_id = start_resp["session_id"]
            try:
                for obj_id in sorted(seeds.keys()):
                    lf, (x1, y1, x2, y2) = seeds[obj_id]
                    cx1 = max(0.0, min(float(orig_w), float(x1)))
                    cy1 = max(0.0, min(float(orig_h), float(y1)))
                    cx2 = max(cx1, min(float(orig_w), float(x2)))
                    cy2 = max(cy1, min(float(orig_h), float(y2)))
                    norm_xywh = [
                        cx1 / orig_w,
                        cy1 / orig_h,
                        max(1e-4, (cx2 - cx1) / orig_w),
                        max(1e-4, (cy2 - cy1) / orig_h),
                    ]
                    self.predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": int(lf),
                            "obj_id": int(obj_id),
                            "bounding_boxes": [norm_xywh],
                            "bounding_box_labels": [1],
                        }
                    )

                prop_req = {"type": "propagate_in_video", "session_id": session_id}
                if hasattr(self.predictor, "handle_stream_request"):
                    outputs = self.predictor.handle_stream_request(prop_req)
                else:
                    prop_resp = self.predictor.handle_request(prop_req)
                    outputs = (
                        prop_resp.get("outputs", []) if isinstance(prop_resp, dict) else prop_resp
                    )

                masks_by_lf: dict[int, dict[int, np.ndarray]] = {}
                for item in outputs:
                    lf = int(item["frame_index"])
                    f_masks: dict[int, np.ndarray] = {}
                    # Real facebookresearch/sam3 schema: item["outputs"]["out_obj_ids" / "out_binary_masks"]
                    nested = item.get("outputs") if isinstance(item, dict) else None
                    if isinstance(nested, dict) and "out_binary_masks" in nested:
                        oids = nested.get("out_obj_ids", ())
                        if hasattr(oids, "tolist"):
                            oids = oids.tolist()
                        bmasks = nested["out_binary_masks"]
                        for oid, arr in zip(oids, bmasks, strict=True):
                            f_masks[int(oid)] = _to_bool_mask(arr)
                    else:
                        raw_masks = item.get("masks", {})
                        if isinstance(raw_masks, dict):
                            for oid, arr in raw_masks.items():
                                f_masks[int(oid)] = _to_bool_mask(arr)
                        elif "obj_ids" in item:
                            for oid, arr in zip(item["obj_ids"], raw_masks, strict=True):
                                f_masks[int(oid)] = _to_bool_mask(arr)
                    if f_masks:
                        masks_by_lf[lf] = f_masks
                return masks_by_lf
            finally:
                with contextlib.suppress(Exception):
                    self.predictor.handle_request(
                        {"type": "close_session", "session_id": session_id}
                    )
