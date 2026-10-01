"""
HOTA (Luiten et al. 2021) for one sequence, following the TrackEval algorithm:
per-frame Hungarian matching biased by a global track-alignment score, averaged
over IoU thresholds 0.05..0.95. Returns DetA, AssA, HOTA (all 0..1).

gt / pred: {frame: (ids Nx, boxes Nx4 as x,y,w,h)}
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.951, 0.05)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, :1], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1:2], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter + 1e-12)


def hota(gt: dict, pred: dict) -> dict:
    frames = sorted(set(gt) | set(pred))
    gids = sorted({int(i) for f in gt.values() for i in f[0]})
    pids = sorted({int(i) for f in pred.values() for i in f[0]})
    if not gids:
        return {"HOTA": float("nan"), "DetA": float("nan"), "AssA": float("nan")}
    gmap, pmap = {g: i for i, g in enumerate(gids)}, {p: i for i, p in enumerate(pids)}
    g_count, p_count = np.zeros(len(gids)), np.zeros(max(len(pids), 1))
    potential = np.zeros((len(gids), max(len(pids), 1)))
    per_frame = []
    empty = (np.zeros(0, int), np.zeros((0, 4)))
    for f in frames:
        gi, gb = gt.get(f, empty)
        pi, pb = pred.get(f, empty)
        gi = np.array([gmap[int(i)] for i in gi], int)
        pi = np.array([pmap[int(i)] for i in pi], int)
        sim = _iou(np.asarray(gb, float).reshape(-1, 4), np.asarray(pb, float).reshape(-1, 4))
        g_count[gi] += 1
        p_count[pi] += 1
        if sim.size:
            potential[np.ix_(gi, pi)] += np.where(sim > 1e-10, sim, 0.0)
        per_frame.append((gi, pi, sim))
    align = potential / np.maximum(g_count[:, None] + p_count[None, :] - potential, 1e-12)

    n_gt, n_pr = int(g_count.sum()), int(p_count.sum())
    det_a, ass_a = [], []
    for alpha in ALPHAS:
        match_cnt = np.zeros_like(potential)
        tp = 0
        for gi, pi, sim in per_frame:
            if not sim.size:
                continue
            score = align[np.ix_(gi, pi)] * sim
            r, c = linear_sum_assignment(-score)
            ok = sim[r, c] >= alpha - 1e-10
            for ri, ci in zip(r[ok], c[ok]):
                match_cnt[gi[ri], pi[ci]] += 1
            tp += int(ok.sum())
        det_a.append(tp / max(n_gt + n_pr - tp, 1))
        if tp:
            a = match_cnt / np.maximum(g_count[:, None] + p_count[None, :] - match_cnt, 1e-12)
            ass_a.append(float((match_cnt * a).sum() / tp))
        else:
            ass_a.append(0.0)
    det_a, ass_a = np.array(det_a), np.array(ass_a)
    return {"HOTA": float(np.sqrt(det_a * ass_a).mean()), "DetA": float(det_a.mean()),
            "AssA": float(ass_a.mean())}
