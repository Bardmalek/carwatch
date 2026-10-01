import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from hota import hota   # noqa: E402


def world(n_frames=60, n_obj=4):
    """objects moving right at different speeds; returns gt dict."""
    gt = {}
    for f in range(n_frames):
        ids = np.arange(n_obj)
        boxes = np.array([[20 + 6 * i * 0 + f * (2 + i), 50 * i + 10, 30, 20] for i in ids], float)
        gt[f] = (ids, boxes)
    return gt


def test_perfect_tracker_scores_one():
    gt = world()
    r = hota(gt, {f: (i + 100, b.copy()) for f, (i, b) in gt.items()})
    assert r["HOTA"] > 0.999 and r["AssA"] > 0.999


def test_id_swap_hurts_association_not_detection():
    gt = world()
    pred = {}
    for f, (i, b) in gt.items():
        ids = i + 100
        if f >= 30:                       # objects 0 and 1 swap identities halfway
            ids = ids.copy(); ids[[0, 1]] = ids[[1, 0]]
        pred[f] = (ids, b.copy())
    r = hota(gt, pred)
    assert r["DetA"] > 0.999
    assert 0.5 < r["AssA"] < 0.95


def test_missing_half_the_objects_hurts_detection():
    gt = world()
    pred = {f: (i[:2] + 100, b[:2].copy()) for f, (i, b) in gt.items()}
    r = hota(gt, pred)
    assert 0.4 < r["DetA"] < 0.6 and r["AssA"] > 0.999
