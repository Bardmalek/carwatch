"""
Speed and acceleration of one vehicle from its recent track, with error bars.

Fits a quadratic to the along-track distance over the last ~2 s:
    s(t) = c0 + c1*t + c2*t^2,   t measured back from "now"
so c1 is the speed now and 2*c2 is the acceleration (positive = speeding up).
The fit's own covariance gives a 1-sigma error, so a noisy track reports a
wide error instead of a confident wrong number. It assumes roughly constant
acceleration inside the window; a sharper change gets averaged.
"""
from __future__ import annotations

import numpy as np

WINDOW_S = 2.0
MIN_POINTS = 15
STATIONARY_M = 0.3        # moved less than this in the window -> treat as stopped


def estimate_motion(t, xy_m, window_s: float = WINDOW_S):
    """t: seconds (N,), xy_m: positions in metres (N,2). Returns a dict or None."""
    t, xy = np.asarray(t, float), np.asarray(xy_m, float)
    keep = t >= t[-1] - window_s
    t, xy = t[keep], xy[keep]
    if len(t) < MIN_POINTS or t[-1] - t[0] < 0.6 * window_s:
        return None
    d = xy[-1] - xy[0]
    dist = float(np.linalg.norm(d))
    if dist < STATIONARY_M:
        return {"speed_ms": 0.0, "speed_err": 0.3 / (t[-1] - t[0]), "acc_ms2": 0.0, "acc_err": 0.0}
    u = d / dist
    s = (xy - xy[0]) @ u
    tt = t - t[-1]
    (c2, c1, _), cov = np.polyfit(tt, s, 2, cov=True)
    return {"speed_ms": abs(float(c1)), "speed_err": float(np.sqrt(cov[1, 1])),
            "acc_ms2": float(2 * c2) * (1 if c1 >= 0 else -1),
            "acc_err": float(2 * np.sqrt(cov[0, 0]))}
