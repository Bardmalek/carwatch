"""
Learned behaviour classifier (optional, `carwatch.py --ml`).

Two gradient-boosted classifiers read a 3 s track window (positions in
car-length units) and output P(weaving) and P(harsh accel/brake). They replace
the hand-set thresholds in Vehicle.analyse; the rules stay as the baseline.

IMPORTANT: with no labelled real footage yet, the training data is randomised
SYNTHETIC trajectories. Scores on synthetic data say the pipeline works, not
that it is accurate on real driving. Real accuracy needs labelled clips.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "behaviour_model.joblib"
FEATURE_NAMES = ["lat_amp", "lat_std", "lat_cross", "lat_dom_hz", "lat_dom_frac",
                 "acc_p90", "acc_min", "acc_max", "dv", "speed_med", "speed_cv",
                 "straightness"]


def window_features(T: np.ndarray, P: np.ndarray) -> np.ndarray | None:
    """Features of one window. T seconds, P Nx2 in car-lengths. None if too short."""
    if len(T) < 12 or len(np.unique(T)) < 12 or np.ptp(T) < 0.5 or not np.all(np.isfinite(P)):
        return None
    dt = np.clip(np.diff(T), 1e-3, None)
    d = P[-1] - P[0]
    if np.linalg.norm(d) < 1e-6:
        return None
    u = d / np.linalg.norm(d)
    n = np.array([-u[1], u[0]])
    tt = T - T[0]
    lat = (P - P[0]) @ n
    # cubic fit removes turns and single lane changes, as in the rule version
    resid = lat - np.polyval(np.polyfit(tt, lat, 3), tt)
    sg = np.sign(resid[np.abs(resid) > 0.05])
    cross = float(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0.0
    # dominant oscillation frequency of the residual (weaving is periodic)
    r = resid - resid.mean()
    spec = np.abs(np.fft.rfft(r * np.hanning(len(r)))) ** 2
    freqs = np.fft.rfftfreq(len(r), float(np.median(dt)))
    spec[0] = 0
    k = int(np.argmax(spec))
    dom_hz, dom_frac = float(freqs[k]), float(spec[k] / (spec.sum() + 1e-12))
    # Smooth along-track distance (~0.4 s), then differentiate. Edge-padding the
    # smoothing creates a fake speed kink at the window ends, so trim k5 samples
    # from each end of speed and acceleration before using them.
    step = float(np.median(np.diff(T)[np.diff(T) > 1e-6])) if np.any(np.diff(T) > 1e-6) else 0.0
    if step <= 0:                                   # repeated timestamps: nothing to differentiate
        return None
    k5 = max(3, min(int(0.4 / step) | 1, (len(T) // 4) | 1))   # ~0.4 s, never longer than the window allows
    along = np.convolve(np.pad((P - P[0]) @ u, k5 // 2, mode="edge"),
                        np.ones(k5) / k5, "valid")
    v = (np.diff(along) / dt)[k5:-k5]
    a = np.diff(v) / dt[k5:-k5][1:]
    a = np.convolve(a, np.ones(k5) / k5, "valid")
    if len(v) < 4 or len(a) < 2:
        return None
    path = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
    return np.array([np.ptp(resid), resid.std(), cross, dom_hz, dom_frac,
                     np.percentile(np.abs(a), 90), a.min(), a.max(), np.ptp(v),
                     np.median(v), v.std() / (abs(v.mean()) + 1e-3),
                     np.linalg.norm(d) / (path + 1e-9)])


def synth_window(rng: np.random.Generator, shift: float = 1.0):
    """One random 3 s window and its labels (weaving, harsh). `shift` widens noise/params
    for out-of-distribution testing."""
    fps = rng.choice([24.0, 30.0, 60.0])
    T = np.arange(0, 3.0, 1 / fps)
    speed = rng.uniform(0.8, 5.0)
    x = speed * T
    y = np.zeros_like(T)
    weave = rng.random() < 0.3
    harsh = rng.random() < 0.3
    kind = rng.choice(["straight", "turn", "lane", "creep"], p=[.35, .25, .3, .1])
    if kind == "lane":
        s = np.clip((T - rng.uniform(0.3, 1.5)) / rng.uniform(1.0, 2.0), 0, 1)
        y = rng.choice([-1, 1]) * rng.uniform(0.5, 1.0) * (3 * s**2 - 2 * s**3)
    elif kind == "creep":
        speed = rng.uniform(0.1, 0.4)
        x = speed * T
    if weave:
        amp = rng.uniform(0.15, 0.45) * shift
        per = rng.uniform(0.7, 1.6)
        y = y + amp * np.sin(2 * np.pi * T / per + rng.uniform(0, 6.28))
    if harsh:
        t0 = rng.uniform(0.5, 1.8)
        decel = rng.uniform(3.0, 6.0) * rng.choice([-1, 1] if speed < 2 else [1, 1, -1])
        v = np.full_like(T, speed)
        v = np.clip(v - np.clip(T - t0, 0, 1.0) * decel, 0, speed * 2.5)
        x = np.cumsum(v) / fps
    elif rng.random() < 0.25:                      # gentle stop: slow down to zero, NOT harsh
        d = rng.uniform(0.3, 1.2)
        v = np.clip(speed - d * np.clip(T - rng.uniform(0, 1.5), 0, None), 0, None)
        x = np.cumsum(v) / fps
    elif rng.random() < 0.3:                       # comfortable speed change
        v = speed + np.linspace(0, rng.uniform(-0.5, 0.5), len(T))
        x = np.cumsum(np.clip(v, 0, None)) / fps
    P = np.stack([x, y], 1)
    if kind == "turn":
        th = rng.uniform(0.3, 1.5) * rng.choice([-1, 1]) * T / 3.0
        c, s_ = np.cos(th), np.sin(th)
        P = np.stack([c * x - s_ * y, s_ * x + c * y], 1)
    P = P + rng.normal(0, rng.uniform(0.005, 0.03) * shift, P.shape)   # detector jitter
    return T, P, weave, harsh


def make_dataset(n: int, seed: int, shift: float = 1.0):
    rng = np.random.default_rng(seed)
    X, yw, yh = [], [], []
    while len(X) < n:
        T, P, w, h = synth_window(rng, shift)
        f = window_features(T, P)
        if f is not None:
            X.append(f); yw.append(int(w)); yh.append(int(h))
    return np.array(X), np.array(yw), np.array(yh)


class BehaviourML:
    def __init__(self, path: Path = MODEL_PATH):
        import joblib
        m = joblib.load(path)
        self.weave, self.harsh = m["weave"], m["harsh"]

    def predict(self, T, P) -> tuple[dict[str, float], dict[str, float]] | None:
        """Probabilities plus the feature values behind them (for the event log)."""
        f = window_features(np.asarray(T), np.asarray(P))
        if f is None:
            return None
        p = {"WEAVING": float(self.weave.predict_proba([f])[0, 1]),
             "HARSH ACCEL/BRAKE": float(self.harsh.predict_proba([f])[0, 1])}
        return p, dict(zip(FEATURE_NAMES, map(float, f)))


# ---------------------------------------------------------------------------
# Injecting known events into REAL tracks (real background + labelled events)
# ---------------------------------------------------------------------------

def _frame(P: np.ndarray):
    d = P[-1] - P[0]
    if np.linalg.norm(d) < 1e-6:
        return None
    u = d / np.linalg.norm(d)
    return u, np.array([-u[1], u[0]])


def inject_weave(T: np.ndarray, P: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    fr = _frame(P)
    if fr is None:
        return None
    _, n = fr
    amp, per = rng.uniform(0.15, 0.45), rng.uniform(0.7, 1.6)
    wob = amp * np.sin(2 * np.pi * (T - T[0]) / per + rng.uniform(0, 6.28))
    return P + wob[:, None] * n


def inject_harsh(T: np.ndarray, P: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """Hard braking (or, 30 % of the time, hard acceleration) layered onto a real path.
    Only when the car is fast enough to actually shed that much speed."""
    fr = _frame(P)
    if fr is None:
        return None
    u, n = fr
    tt = T - T[0]
    s, lat = (P - P[0]) @ u, (P - P[0]) @ n
    v0 = float(np.median(np.gradient(s, tt)))
    d, tau, t0 = rng.uniform(1.5, 4.0), rng.uniform(0.6, 1.2), rng.uniform(0.5, 1.6)
    accel = rng.random() < 0.3
    if not accel:
        tau = min(tau, 0.8 * v0 / d) if v0 > 0 else 0
        if tau < 0.3:
            return None
    e = np.clip(tt - t0, 0, None)
    off = np.where(e < tau, 0.5 * d * e**2, 0.5 * d * tau**2 + d * tau * (e - tau))
    s_new = s + off if accel else np.maximum.accumulate(s - off)
    return P[0] + s_new[:, None] * u + lat[:, None] * n
