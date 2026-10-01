"""
Per-vehicle speed records, a persistent history file, and a live linear regression.

One record is written per vehicle when its track ends: its typical (median) moving
speed, colour, type, street, hour of day, and whether it was over the limit.
Everything stays in local CSV files; no plates, no faces, nothing leaves the machine.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

MIN_SAMPLES = 5            # moving speed estimates needed before a vehicle is recorded
MARGIN = 1.10              # same 10 % over-limit margin as the SPEEDING flag
MODEL_MIN_N = 25


@dataclass
class SpeedRecord:
    time: str              # ISO timestamp
    hour: float            # hour of day, e.g. 17.5 = 5:30 pm
    street: str
    vehicle: str
    color: str
    type: str
    speed_kmh: float       # median while moving
    limit_kmh: float
    over: int              # 1 if speed > limit * MARGIN
    basis: str             # calibrated / ppm / approx (approx = car-length guess)
    seconds: float


FIELDS = [f.name for f in fields(SpeedRecord)]


def record_for(v, street: str, when: datetime, limit: float) -> SpeedRecord | None:
    """Build a record from a finished vehicle (duck-typed on carwatch.Vehicle)."""
    if len(v.speed_samples) < MIN_SAMPLES:
        return None
    sp = float(np.median(v.speed_samples))
    return SpeedRecord(when.isoformat(timespec="seconds"), when.hour + when.minute / 60,
                       street, f"V{v.id:04d}", v.color or "unknown", v.label, round(sp, 1),
                       limit, int(sp > limit * MARGIN), v.basis, round(v.last_t - v.first_t, 1))


class SpeedStore:
    """Appends records to a history CSV and keeps them in memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[SpeedRecord] = load_records(self.path)

    def add(self, r: SpeedRecord) -> None:
        self.records.append(r)
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(asdict(r))


def load_records(path: str | Path) -> list[SpeedRecord]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append(SpeedRecord(row["time"], float(row["hour"]), row["street"], row["vehicle"],
                                       row["color"], row["type"], float(row["speed_kmh"]),
                                       float(row["limit_kmh"]), int(row["over"]), row["basis"],
                                       float(row["seconds"])))
            except (KeyError, ValueError):
                continue
    return out


# ---------------------------------------------------------------------------
# Linear regression: speed ~ colour + street + hour
# ---------------------------------------------------------------------------

def _dummies(values: list[str], min_count: int = 5):
    """One-hot with rare levels merged into 'other'; baseline = most common level."""
    from collections import Counter
    cnt = Counter(values)
    keep = {k for k, c in cnt.items() if c >= min_count}
    vals = [v if v in keep else "other" for v in values]
    cnt = Counter(vals)
    if len(cnt) < 2:
        return None, None, None
    base = cnt.most_common(1)[0][0]
    levels = sorted(k for k in cnt if k != base)
    X = np.array([[1.0 if v == lv else 0.0 for lv in levels] for v in vals])
    return X, levels, base


def fit_model(records: list[SpeedRecord]) -> dict:
    """OLS of median speed on colour, street and hour (cyclic). Returns a JSON-friendly dict."""
    n = len(records)
    if n < MODEL_MIN_N:
        return {"ready": False, "n": n, "need": MODEL_MIN_N}
    y = np.array([r.speed_kmh for r in records])
    cols, names, baseline = [np.ones((n, 1))], ["intercept"], {}
    for label, vals in (("colour", [r.color for r in records]), ("street", [r.street for r in records])):
        X, levels, base = _dummies(vals)
        if X is not None:
            cols.append(X); names += [f"{label}: {lv}" for lv in levels]; baseline[label] = base
    hours = np.array([r.hour for r in records])
    if hours.max() - hours.min() >= 1.0:          # need at least an hour of spread
        ang = 2 * np.pi * hours / 24
        cols.append(np.stack([np.sin(ang), np.cos(ang)], 1)); names += ["hour (sin)", "hour (cos)"]
    X = np.hstack(cols)
    k = X.shape[1]
    if n < k + 10:
        return {"ready": False, "n": n, "need": k + 10}
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 0, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    terms = [{"name": nm, "beta": float(b), "se": float(s), "t": float(b / s) if s > 0 else 0.0,
              "sig": bool(s > 0 and abs(b / s) > 2)} for nm, b, s in zip(names, beta, se)]
    return {"ready": True, "n": n, "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / dof,
            "rmse": float(np.sqrt(sigma2)), "baseline": baseline, "terms": terms,
            "note": "Descriptive only. Colour does not cause speed; treat any 'significant' "
                    "colour term as likely chance unless it repeats on new data."}


def demo_records(n: int = 400, seed: int = 0) -> list[SpeedRecord]:
    """Clearly synthetic records for trying the dashboard without any footage."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        street = str(rng.choice(["Main St", "Kingsway"], p=[.6, .4]))
        color = str(rng.choice(["white", "black", "silver", "gray", "blue", "red"], p=[.28, .22, .2, .15, .1, .05]))
        hour = float(np.clip(rng.normal(14, 4), 6, 22))
        rush = 1.0 if 7 <= hour <= 9 or 16 <= hour <= 18 else 0.0
        sp = float(np.clip(rng.normal(46 + (7 if street == "Kingsway" else 0) - 5 * rush, 8), 8, 95))
        out.append(SpeedRecord(f"2026-01-01T{int(hour):02d}:{int((hour % 1) * 60):02d}:00", hour, street,
                               f"V{i:04d}", color, "car", round(sp, 1), 50.0, int(sp > 50 * MARGIN),
                               "calibrated", 6.0))
    return out
