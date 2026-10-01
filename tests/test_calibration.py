"""Calibration must recover true metres from a perspective view of a known road."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import Calibration, speed_kmh   # noqa: E402

# A made-up camera looking down a road at an angle: metres -> pixels.
TRUE_H_M2PX = np.array([[60.0, 25.0, 300.0], [0.0, 30.0, 700.0], [0.0, 0.030, 1.0]])


def m2px(m):
    return cv2.perspectiveTransform(np.asarray(m, float).reshape(-1, 1, 2), TRUE_H_M2PX).reshape(-1, 2)


def test_recovers_metres_from_rectangle():
    rect_m = [(0, 0), (3.6, 0), (3.6, 12), (0, 12)]
    c = Calibration.from_rectangle(m2px(rect_m), 3.6, 12)
    probe = [(1.0, 5.0), (2.5, 20.0), (0.5, 35.0)]      # outside the clicked rectangle too
    err = np.linalg.norm(c.to_metres(m2px(probe)) - np.array(probe), axis=1)
    assert err.max() < 1e-3      # 1 mm


def test_speed_within_error_with_click_noise():
    rng = np.random.default_rng(0)
    rect_m = [(0, 0), (3.6, 0), (3.6, 12), (0, 12)]
    c = Calibration.from_rectangle(m2px(rect_m) + rng.normal(0, 1.5, (4, 2)), 3.6, 12)  # sloppy clicks
    t = np.linspace(0, 2.0, 60)
    true_ms = 13.9                                        # 50 km/h
    track_m = np.stack([np.full_like(t, 1.8), 4 + true_ms * t], 1)
    v, err = speed_kmh(c, t, m2px(track_m) + rng.normal(0, 1.0, (60, 2)))
    assert abs(v - 50) < 4, v            # within ~8% despite noisy clicks
    assert 0 < err < 10


def test_speeding_flag_uses_true_metres():
    """70 km/h in a calibrated scene flags SPEEDING (limit 50); 45 km/h does not."""
    import carwatch
    rect_m = [(0, 0), (3.6, 0), (3.6, 12), (0, 12)]
    c = Calibration.from_rectangle(m2px(rect_m), 3.6, 12)
    fps = 30.0
    t = np.arange(0, 4.0, 1 / fps)

    def flags(kmh):
        ms = kmh / 3.6
        px = m2px(np.stack([np.full_like(t, 1.8), 2 + ms * t], 1))
        v = carwatch.Vehicle(1, "car", 0.0)
        for ti, p in zip(t, px):
            v.add(float(ti), p[0], p[1], 80.0, None)
            v.analyse(50, None, True, None, c)
        return v.ever_flagged

    assert "SPEEDING" in flags(70)
    assert "SPEEDING" not in flags(45)
