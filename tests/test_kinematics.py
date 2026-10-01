import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from appearance import color_name                 # noqa: E402
from kinematics import estimate_motion            # noqa: E402

FPS = 30.0


def track(v0, a, noise=0.05, seconds=2.0, seed=0):
    t = np.arange(0, seconds, 1 / FPS)
    s = v0 * t + 0.5 * a * t**2
    rng = np.random.default_rng(seed)
    xy = np.stack([s, 0.3 * np.ones_like(s)], 1) + rng.normal(0, noise, (len(t), 2))
    return t, xy


def test_constant_speed():
    m = estimate_motion(*track(10.0, 0.0))
    assert abs(m["speed_ms"] - 10) < 3 * m["speed_err"] + 0.3
    assert abs(m["acc_ms2"]) < 3 * m["acc_err"] + 0.3


def test_hard_brake_and_acceleration_recovered():
    for a in (-5.0, 2.5):
        errs = []
        for seed in range(20):
            m = estimate_motion(*track(14.0, a, seed=seed))
            errs.append(m["acc_ms2"] - a)
        assert abs(np.mean(errs)) < 0.3                # unbiased
        assert np.std(errs) < 1.5                      # 5 cm jitter -> roughly +/-1 m/s^2


def test_reported_error_is_honest():
    """~68% of estimates should fall within 1 reported sigma."""
    hits = 0
    for seed in range(200):
        m = estimate_motion(*track(12.0, -3.0, seed=seed))
        hits += abs(m["acc_ms2"] + 3.0) <= m["acc_err"]
    assert 0.5 < hits / 200 < 0.85


def test_stationary_and_short_tracks():
    t = np.arange(0, 2, 1 / FPS)
    assert estimate_motion(t, np.zeros((len(t), 2)) + 0.01)["speed_ms"] == 0.0
    assert estimate_motion(t[:5], np.zeros((5, 2))) is None


def test_colour_names():
    def patch(bgr):
        return np.full((100, 100, 3), bgr, np.uint8)
    cases = {"red": (30, 30, 200), "blue": (200, 80, 20), "white": (240, 240, 240),
             "black": (15, 15, 15), "green": (60, 170, 50), "silver": (170, 170, 170)}
    for name, bgr in cases.items():
        assert color_name(patch(bgr), (0, 0, 100, 100)) == name
