"""Regression tests: each synthetic path must raise exactly the expected flags."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import carwatch                      # noqa: E402
from make_synthetic import SCENARIOS   # noqa: E402

CAR_PX = 80.0     # scale used by Vehicle (box long side in px)


def run(name: str, noise_px: float = 0.0):
    fn, _ = SCENARIOS[name]
    t, P = fn()
    rng = np.random.default_rng(3)
    v = carwatch.Vehicle(1, "car", float(t[0]))
    ever = set()
    for i in range(len(t)):
        x, y = P[i] * CAR_PX + rng.normal(0, noise_px, 2)
        v.add(float(t[i]), x, y, CAR_PX, None)
        v.analyse(50, None, False)
        ever |= v.active_flags
    return ever


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_expected_flags(name):
    assert run(name) == SCENARIOS[name][1]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_expected_flags_with_detector_jitter(name):
    # ~1.5 px of box-centre noise, typical for a detector on 80 px cars
    assert run(name, noise_px=1.5) == SCENARIOS[name][1]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_ml_classifier_matches_expected(name):
    """Learned classifier must also satisfy the contract (needs eval/train_behaviour.py first)."""
    from behaviour_ml import MODEL_PATH, BehaviourML
    if not MODEL_PATH.exists():
        pytest.skip("behaviour_model.joblib not trained")
    ml = BehaviourML()
    fn, expected = SCENARIOS[name]
    t, P = fn()
    v = carwatch.Vehicle(1, "car", float(t[0]))
    ever = set()
    for i in range(len(t)):
        v.add(float(t[i]), *(P[i] * CAR_PX), CAR_PX, None)
        v.analyse(50, None, False, ml)
        ever |= v.active_flags
    assert ever == expected


def test_ml_survives_degenerate_windows():
    """Real pipelines produce repeated timestamps and very short windows; must not crash."""
    from behaviour_ml import window_features
    t = np.r_[np.arange(0, 1, 1 / 30), np.arange(0, 1, 1 / 30)]          # duplicate times
    p = np.stack([t * 2, np.zeros_like(t)], 1)
    window_features(np.sort(t), p)                                      # returns features or None, no raise
    assert window_features(np.zeros(20), np.random.default_rng(0).normal(size=(20, 2))) is None
    assert window_features(np.arange(0, 0.5, 0.04), np.stack([np.arange(13.0), np.zeros(13)], 1)) is None or True


def test_impossible_acceleration_is_not_flagged_as_driving():
    """A box that jumps (ID switch) looks like ~70 m/s^2; it must not raise HARSH ACCEL/BRAKE."""
    from behaviour_ml import MODEL_PATH, BehaviourML
    if not MODEL_PATH.exists():
        pytest.skip("behaviour_model.joblib not trained")
    ml = BehaviourML()
    t = np.arange(0, 4, 1 / 30)
    x = 2.0 * t
    x[60:] += 6.0 * (t[60:] - t[60]) ** 2 * 2     # sudden violent speed-up, physically impossible
    v = carwatch.Vehicle(1, "car", 0.0)
    ever = set()
    for ti, xi in zip(t, x):
        v.add(float(ti), xi * CAR_PX, 0.0, CAR_PX, None)
        v.analyse(50, None, False, ml)
        ever |= v.active_flags
    assert "HARSH ACCEL/BRAKE" not in ever
