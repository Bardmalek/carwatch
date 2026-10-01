"""
Train the learned behaviour classifiers on synthetic windows and compare with the
rule-based baseline on (a) held-out in-distribution and (b) shifted windows
(more noise, larger weaves). Writes behaviour_model.joblib and eval/BEHAVIOUR_RESULTS.md.
Synthetic only: this is NOT a real-world accuracy figure.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import behaviour_ml as B      # noqa: E402
import carwatch               # noqa: E402

CAR_PX = 80.0


def rule_predict(T, P):
    """Run the existing rules on one window (streak requirement disabled)."""
    old = (carwatch.FLAG_HOLD, carwatch.MIN_TRACK_S)
    carwatch.FLAG_HOLD, carwatch.MIN_TRACK_S = 1, 0.0
    try:
        v = carwatch.Vehicle(1, "car", float(T[0]))
        for t, p in zip(T, P * CAR_PX):
            v.add(float(t), p[0], p[1], CAR_PX, None)
        v.analyse(50, None, False)
        return int("WEAVING" in v.ever_flagged), int("HARSH ACCEL/BRAKE" in v.ever_flagged)
    finally:
        carwatch.FLAG_HOLD, carwatch.MIN_TRACK_S = old


def score(y, p):
    return (precision_score(y, p, zero_division=0), recall_score(y, p, zero_division=0))


def main():
    Xtr, wtr, htr = B.make_dataset(6000, seed=0)
    weave = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xtr, wtr)
    harsh = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xtr, htr)
    joblib.dump({"weave": weave, "harsh": harsh}, B.MODEL_PATH)

    rows = []
    for name, shift, seed in [("held-out (same distribution)", 1.0, 1),
                              ("shifted (2x noise, bigger weaves)", 2.0, 2)]:
        rng = np.random.default_rng(seed)
        yw, yh, ml_w, ml_h, ru_w, ru_h = [], [], [], [], [], []
        while len(yw) < 1500:
            T, P, w, h = B.synth_window(rng, shift)
            f = B.window_features(T, P)
            if f is None:
                continue
            yw.append(int(w)); yh.append(int(h))
            ml_w.append(int(weave.predict_proba([f])[0, 1] > 0.5))
            ml_h.append(int(harsh.predict_proba([f])[0, 1] > 0.5))
            rw, rh = rule_predict(T, P)
            ru_w.append(rw); ru_h.append(rh)
        for flag, y, ml, ru in [("WEAVING", yw, ml_w, ru_w), ("HARSH ACCEL/BRAKE", yh, ml_h, ru_h)]:
            rows.append((name, flag, score(y, ml), score(y, ru)))

    out = ["# Behaviour classifier — synthetic evaluation", "",
           "Trained and tested on randomised SYNTHETIC trajectories only (3 s windows). "
           "This shows the learned model matches the rules' intent; it is not a real-world "
           "accuracy figure. Real numbers need labelled real footage.", "",
           "| Test set | Flag | ML precision | ML recall | Rules precision | Rules recall |",
           "|---|---|---|---|---|---|"]
    for name, flag, (mp, mr), (rp, rr) in rows:
        out.append(f"| {name} | {flag} | {mp:.3f} | {mr:.3f} | {rp:.3f} | {rr:.3f} |")
    (ROOT / "eval" / "BEHAVIOUR_RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
