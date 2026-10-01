"""
Does training on REAL vehicle tracks (with injected weaving / hard braking) beat
training on synthetic paths only? Leave-one-sequence-out on VisDrone ground truth.

Windows: 3 s of a real track, only when the car is moving (same gate as carwatch).
Clean real windows count as "normal"; weaving / hard braking is injected with known
labels. We report, per method: recall on injected events and false-alarm rate on
CLEAN real windows. Caveats: GT tracks are human-labelled (smoother than detector
output, so jitter is added), real windows may contain real events we label normal,
and injected events are still my definition of the events.
  python eval/real_augment.py            # evaluate
  python eval/real_augment.py --ship     # retrain final model on all sequences + synthetic
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import behaviour_ml as B                       # noqa: E402
import carwatch                                # noqa: E402
from real_tracks import FPS, load              # noqa: E402
from train_behaviour import rule_predict       # noqa: E402

WIN = int(3.0 * FPS)
STRIDE = int(0.5 * FPS)


def windows(tracks_by_seq):
    """{seq: [(T, P_in_car_lengths)]} for moving windows of continuous frames."""
    out = {}
    for seq, tracks in tracks_by_seq.items():
        ws = []
        for arr in tracks.values():
            fr = arr[:, 0]
            cuts = np.where(np.diff(fr) != 1)[0] + 1            # split at gaps
            for seg in np.split(arr, cuts):
                for a in range(0, len(seg) - WIN + 1, STRIDE):
                    w = seg[a:a + WIN]
                    scale = float(np.median(w[:, 3]))
                    T, P = (w[:, 0] - w[0, 0]) / FPS, w[:, 1:3] / scale
                    v = np.linalg.norm(np.diff(P, axis=0), axis=1) * FPS
                    if np.percentile(v, 75) > carwatch.MOVING_BL_S and B.window_features(T, P) is not None:
                        ws.append((T, P))
        out[seq] = ws
    return out


def jitter(P, rng):
    return P + rng.normal(0, rng.uniform(0.005, 0.03), P.shape)   # detector noise GT lacks


def build(ws, rng, with_events=True):
    """Feature rows + labels (weave, harsh) from real windows (+ injected variants)."""
    X, yw, yh, P_all, kinds = [], [], [], [], []
    for T, P in ws:
        variants = [("clean", P, 0, 0)]
        if with_events:
            pw, ph = B.inject_weave(T, P, rng), B.inject_harsh(T, P, rng)
            if pw is not None:
                variants.append(("weave", pw, 1, 0))
            if ph is not None:
                variants.append(("harsh", ph, 0, 1))
        for kind, Q, w, h in variants:
            Q = jitter(Q, rng)
            f = B.window_features(T, Q)
            if f is not None:
                X.append(f); yw.append(w); yh.append(h); P_all.append((T, Q)); kinds.append(kind)
    return np.array(X), np.array(yw), np.array(yh), P_all, np.array(kinds)


def fit(X, yw, yh):
    mk = lambda: HistGradientBoostingClassifier(max_iter=200, random_state=0)   # noqa: E731
    return mk().fit(X, yw), mk().fit(X, yh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", action="store_true")
    a = ap.parse_args()
    W = windows(load())
    print({k: len(v) for k, v in W.items()}, flush=True)
    Xs, ws_, hs_ = B.make_dataset(4000, seed=10)                 # synthetic training data
    syn = fit(Xs, ws_, hs_)
    shipped = joblib.load(B.MODEL_PATH)
    shipped = (shipped["weave"], shipped["harsh"])

    if a.ship:
        rng = np.random.default_rng(0)
        Xr, yw, yh, _, _ = build([w for v in W.values() for w in v], rng)
        Xm, ym_w, ym_h = np.vstack([Xr, Xs]), np.r_[yw, ws_], np.r_[yh, hs_]
        wm, hm = fit(Xm, ym_w, ym_h)
        joblib.dump({"weave": wm, "harsh": hm}, B.MODEL_PATH)
        print(f"shipped real+synthetic model ({len(Xr)} real-background rows + {len(Xs)} synthetic)")
        return

    res = {m: {"weave": [[], []], "harsh": [[], []]} for m in ("Rules", "Synthetic-only ML", "Real-aug ML", "Real+synthetic ML")}
    # res[m][flag] = [clean predictions, injected predictions]
    for held in W:
        train = [w for s, v in W.items() if s != held for w in v]
        rng = np.random.default_rng(1)
        Xr, yw, yh, _, _ = build(train, rng)
        real = fit(Xr, yw, yh)
        both = fit(np.vstack([Xr, Xs]), np.r_[yw, ws_], np.r_[yh, hs_])
        Xt, tw, th, PT, kinds = build(W[held], np.random.default_rng(99))
        models = {"Synthetic-only ML": syn, "Real-aug ML": real, "Real+synthetic ML": both}
        for name, (mw, mh) in models.items():
            for flag, m in (("weave", mw), ("harsh", mh)):
                pred = m.predict_proba(Xt)[:, 1] > 0.5
                res[name][flag][0] += list(pred[kinds == "clean"])
                res[name][flag][1] += list(pred[kinds == flag])
        rp = np.array([rule_predict(T, P) for T, P in PT])
        for i, flag in enumerate(("weave", "harsh")):
            res["Rules"][flag][0] += list(rp[kinds == "clean", i].astype(bool))
            res["Rules"][flag][1] += list(rp[kinds == flag, i].astype(bool))
        print("fold done:", held, flush=True)

    out = ["# Real-track evaluation — leave-one-sequence-out on VisDrone ground truth", "",
           "Windows are 3 s of real moving vehicle tracks (stabilized). Clean windows = normal traffic; "
           "weaving / hard braking is INJECTED with known labels. Recall is on injected events; "
           "false-alarm rate is on clean real windows (some of which may contain real events, so it is "
           "an upper bound on the true false-alarm rate). Each fold trains on 5 sequences and tests on the 6th.",
           "", f"Clean windows: {len(res['Rules']['weave'][0])}; injected weave: {len(res['Rules']['weave'][1])}; "
           f"injected hard brake/accel: {len(res['Rules']['harsh'][1])}.", "",
           "| Method | Weave recall | Weave false alarms (clean) | Harsh recall | Harsh false alarms (clean) |",
           "|---|---|---|---|---|"]
    for name, r in res.items():
        f = lambda x: float(np.mean(x)) if len(x) else float("nan")   # noqa: E731
        out.append(f"| {name} | {f(r['weave'][1]):.3f} | {f(r['weave'][0]):.3f} | "
                   f"{f(r['harsh'][1]):.3f} | {f(r['harsh'][0]):.3f} |")
    (ROOT / "eval" / "REAL_RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
