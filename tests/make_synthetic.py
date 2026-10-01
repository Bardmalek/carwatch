"""
Synthetic vehicle scenarios for the behaviour regression suite.

Each scenario is a path in car-length units at 30 fps. Tests feed the path to
carwatch.Vehicle directly (fast, deterministic, no detector needed). The same
paths can also be rendered to mp4 clips with `python tests/make_synthetic.py`
for eyeballing the full pipeline.

Expected flags are the contract: turns and single lane changes must NOT flag.
"""
import sys
from pathlib import Path

import numpy as np

FPS = 30.0
SECONDS = 8.0
PX_PER_BL = 80.0            # pixels per car length in rendered clips
CRUISE = 2.0                # car-lengths per second (~32 km/h at 4.5 m)


def _t() -> np.ndarray:
    return np.arange(0, SECONDS, 1 / FPS)


def straight():
    t = _t()
    return t, np.stack([CRUISE * t, np.zeros_like(t)], 1)


def weave():
    # 0.6 car-length swing, ~1.1 s period: clearly zig-zag
    t = _t()
    return t, np.stack([CRUISE * t, 0.3 * np.sin(2 * np.pi * t / 1.1)], 1)


def brake():
    # cruise, then hard stop between t=3 s and 4.2 s
    t = _t()
    v = np.where(t < 3, 3.5, np.clip(3.5 - (t - 3) * 3.5 / 1.2, 0, None))
    return t, np.stack([np.cumsum(v) / FPS, np.zeros_like(t)], 1)


def lane_change():
    # one smooth 0.8 car-length lateral shift over 1.5 s
    t = _t()
    s = np.clip((t - 3) / 1.5, 0, 1)
    y = 0.8 * (3 * s**2 - 2 * s**3)
    return t, np.stack([CRUISE * t, y], 1)


def turn():
    # constant-speed 90 degree arc, then straight along the new heading
    t = _t()
    radius = 6.0
    t_arc = radius * np.pi / 2 / CRUISE
    ang = np.clip(CRUISE * t / radius, 0, np.pi / 2)
    after = np.clip(t - t_arc, 0, None) * CRUISE
    return t, np.stack([radius * np.sin(ang), radius * (1 - np.cos(ang)) + after], 1)


def slow_stop():
    # gentle, comfortable stop: must not read as harsh braking
    t = _t()
    v = np.clip(CRUISE * (1 - t / SECONDS * 1.0), 0, None)
    return t, np.stack([np.cumsum(v) / FPS, np.zeros_like(t)], 1)


SCENARIOS = {
    "straight": (straight, set()),
    "weave": (weave, {"WEAVING"}),
    "brake": (brake, {"HARSH ACCEL/BRAKE"}),
    "lane_change": (lane_change, set()),
    "turn": (turn, set()),
    "slow_stop": (slow_stop, set()),
}


def shake(t: np.ndarray, seed: int = 0, amp_px: float = 6.0) -> np.ndarray:
    """Handheld-camera offsets in pixels, smooth-ish random walk."""
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (len(t), 2))
    k = np.ones(5) / 5
    n = np.stack([np.convolve(n[:, j], k, "same") for j in range(2)], 1)
    return n / n.std() * amp_px


def render(name: str, out_dir: Path, with_shake: bool = False) -> Path:
    import cv2
    fn, _ = SCENARIOS[name]
    t, P = fn()
    w, h = 1280, 720
    off = shake(t) if with_shake else np.zeros((len(t), 2))
    path = out_dir / f"{name}{'_shake' if with_shake else ''}.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    rng = np.random.default_rng(1)
    bg = rng.integers(60, 140, (h, w, 3), dtype=np.uint8)   # textured "road" for optical flow
    for i in range(len(t)):
        M = np.float32([[1, 0, off[i, 0]], [0, 1, off[i, 1]]])
        fr = cv2.warpAffine(bg, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        cx, cy = 100 + P[i, 0] * PX_PER_BL + off[i, 0], h / 2 + P[i, 1] * PX_PER_BL + off[i, 1]
        cv2.rectangle(fr, (int(cx - 40), int(cy - 20)), (int(cx + 40), int(cy + 20)), (30, 30, 200), -1)
        vw.write(fr)
    vw.release()
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/clips")
    out.mkdir(parents=True, exist_ok=True)
    for n in SCENARIOS:
        for s in (False, True):
            print(render(n, out, s))
