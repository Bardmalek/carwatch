"""
Road-plane calibration: pixels -> metres via a homography.

Click 4+ points on the road whose real spacing you know (e.g. the corners of a
lane segment: lane width 3.5-3.7 m, length from dashed-line spacing). The
homography replaces the "car = 4.5 m" guess with true metres on the road plane.

Points are given in FIRST-FRAME pixels. carwatch's stabilizer maps every frame
into that same world frame, so calibration stays valid when the camera moves.
It is only valid for things on the road plane; a box centre is above the road,
so street-view speeds carry a bias the error estimate does not include.

  python calibration.py --source clip.mp4 --width 3.6 --length 12 --out scene.json
Click order: near-left, near-right, far-right, far-left of the known rectangle.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    H: np.ndarray                 # 3x3, first-frame pixels -> metres
    rms_error_m: float            # RMS reprojection error of the clicked points, metres
    n_points: int

    @classmethod
    def from_points(cls, px, metres) -> "Calibration":
        px, m = np.asarray(px, np.float64), np.asarray(metres, np.float64)
        if len(px) < 4 or len(px) != len(m):
            raise ValueError("need >= 4 matching point pairs")
        H, _ = cv2.findHomography(px, m, 0)          # least squares over all points
        if H is None:
            raise ValueError("points are degenerate (collinear?)")
        c = cls(H, 0.0, len(px))
        c.rms_error_m = float(np.sqrt(np.mean(np.sum((c.to_metres(px) - m) ** 2, 1))))
        return c

    @classmethod
    def from_rectangle(cls, px, width_m: float, length_m: float) -> "Calibration":
        """px order: near-left, near-right, far-right, far-left."""
        return cls.from_points(px, [(0, 0), (width_m, 0), (width_m, length_m), (0, length_m)])

    def to_metres(self, pts) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, np.float64))
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        return q

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"H": self.H.tolist(), "rms_error_m": self.rms_error_m,
                                          "n_points": self.n_points}, indent=2))

    @classmethod
    def load(cls, path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["H"]), d["rms_error_m"], d["n_points"])


def speed_kmh(calib: Calibration, t, xy_px, pixel_sigma: float = 2.0) -> tuple[float, float]:
    """Mean speed over a track segment (km/h) and a rough 1-sigma error (km/h).

    Error comes from propagating `pixel_sigma` of position noise at both ends of
    the segment through the homography; it does not include road-plane bias."""
    t, P = np.asarray(t, float), calib.to_metres(xy_px)
    span = t[-1] - t[0]
    if span <= 0 or len(P) < 2:
        return 0.0, float("inf")
    dist = np.linalg.norm(P[-1] - P[0])
    v = dist / span * 3.6
    # local metres-per-pixel at both ends, via finite difference of the homography
    e = []
    for p in (np.asarray(xy_px[0], float), np.asarray(xy_px[-1], float)):
        base = calib.to_metres(p)[0]
        j = np.stack([calib.to_metres(p + d)[0] - base for d in ([1, 0], [0, 1])], 1)
        e.append(np.linalg.norm(j, 2) * pixel_sigma)
    err = float(np.hypot(*e)) / span * 3.6
    return float(v), err


def click_points(frame: np.ndarray, n: int = 4) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    win = "calibrate: click near-left, near-right, far-right, far-left (Q abort)"
    img = frame.copy()

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv2.circle(img, (x, y), 6, (0, 200, 255), -1)
            cv2.putText(img, str(len(pts)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, cb)
    while len(pts) < n:
        cv2.imshow(win, img)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
    cv2.imshow(win, img); cv2.waitKey(400); cv2.destroyAllWindows()
    return pts


def main():
    ap = argparse.ArgumentParser(description="Create a per-scene calibration JSON")
    ap.add_argument("--source", required=True, help="video file or image folder (first frame is used)")
    ap.add_argument("--width", type=float, required=True, help="rectangle width in metres (e.g. lane 3.6)")
    ap.add_argument("--length", type=float, required=True, help="rectangle length in metres")
    ap.add_argument("--out", default="scene.json")
    a = ap.parse_args()
    if Path(a.source).is_dir():
        first = sorted(p for p in Path(a.source).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[0]
        frame = cv2.imread(str(first))
    else:
        cap = cv2.VideoCapture(a.source); ok, frame = cap.read(); cap.release()
        if not ok:
            raise SystemExit(f"cannot read {a.source}")
    c = Calibration.from_rectangle(click_points(frame), a.width, a.length)
    c.save(a.out)
    print(f"saved {a.out}  (rms point error {c.rms_error_m:.3f} m)")


if __name__ == "__main__":
    main()
