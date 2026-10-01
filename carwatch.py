#!/usr/bin/env python3
"""
CarWatch v1 - ML vehicle tracking from an iPhone (street level or overhead).

Pipeline
  1. Detection     YOLO26 (street view: COCO car/motorcycle/bus/truck;
                   aerial view: oriented boxes trained on overhead imagery)
  2. Tracking      ByteTrack - persistent IDs across frames
  3. Stabilization Background optical flow cancels handheld / drone camera
                   motion, so trajectories reflect the car, not your hand
  4. Behaviour     Per-vehicle weaving, harsh accel/brake, speed estimate,
                   measured in car-lengths so it works at any zoom or height
  5. Output        Live overlay, counting line, CSV logs, optional video save

Behaviour flags are driving-pattern indicators only. They do not identify
impairment or prove a violation; treat them as "worth a closer look".

Usage
  python carwatch.py --list-cameras            # find your iPhone's index
  python carwatch.py --source 1                # live, street view
  python carwatch.py --source 1 --view aerial  # live, phone pointing down
  python carwatch.py --source clip.mp4 --save out.mp4
  python carwatch.py --source path/to/jpg_folder --view aerial

Keys
  L  draw counting line (click two points)   C  clear line
  T  toggle trails    P  pause    S  screenshot    E  export CSV    Q  quit
"""

import argparse
import csv
import math
import os
import platform
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np

from appearance import SWATCH, color_name
from dashboard import Dashboard
from evidence import EvidenceRecorder
from kinematics import estimate_motion
from speedlog import SpeedStore, fit_model, record_for

try:
    from ultralytics import YOLO
except ImportError:
    print("Missing dependency. Run:  pip install ultralytics lap")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Vehicle classes are picked by NAME from the model, so COCO weights, fine-tuned VisDrone
# weights and DOTA oblique-box weights all work without hard-coded class numbers.
STREET_NAMES = {"car", "van", "truck", "bus", "motorcycle", "motor"}
AERIAL_NAMES = {"large vehicle", "small vehicle"}               # DOTA labels


def pick_classes(model, aerial: bool) -> dict:
    names = AERIAL_NAMES if aerial else STREET_NAMES
    found = {i: ("motorcycle" if n == "motor" else n) for i, n in model.names.items() if n in names}
    if not found:
        raise SystemExit(f"Model has no vehicle classes (looked for {sorted(names)}); it knows: {list(model.names.values())[:12]}")
    return found

CAR_LENGTH_M = 4.5           # used to turn car-lengths into metres
WINDOW_S = 3.0               # analysis window for behaviour metrics
MIN_TRACK_S = 1.5            # ignore brand-new tracks
MOVING_BL_S = 0.6            # below this (car-lengths/s) a car is "not moving"
WEAVE_AMP = 0.25             # peak-to-peak lateral swing, in car lengths (~1.1 m)
WEAVE_CROSSINGS = 3          # direction reversals inside the window
HARSH_ACCEL_BL_S2 = 1.0      # ~4.5 m/s^2 for a 4.5 m car
HARSH_DV_BL_S = 0.8          # speed must actually change ~13 km/h
MAX_PLAUSIBLE_ACCEL_BL_S2 = 6.0   # ~2.7 g at 4.5 m/car: beyond this it is a tracking glitch
FLAG_HOLD = 8                # consecutive evaluations before a flag shows
LOST_TIMEOUT_S = 2.0

# BGR
C_OK = (120, 220, 90)
C_WARN = (0, 190, 255)
C_BAD = (40, 40, 235)
C_LINE = (255, 200, 0)
C_TXT = (235, 235, 235)
C_DIM = (140, 140, 140)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def list_cameras(max_index=6):
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["system_profiler", "SPCameraDataType"],
                                 capture_output=True, text=True, timeout=10).stdout
            names = [ln.strip().rstrip(":") for ln in out.splitlines()
                     if ln.startswith("    ") and ln.strip().endswith(":")
                     and not ln.startswith("      ")]
            if names:
                print("Cameras macOS reports (order usually matches index):")
                for n in names:
                    print(f"   - {n}")
        except Exception:
            pass
    print("\nOpenCV indices that open:")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, fr = cap.read()
            res = f"{fr.shape[1]}x{fr.shape[0]}" if ok else "no frame"
            print(f"   index {i}: {res}")
        cap.release()
    print("\niPhone tip: unlock it, keep it near the Mac on the same Apple ID "
          "with Wi-Fi + Bluetooth on (Continuity Camera). It shows up as an "
          "extra index. Or plug it in with a cable.")


class Source:
    """Webcam index, video file, RTSP URL, or folder of images."""

    def __init__(self, src):
        self.images = None
        self.cap = None
        self.fps = 30.0
        if isinstance(src, str) and os.path.isdir(src):
            exts = (".jpg", ".jpeg", ".png", ".bmp")
            self.images = sorted(os.path.join(src, f) for f in os.listdir(src)
                                 if f.lower().endswith(exts))
            if not self.images:
                raise SystemExit(f"No images in {src}")
            self.i = 0
        else:
            self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open source: {src}")
            if str(src).isdigit():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            f = self.cap.get(cv2.CAP_PROP_FPS)
            if f and 1 < f < 240:
                self.fps = f
        self.live = self.images is None and str(src).isdigit()

    def read(self):
        if self.images is not None:
            if self.i >= len(self.images):
                return None
            fr = cv2.imread(self.images[self.i])
            self.i += 1
            return fr
        ok, fr = self.cap.read()
        return fr if ok else None

    def release(self):
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# Camera-motion compensation
# ---------------------------------------------------------------------------

class Stabilizer:
    """
    Tracks background features between frames and keeps a cumulative
    transform C that maps current-frame pixels into a fixed "world" frame.
    Vehicle boxes are masked out so moving cars don't count as camera motion.
    """

    def __init__(self, enabled=True, work_w=640):
        self.enabled = enabled
        self.work_w = work_w
        self.prev = None
        self.prev_mask = None
        self.C = np.eye(3)
        self.Cinv = np.eye(3)

    def update(self, frame, boxes_xyxy):
        if not self.enabled:
            return
        h, w = frame.shape[:2]
        s = self.work_w / w
        g = cv2.cvtColor(cv2.resize(frame, (self.work_w, int(h * s))),
                         cv2.COLOR_BGR2GRAY)
        mask = np.full(g.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes_xyxy:
            cv2.rectangle(mask, (int(x1 * s) - 4, int(y1 * s) - 4),
                          (int(x2 * s) + 4, int(y2 * s) + 4), 0, -1)

        if self.prev is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev, 300, 0.01, 8,
                                         mask=self.prev_mask)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None)
                good = st.reshape(-1) == 1
                if good.sum() >= 12:
                    A, inl = cv2.estimateAffinePartial2D(
                        p1[good], p0[good], method=cv2.RANSAC,
                        ransacReprojThreshold=2.0)
                    if A is not None and inl is not None and inl.sum() >= 10:
                        S = np.diag([s, s, 1.0])
                        A3 = np.vstack([A, [0, 0, 1]])
                        step = np.linalg.inv(S) @ A3 @ S   # cur -> prev
                        self.C = self.C @ step
                        self.Cinv = np.linalg.inv(self.C)
        self.prev, self.prev_mask = g, mask

    def to_world(self, x, y):
        v = self.C @ np.array([x, y, 1.0])
        return v[0], v[1]

    def to_frame(self, pts):
        if len(pts) == 0:
            return pts
        P = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))])
        return (self.Cinv @ P.T).T[:, :2]


# ---------------------------------------------------------------------------
# Per-vehicle state + behaviour analysis
# ---------------------------------------------------------------------------

def smooth(a, k=5):
    if len(a) < k:
        return a
    ker = np.ones(k) / k
    pad = np.pad(a, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], ker, "valid") for j in range(a.shape[1])], 1)


class Vehicle:
    def __init__(self, tid, label, t):
        self.id = tid
        self.label = label
        self.first_t = t
        self.last_t = t
        self.hist = deque(maxlen=600)        # (t, wx, wy, clean)
        self.lengths = deque(maxlen=60)      # box long side, px (world scale)
        self.poly = None                     # current outline in frame coords
        self.flag_streak = {"WEAVING": 0, "HARSH ACCEL/BRAKE": 0, "SPEEDING": 0}
        self.active_flags = set()
        self.ever_flagged = set()
        self.speed_bl = 0.0                  # car-lengths per second
        self.max_kmh = 0.0
        self.counted = False
        self.last_frame = -1
        self.votes = {}                      # class label majority vote
        self.detail = {}                     # metric values behind the latest ML decision
        self.cal_kmh = None                  # calibrated speed, km/h
        self.speed_samples = []              # km/h while clearly moving, for the speed record
        self.kin_hist = deque(maxlen=600)    # (t, km/h, m/s^2) for evidence plots
        self.over_streak = 0                 # consecutive evaluations above the limit
        self.over_logged = False
        self.motion = None                   # latest speed/accel estimate with errors (kinematics.py)
        self.basis = "approx"                # scale behind it: calibrated / ppm / approx (car-length)
        self.peak_kmh = 0.0
        self.peak_acc = 0.0                  # strongest significant speed-up, m/s^2
        self.peak_dec = 0.0                  # strongest significant braking (negative), m/s^2
        self.color_votes = {}
        self.cal_err = None                  # its rough 1-sigma error, km/h

    @property
    def color(self):
        # need a few agreeing frames before naming a colour, so one bad crop can't label a car
        if sum(self.color_votes.values()) < 3:
            return None
        return max(self.color_votes, key=self.color_votes.get)

    def add(self, t, wx, wy, long_side, poly, clean=True):
        # clean=False when the box is cut off by the frame edge: its centre
        # and size are wrong, so it is drawn but kept out of the metrics.
        self.hist.append((t, wx, wy, clean))
        if clean:
            self.lengths.append(long_side)
        self.last_t = t
        self.poly = poly

    @property
    def scale(self):
        return float(np.median(self.lengths)) if self.lengths else 1.0

    def kmh(self, ppm=None):
        if self.cal_kmh is not None:
            return self.cal_kmh
        if ppm:
            return self.speed_bl * self.scale / ppm * 3.6
        return self.speed_bl * CAR_LENGTH_M * 3.6

    def update_motion(self, H, ppm, calib, speed_limit):
        # Metres: true (calibration) > pixels-per-metre > car-length guess (4.5 m).
        # The guess is per-vehicle and rough, so the overlay marks it with "~".
        xy = H[:, 1:3]
        if calib is not None:
            m, self.basis = calib.to_metres(xy), "calibrated"
        elif ppm:
            m, self.basis = xy / ppm, "ppm"
        else:
            m, self.basis = xy / self.scale * CAR_LENGTH_M, "approx"
        est = estimate_motion(H[:, 0], m)
        if est is None:
            return
        self.motion = est
        self.kin_hist.append((float(H[-1, 0]), est["speed_ms"] * 3.6, est["acc_ms2"]))
        if est["speed_ms"] >= max(0.8, 2 * est["speed_err"]):
            self.speed_samples.append(est["speed_ms"] * 3.6)
        self.peak_kmh = max(self.peak_kmh, est["speed_ms"] * 3.6)
        # Same 10 % margin as the SPEEDING flag, and a hold so one jittery frame can't alert.
        over = est["speed_ms"] * 3.6 > speed_limit * 1.10
        self.over_streak = self.over_streak + 1 if over else 0
        # only record peaks that stand out from the estimate's own noise
        if abs(est["acc_ms2"]) > 2 * est["acc_err"]:
            self.peak_acc = max(self.peak_acc, est["acc_ms2"])
            self.peak_dec = min(self.peak_dec, est["acc_ms2"])

    def analyse(self, speed_limit, ppm, use_kmh, ml=None, calib=None):
        now = self.last_t
        if now - self.first_t < MIN_TRACK_S:
            return
        H = np.array([h[:3] for h in self.hist if now - h[0] <= WINDOW_S and h[3]])
        if len(H) < 12:
            return
        self.update_motion(H, ppm, calib, speed_limit)
        T, P = H[:, 0], smooth(H[:, 1:3] / self.scale, 5)   # car-length units
        dt = np.diff(T)
        dt[dt <= 0] = 1e-3
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) / dt
        vs = smooth(v.reshape(-1, 1), 5).ravel()
        recent = vs[-max(3, len(vs) // 4):]
        self.speed_bl = float(np.median(recent))
        moving = np.percentile(vs, 75) > MOVING_BL_S
        if calib is not None and moving:
            # True metres on the road plane (see calibration.py). Use the last
            # ~1 s: long enough to average out jitter, short enough to track speed.
            recent_h = H[H[:, 0] >= now - 1.0]
            if len(recent_h) >= 6:
                from calibration import speed_kmh
                self.cal_kmh, self.cal_err = speed_kmh(calib, recent_h[:, 0], recent_h[:, 1:3])

        raw = {"WEAVING": False, "HARSH ACCEL/BRAKE": False, "SPEEDING": False}

        if moving:
            # Weaving: lateral residual after removing the smooth path (turns
            # are curved but smooth, and a single lane change is an S-curve;
            # a cubic fit absorbs both, while repeated zig-zags survive it).
            d = P[-1] - P[0]
            if np.linalg.norm(d) > 1e-6:
                u = d / np.linalg.norm(d)
                nrm = np.array([-u[1], u[0]])
                lat = (P - P[0]) @ nrm
                tt = T - T[0]
                resid = lat - np.polyval(np.polyfit(tt, lat, 3), tt)
                sign = np.sign(resid[np.abs(resid) > WEAVE_AMP * 0.2])
                crossings = int(np.sum(sign[1:] != sign[:-1])) if len(sign) > 1 else 0
                amp = np.max(resid) - np.min(resid)
                raw["WEAVING"] = crossings >= WEAVE_CROSSINGS and amp >= WEAVE_AMP

                # Harsh acceleration / braking: along-track speed only, so
                # side-to-side motion is not mistaken for braking.
                # Smooth over ~0.4 s so detector jitter doesn't look like
                # braking, and require a real speed change, not just noise.
                rate = len(T) / max(T[-1] - T[0], 1e-3)
                k = max(5, int(0.4 * rate) | 1)
                along = smooth(((H[:, 1:3] / self.scale - P[0]) @ u).reshape(-1, 1), k).ravel()
                v_long = np.diff(along) / dt
                if len(v_long) > k + 2:
                    v_long = v_long[k // 2: len(v_long) - k // 2]
                    acc = np.diff(v_long) / dt[k // 2 + 1: k // 2 + len(v_long)]
                    acc = smooth(acc.reshape(-1, 1), k).ravel()
                    raw["HARSH ACCEL/BRAKE"] = (
                        np.percentile(np.abs(acc), 90) > HARSH_ACCEL_BL_S2
                        and np.ptp(v_long) > HARSH_DV_BL_S)

            # Optional learned classifier overrides the two rule-based flags.
            # Its feature values are kept so every flag stays explainable.
            if ml is not None:
                out = ml.predict(T - T[0], H[:, 1:3] / self.scale)
                if out is not None:
                    prob, feats = out
                    for name in ("WEAVING", "HARSH ACCEL/BRAKE"):
                        raw[name] = prob[name] > 0.5
                    # Physics gate: no road vehicle sustains more than ~1.3 g (~2.7 car-lengths/s^2),
                    # so a 90th-percentile |accel| far beyond that is a tracker glitch (ID switch,
                    # box jump), not driving. Don't flag it.
                    if feats["acc_p90"] > MAX_PLAUSIBLE_ACCEL_BL_S2:
                        raw["HARSH ACCEL/BRAKE"] = False
                    self.detail = {**feats, **{f"p_{k}": v for k, v in prob.items()}}

            # Speed (only meaningful with a scale: aerial view or --ppm)
            if use_kmh and not (calib is not None and self.cal_kmh is None):
                # (with a calibration, never fall back to the car-length guess)
                k = self.kmh(ppm)
                self.max_kmh = max(self.max_kmh, k)
                raw["SPEEDING"] = k > speed_limit * 1.10

        for name, on in raw.items():
            self.flag_streak[name] = self.flag_streak[name] + 1 if on else max(0, self.flag_streak[name] - 2)
            if self.flag_streak[name] >= FLAG_HOLD:
                self.active_flags.add(name)
                self.ever_flagged.add(name)
            elif self.flag_streak[name] == 0:
                self.active_flags.discard(name)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self):
        self.events = []
        self._seen = set()

    def flag(self, veh, name, extra=""):
        if veh.detail and name not in ("SPEEDING", "OVER LIMIT"):
            extra = (extra + " " + " ".join(f"{k}={v:.2f}" for k, v in veh.detail.items())).strip()
        key = (veh.id, name)
        if key in self._seen:
            return None
        self._seen.add(key)
        row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "vehicle": f"V{veh.id:04d}", "type": veh.label,
               "flag": name, "detail": extra}
        self.events.append(row)
        print(f"[FLAG] {row['time']}  V{veh.id:04d} ({veh.label})  {name}  {extra}")
        return row

    def export(self, vehicles, counts):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ev = f"carwatch_events_{stamp}.csv"
        with open(ev, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["time", "vehicle", "type", "flag", "detail"])
            w.writeheader()
            w.writerows(self.events)
        tr = f"carwatch_tracks_{stamp}.csv"
        with open(tr, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vehicle", "type", "color", "seconds_tracked", "max_kmh_est", "peak_kmh",
                        "peak_accel_ms2", "peak_brake_ms2", "scale_basis", "flags"])
            for v in vehicles:
                w.writerow([f"V{v.id:04d}", v.label, v.color or "", f"{v.last_t - v.first_t:.1f}",
                            f"{v.max_kmh:.0f}" if v.max_kmh else "",
                            f"{v.peak_kmh:.0f}" if v.motion else "",
                            f"{v.peak_acc:.1f}" if v.motion else "",
                            f"{v.peak_dec:.1f}" if v.motion else "",
                            v.basis if v.motion else "",
                            "; ".join(sorted(v.ever_flagged))])
        print(f"[EXPORT] {ev}  {tr}  | line counts: {counts}")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_label(img, text, org, color, scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def vehicle_color(v):
    if len(v.active_flags) >= 2:
        return C_BAD
    if v.active_flags:
        return C_WARN
    return C_OK


def draw_hud(img, info, flagged, paused):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    txt = "   ".join(f"{k} {v}" for k, v in info.items())
    cv2.putText(img, "CARWATCH  " + txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TXT, 1, cv2.LINE_AA)
    if paused:
        draw_label(img, "PAUSED", (w - 90, 60), C_WARN, 0.6)

    if flagged:
        x0, y = w - 300, 50
        ov = img.copy()
        cv2.rectangle(ov, (x0 - 8, y - 18), (w - 6, y + 22 * len(flagged[:8]) - 4), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        for v in flagged[:8]:
            cv2.putText(img, f"V{v.id:04d}  " + ", ".join(sorted(v.active_flags)),
                        (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, vehicle_color(v), 1, cv2.LINE_AA)
            y += 22
    cv2.putText(img, "L line  C clear  T trails  P pause  S shot  E export  Q quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="CarWatch - ML vehicle tracking")
    ap.add_argument("--source", default="0", help="camera index, video, RTSP URL, or image folder")
    ap.add_argument("--view", choices=["street", "aerial"], default="street")
    ap.add_argument("--model", help="override weights (default yolo26s / yolo26s-obb)")
    ap.add_argument("--imgsz", type=int, help="inference size (default 640 street, 1024 aerial)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (auto if omitted)")
    ap.add_argument("--speed-limit", type=float, default=50, help="km/h (Vancouver default is 50)")
    ap.add_argument("--ppm", type=float, help="pixels per metre, for exact speeds in a fixed scene")
    ap.add_argument("--tracker", choices=["bytetrack", "botsort"], default="botsort",
                    help="botsort had ~1/3 the ID switches of bytetrack on VisDrone (eval/RESULTS.md)")
    ap.add_argument("--street", default="", help="street name, stored with each speed record")
    ap.add_argument("--start-time", help='clock time at frame 0 for video files, e.g. "2026-09-30 17:30" (default: now)')
    ap.add_argument("--history", default="speed_history.csv", help="CSV of all speed records, appended across sessions")
    ap.add_argument("--dashboard", action="store_true", help="live charts at http://127.0.0.1:8765 (local only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--evidence", action="store_true",
                    help="save an evidence packet (clip, plots, metrics) per flag; stores footage locally")
    ap.add_argument("--evidence-dir", default="evidence")
    ap.add_argument("--print-speeds", action="store_true",
                    help="print every moving car's speed/accel to the terminal once a second")
    ap.add_argument("--calib", help="scene JSON from calibration.py: true km/h on the road plane")
    ap.add_argument("--no-stabilize", action="store_true")
    ap.add_argument("--rules", action="store_true",
                    help="use the hand-set threshold rules instead of the learned behaviour model "
                         "(rules false-alarmed on 54%% of clean real harsh-braking windows, eval/REAL_RESULTS.md)")
    ap.add_argument("--save", help="write annotated video to this path")
    ap.add_argument("--headless", action="store_true", help="no window (for servers/tests)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--list-cameras", action="store_true")
    a = ap.parse_args()

    if a.list_cameras:
        list_cameras()
        return

    aerial = a.view == "aerial"
    weights = a.model or ("yolo26s-obb.pt" if aerial else "yolo26s.pt")
    imgsz = a.imgsz or (1024 if aerial else 640)
    device = a.device or pick_device()
    # km/h only when there is a real scale: overhead view (car length is
    # visible) or a user-supplied pixels-per-metre calibration.
    use_kmh = aerial or bool(a.ppm) or bool(a.calib)

    print(f"[INIT] {weights}  imgsz={imgsz}  device={device}  view={a.view}")
    model = YOLO(weights)
    classes = pick_classes(model, aerial)
    print(f"[CLASSES] {sorted(set(classes.values()))}")
    src = Source(a.source)
    stab = Stabilizer(enabled=not a.no_stabilize)
    log = Logger()
    calib = None
    if a.calib:
        from calibration import Calibration
        calib = Calibration.load(a.calib)
        print(f"[CALIB] {a.calib}  (click error {calib.rms_error_m:.3f} m). "
              "Calibrated in first-frame pixels; needs stabilization on.")
    ml = None
    from behaviour_ml import MODEL_PATH
    if not a.rules and MODEL_PATH.exists():
        from behaviour_ml import BehaviourML
        ml = BehaviourML()
    print(f"[BEHAVIOUR] {'learned model' if ml else 'threshold rules'}")
    if not ml and not a.rules:
        print("[BEHAVIOUR] no trained model found; train one (synthetic data, no downloads): "
              "python eval/train_behaviour.py")

    store = SpeedStore(a.history)
    dash = Dashboard(a.port, a.evidence_dir if a.evidence else None) if a.dashboard else None
    if dash:
        print(f"[DASH] {dash.url}  (this computer only)")
    base_time = datetime.strptime(a.start_time, "%Y-%m-%d %H:%M") if a.start_time else datetime.now()
    t_first = [None]
    model_cache = {"t": 0.0, "m": None}

    def push_dashboard(live_vehicles):
        if not dash:
            return
        if time.time() - model_cache["t"] > 3:         # refit the regression every few seconds
            model_cache.update(t=time.time(), m=fit_model(store.records))
        live = [{"id": f"V{v.id:04d}", "color": v.color or "", "speed": v.motion["speed_ms"] * 3.6,
                 "over": v.over_streak >= FLAG_HOLD} for v in live_vehicles
                if v.motion and v.motion["speed_ms"] >= max(0.8, 2 * v.motion["speed_err"])]
        from collections import Counter
        counts = Counter(e["flag"] for e in log.events)
        recent = []
        for e in reversed(log.events[-40:]):
            safe = e["flag"].replace("/", "-").replace(" ", "_")
            folder = next((d.name for d in reversed(evid.written) if d.name.endswith(f"_{e['vehicle']}_{safe}")), None) \
                if evid else None
            recent.append({**e, "detail": e["detail"][:70], "evidence": folder})
        meta = {"behaviour": "learned model" if ml else "threshold rules", "tracker": a.tracker,
                "fps": f"{fps_est:.0f}", "scale": "calibrated" if calib else ("ppm" if a.ppm else "estimated")}
        # Cars still in view count as provisional points so the charts fill in live; they only
        # enter the history file once their track ends (emit), so nothing is double-counted.
        prov = [r for r in (record_for(v, a.street or "unspecified",
                                       base_time + timedelta(seconds=v.last_t - t_first[0]), a.speed_limit)
                            for v in live_vehicles) if r]
        dash.update(store.records + prov, a.speed_limit, a.street, live, model_cache["m"],
                    {"counts": dict(counts), "recent": recent}, meta)

    def emit(v):
        r = record_for(v, a.street or "unspecified", base_time + timedelta(seconds=v.last_t - t_first[0]),
                       a.speed_limit)
        if r:
            store.add(r)
            print(f"[RECORD] V{v.id:04d} {r.color} {r.type}  {r.speed_kmh:.0f} km/h  "
                  f"{'OVER' if r.over else 'ok'}  ({r.basis} scale)")

    evid = EvidenceRecorder(a.evidence_dir) if a.evidence else None

    def evidence_for(v, name, row, t_now):
        """Hand a new flag to the recorder with every number that caused it."""
        if not (evid and row):
            return
        m = dict(v.detail)
        if v.motion:
            m.update({"speed_kmh": v.motion["speed_ms"] * 3.6, "speed_err_kmh": v.motion["speed_err"] * 3.6,
                      "accel_ms2": v.motion["acc_ms2"], "accel_err_ms2": v.motion["acc_err"]})
        m.update({"scale_basis": v.basis, "speed_limit_kmh": a.speed_limit,
                  "behaviour_model": "learned" if ml else "threshold rules"})
        traj = np.array([[h[0], h[1] / v.scale, h[2] / v.scale] for h in v.hist if h[3]])
        evid.trigger(t_now, f"V{v.id:04d}", v.label, v.color, name, row["detail"], m, traj, list(v.kin_hist))

    vehicles = {}
    aliases = {}             # tracker ID -> stitched vehicle ID
    finished = []
    line_world = []          # two points in world coords
    clicks = []
    counts = {"A->B": 0, "B->A": 0}
    show_trails = True
    paused = False
    writer = None
    fps_est, t_prev = 0.0, time.time()
    frame_i = 0
    win = "CarWatch"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and param["drawing"]:
            clicks.append((x, y))
            if len(clicks) == 2:
                line_world[:] = [stab.to_world(*p) for p in clicks]
                clicks.clear()
                param["drawing"] = False
                print("[LINE] counting line set")

    mouse_state = {"drawing": False}
    if not a.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, on_mouse, mouse_state)

    frame = None
    while True:
        if not paused or frame is None:
            frame = src.read()
            if frame is None:
                print("[END] source finished")
                break
            frame_i += 1
            # Timestamp: wall clock for live cameras, frame count for files
            t = time.time() if src.live else frame_i / src.fps
            if t_first[0] is None:
                t_first[0] = t

            res = model.track(frame, persist=True, conf=a.conf, imgsz=imgsz,
                              classes=list(classes), tracker=f"{a.tracker}.yaml",
                              device=device, verbose=False)[0]

            dets = []   # (id, label, cx, cy, long_side, poly, xyxy)
            if aerial and res.obb is not None and res.obb.id is not None:
                for xywhr, cls, tid, pts in zip(res.obb.xywhr.cpu().numpy(),
                                                res.obb.cls.cpu().numpy(),
                                                res.obb.id.cpu().numpy(),
                                                res.obb.xyxyxyxy.cpu().numpy()):
                    x, y, w, h, _ = xywhr
                    xs, ys = pts[:, 0], pts[:, 1]
                    dets.append((int(tid), classes.get(int(cls), "vehicle"), x, y,
                                 max(w, h), pts.astype(np.int32),
                                 (xs.min(), ys.min(), xs.max(), ys.max())))
            elif not aerial and res.boxes is not None and res.boxes.id is not None:
                for (x1, y1, x2, y2), cls, tid in zip(res.boxes.xyxy.cpu().numpy(),
                                                      res.boxes.cls.cpu().numpy(),
                                                      res.boxes.id.cpu().numpy()):
                    poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                    dets.append((int(tid), classes.get(int(cls), "vehicle"),
                                 (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1),
                                 poly, (x1, y1, x2, y2)))

            stab.update(frame, [d[6] for d in dets])
            fh, fw = frame.shape[:2]

            for tid, label, cx, cy, long_side, poly, (bx1, by1, bx2, by2) in dets:
                wx, wy = stab.to_world(cx, cy)
                tid = aliases.get(tid, tid)
                v = vehicles.get(tid)
                if v is None:
                    # Track stitching: the detector sometimes drops an ID or
                    # flips car->truck. If a vehicle vanished within the last
                    # second right where this "new" one appears, it's the same car.
                    best, best_d = None, 0.6
                    for old in vehicles.values():
                        if old.last_frame >= frame_i or frame_i - old.last_frame > src.fps:
                            continue
                        d = math.hypot(old.hist[-1][1] - wx, old.hist[-1][2] - wy) / old.scale
                        if d < best_d:
                            best, best_d = old, d
                    if best is not None:
                        aliases[tid] = best.id
                        v = best
                    else:
                        v = vehicles[tid] = Vehicle(tid, label, t)
                v.votes[label] = v.votes.get(label, 0) + 1
                v.label = max(v.votes, key=v.votes.get)
                clean = bx1 > 4 and by1 > 4 and bx2 < fw - 4 and by2 < fh - 4
                v.add(t, wx, wy, long_side, poly, clean)
                if clean and frame_i % 5 == 0:
                    c = color_name(frame, (bx1, by1, bx2, by2))
                    if c:
                        v.color_votes[c] = v.color_votes.get(c, 0) + 1
                v.last_frame = frame_i
                v.analyse(a.speed_limit, a.ppm, use_kmh, ml, calib)
                if v.over_streak >= FLAG_HOLD and not v.over_logged and v.motion:
                    v.over_logged = True
                    m = v.motion
                    note = "scale is a car-length GUESS" if v.basis == "approx" else f"{v.basis} scale"
                    row = log.flag(v, "OVER LIMIT", f"{m['speed_ms'] * 3.6:.0f} +/-{m['speed_err'] * 3.6:.0f} km/h "
                                   f"vs limit {a.speed_limit:.0f} ({note})")
                    evidence_for(v, "OVER LIMIT", row, t)
                for f in v.active_flags:
                    row = log.flag(v, f, (f"~{v.kmh(a.ppm):.0f} km/h"
                                 + (f" +/-{v.cal_err:.0f}" if v.cal_err is not None else "")
                                 + f" (limit {a.speed_limit:.0f})") if f == "SPEEDING" else "")
                    evidence_for(v, f, row, t)

                # counting line crossing (world coords, so camera drift is ok)
                if line_world and not v.counted and len(v.hist) >= 6:
                    (ax, ay), (bx, by) = line_world
                    p_old, p_new = v.hist[-6][1:3], v.hist[-1][1:3]
                    s_old = (bx - ax) * (p_old[1] - ay) - (by - ay) * (p_old[0] - ax)
                    s_new = (bx - ax) * (p_new[1] - ay) - (by - ay) * (p_new[0] - ax)
                    if s_old * s_new < 0:
                        seg_t = ((p_new[0] - ax) * (bx - ax) + (p_new[1] - ay) * (by - ay)) / \
                                ((bx - ax) ** 2 + (by - ay) ** 2 + 1e-9)
                        if 0 <= seg_t <= 1:
                            counts["A->B" if s_new > 0 else "B->A"] += 1
                            v.counted = True

            for tid in [k for k, v in vehicles.items() if t - v.last_t > LOST_TIMEOUT_S]:
                finished.append(vehicles.pop(tid))
                emit(finished[-1])

            if a.print_speeds and frame_i % max(int(src.fps), 1) == 0:
                rows = sorted((v for v in vehicles.values() if v.last_frame == frame_i and v.motion
                               and v.motion["speed_ms"] >= max(0.8, 2 * v.motion["speed_err"])),
                              key=lambda v: -v.motion["speed_ms"])
                if rows:
                    tag = "est " if any(v.basis == "approx" for v in rows) else ""
                    print(f"[SPEED t={t:5.1f}s] " + " | ".join(
                        f"V{v.id:04d} {v.color or ''} {v.label} {tag}{v.motion['speed_ms'] * 3.6:.0f} km/h "
                        f"a{v.motion['acc_ms2']:+.1f}" + (" OVER" if v.over_streak >= FLAG_HOLD else "")
                        for v in rows))

            if dash and time.time() - model_cache.get("push", 0) > 0.5:
                model_cache["push"] = time.time()
                push_dashboard([v for v in vehicles.values() if v.last_frame == frame_i])

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

        # ---- draw ----
        out = frame.copy()
        visible = [v for v in vehicles.values() if v.last_frame == frame_i]
        for v in visible:
            col = vehicle_color(v)
            if show_trails and len(v.hist) > 2:
                pts = stab.to_frame([h[1:3] for h in list(v.hist)[-90:]]).astype(np.int32)
                cv2.polylines(out, [pts], False, col, 2, cv2.LINE_AA)
            cv2.polylines(out, [v.poly], True, col, 2 if not v.active_flags else 3, cv2.LINE_AA)
            x0, y0 = int(v.poly[:, 0].min()), int(v.poly[:, 1].min())
            name = f"V{v.id:04d} {(v.color + ' ') if v.color else ''}{v.label}"
            ly = max(y0 - 4, 48)
            if v.motion:
                m = v.motion
                # OpenCV's font has no "~" (it prints as "-", which reads as a minus
                # sign), so an estimated scale is marked with the word "est".
                est = "est " if v.basis == "approx" else ""
                if m["speed_ms"] < max(0.8, 2 * m["speed_err"]):     # jitter, not motion
                    kin = "stopped"
                else:
                    kin = f"{est}{m['speed_ms'] * 3.6:.0f} km/h  a {m['acc_ms2']:+.1f} m/s2"
                draw_label(out, kin, (x0, ly), col, 0.45)
                ly -= 20
            draw_label(out, name, (x0, ly), col, 0.45)
            if v.color:                                   # small chip in the car's own colour
                cv2.rectangle(out, (x0 - 14, ly - 14), (x0 - 2, ly - 2), SWATCH[v.color], -1)
                cv2.rectangle(out, (x0 - 14, ly - 14), (x0 - 2, ly - 2), (255, 255, 255), 1)

        if line_world:
            (p, q) = stab.to_frame(line_world).astype(int)
            cv2.line(out, tuple(p), tuple(q), C_LINE, 2, cv2.LINE_AA)
            draw_label(out, "A", tuple(p), C_LINE)
            draw_label(out, "B", tuple(q), C_LINE)
        for c in clicks:
            cv2.circle(out, c, 5, C_LINE, -1)

        flagged = [v for v in visible if v.active_flags]
        info = {"view": a.view, "fps": f"{fps_est:.0f}", "tracking": len(visible),
                "flagged": len(flagged)}
        if line_world:
            info["count"] = f"{counts['A->B']}/{counts['B->A']}"
        draw_hud(out, info, flagged, paused)
        if mouse_state["drawing"]:
            draw_label(out, "Click two points for the counting line", (10, 64), C_LINE, 0.55)

        if evid and not paused:
            evid.push(out, t)

        if a.save:
            if writer is None:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src.fps, (out.shape[1], out.shape[0]))
            if not paused:
                writer.write(out)

        if a.headless:
            if a.max_frames and frame_i >= a.max_frames:
                break
            continue

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("l"):
            clicks.clear()
            mouse_state["drawing"] = True
        elif key == ord("c"):
            line_world.clear()
            counts.update({"A->B": 0, "B->A": 0})
        elif key == ord("t"):
            show_trails = not show_trails
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            fn = f"carwatch_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(fn, out)
            print(f"[SHOT] {fn}")
        elif key == ord("e"):
            log.export(finished + list(vehicles.values()), counts)

    for v in list(vehicles.values()):
        emit(v)
    push_dashboard([])
    if evid:
        evid.close()
    if store.records:
        from dashboard import make_chart
        rep = make_chart(store.records, a.speed_limit, f"carwatch_report_{datetime.now():%Y%m%d_%H%M%S}.html",
                         a.street)
        print(f"[REPORT] {rep}  ({len(store.records)} vehicles in history)")
    src.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    log.export(finished + list(vehicles.values()), counts)
    print(f"[DONE] {len(finished) + len(vehicles)} vehicles tracked, {len(log.events)} flags")


if __name__ == "__main__":
    main()
