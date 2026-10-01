"""
Evidence packets: for every flag, a folder with
  clip.mp4        5 s before to 5 s after the moment the flag fired (annotated frames)
  trajectory.png  the vehicle's path, plus speed and acceleration over time
  packet.json     the exact metric values behind the flag, scale basis, model used
  summary.html    one page tying it together, with the disclaimer
Stays on disk locally. Opt-in (`carwatch.py --evidence`) because it stores footage.
"""
from __future__ import annotations

import base64
import html
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DISCLAIMER = ("Driving-pattern indicator only. It does not identify impairment and does not prove a "
              "violation; it marks a vehicle worth a closer look by a person.")


class EvidenceRecorder:
    def __init__(self, out_dir="evidence", pre_s=5.0, post_s=5.0, max_width=960, quality=80):
        self.dir, self.pre_s, self.post_s = Path(out_dir), pre_s, post_s
        self.max_width, self.quality = max_width, quality
        self.buf: deque = deque()            # (t, jpeg bytes)
        self.pending: list[dict] = []
        self.written: list[Path] = []

    def _encode(self, frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()

    def push(self, frame: np.ndarray, t: float) -> None:
        jpg = self._encode(frame)
        self.buf.append((t, jpg))
        while self.buf and self.buf[0][0] < t - self.pre_s:
            self.buf.popleft()
        for p in self.pending:
            p["post"].append((t, jpg))
        for p in [p for p in self.pending if t >= p["until"]]:
            self._write(p)
            self.pending.remove(p)

    def trigger(self, t: float, vehicle: str, label: str, color: str | None, flag: str,
                detail: str, metrics: dict, traj: np.ndarray, kin: list) -> None:
        """traj: Nx3 (t, x, y) in car-lengths; kin: [(t, km/h, m/s^2)]."""
        self.pending.append({"t": t, "until": t + self.post_s, "pre": list(self.buf), "post": [],
                             "vehicle": vehicle, "label": label, "color": color or "unknown",
                             "flag": flag, "detail": detail, "metrics": metrics,
                             "traj": np.asarray(traj, float), "kin": list(kin)})

    def close(self) -> None:
        for p in self.pending:
            self._write(p)
        self.pending.clear()

    # ---- writing ----
    def _write(self, p: dict) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = p["flag"].replace("/", "-").replace(" ", "_")
        d = self.dir / f"{stamp}_{p['vehicle']}_{safe}"
        d.mkdir(parents=True, exist_ok=True)
        frames = p["pre"] + [f for f in p["post"] if not p["pre"] or f[0] > p["pre"][-1][0]]
        if frames:
            imgs = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for _, j in frames]
            span = max(frames[-1][0] - frames[0][0], 1e-3)
            fps = min(60.0, max(5.0, (len(frames) - 1) / span))
            vw = cv2.VideoWriter(str(d / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (imgs[0].shape[1], imgs[0].shape[0]))
            for im in imgs:
                vw.write(im)
            vw.release()
        cv2.imwrite(str(d / "trajectory.png"), self._plot(p))
        meta = {"vehicle": p["vehicle"], "type": p["label"], "color": p["color"], "flag": p["flag"],
                "detail": p["detail"], "metrics": p["metrics"], "frames_in_clip": len(frames),
                "written": datetime.now().isoformat(timespec="seconds"), "disclaimer": DISCLAIMER}
        (d / "packet.json").write_text(json.dumps(meta, indent=2))
        (d / "summary.html").write_text(self._html(meta, d / "trajectory.png"), encoding="utf-8")
        self.written.append(d)
        print(f"[EVIDENCE] {d}")

    @staticmethod
    def _plot(p: dict, w=720, h=620) -> np.ndarray:
        img = np.full((h, w, 3), 250, np.uint8)
        ink, dim = (40, 40, 40), (140, 140, 140)
        # path panel (top): equal axes, colour runs blue -> red with time
        tr = p["traj"]
        x0, y0, pw, ph = 40, 30, w - 80, 320
        cv2.putText(img, f"{p['vehicle']} {p['color']} {p['label']}  |  {p['flag']}", (x0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ink, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), dim, 1)
        if len(tr) > 2:
            xy = tr[:, 1:3] - tr[:, 1:3].mean(0)
            span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0)
            pts = (xy / span * (min(pw, ph) - 40) + [pw / 2, ph / 2] + [x0, y0]).astype(np.int32)
            for i in range(len(pts) - 1):
                f = i / max(len(pts) - 2, 1)
                cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (int(255 * (1 - f)), 60, int(255 * f)), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[0]), 5, (200, 120, 0), -1)
            cv2.circle(img, tuple(pts[-1]), 5, (0, 0, 200), -1)
            cv2.putText(img, "path (car-lengths, north-up not implied). blue=start  red=end", (x0, y0 + ph + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, dim, 1, cv2.LINE_AA)
        # speed and acceleration over time (bottom)
        kin = np.array(p["kin"], float).reshape(-1, 3)
        by = y0 + ph + 40
        for k, (col, name, unit) in enumerate(((1, "speed", "km/h"), (2, "acceleration", "m/s2"))):
            bx0, bw, bh = x0 + k * (pw // 2 + 10), pw // 2 - 10, 190
            cv2.rectangle(img, (bx0, by), (bx0 + bw, by + bh), dim, 1)
            cv2.putText(img, f"{name} ({unit})", (bx0, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ink, 1, cv2.LINE_AA)
            if len(kin) > 1:
                t, v = kin[:, 0], kin[:, col]
                lo, hi = min(v.min(), 0), max(v.max(), 1e-6 + (0 if col == 1 else 1))
                xs = (bx0 + (t - t[0]) / max(t[-1] - t[0], 1e-6) * bw).astype(int)
                ys = (by + bh - (v - lo) / max(hi - lo, 1e-6) * bh).astype(int)
                cv2.polylines(img, [np.stack([xs, ys], 1)], False, (60, 60, 200), 2, cv2.LINE_AA)
                cv2.putText(img, f"{hi:.0f}" if col == 1 else f"{hi:+.1f}", (bx0 + 3, by + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)
                cv2.putText(img, f"{lo:.0f}" if col == 1 else f"{lo:+.1f}", (bx0 + 3, by + bh - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)
        cv2.putText(img, "Driving-pattern indicator only - not proof of impairment or a violation.",
                    (x0, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dim, 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _html(meta: dict, png: Path) -> str:
        b64 = base64.b64encode(png.read_bytes()).decode()
        rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(f'{v:.3f}' if isinstance(v, float) else str(v))}</td></tr>"
                       for k, v in meta["metrics"].items())
        return (f"<!doctype html><meta charset=utf-8><title>Evidence {html.escape(meta['vehicle'])}</title>"
                "<style>body{font:14px system-ui;max-width:860px;margin:20px auto;padding:0 12px}"
                "td{padding:2px 10px;border-bottom:1px solid #ddd}video,img{max-width:100%}.d{background:#fff4d6;padding:8px 12px;border-radius:6px}</style>"
                f"<h2>{html.escape(meta['vehicle'])} &middot; {html.escape(meta['color'])} {html.escape(meta['type'])} &middot; {html.escape(meta['flag'])}</h2>"
                f"<p class=d>{html.escape(meta['disclaimer'])}</p>"
                f"<video src=clip.mp4 controls></video><p>{html.escape(meta['detail'])}</p>"
                f"<img alt=trajectory src='data:image/png;base64,{b64}'>"
                f"<h3>Metric values behind the flag</h3><table>{rows}</table>")
