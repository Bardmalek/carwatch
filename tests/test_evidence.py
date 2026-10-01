import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence import EvidenceRecorder   # noqa: E402


def test_packet_written_with_pre_and_post_frames(tmp_path):
    rec = EvidenceRecorder(tmp_path, pre_s=1.0, post_s=1.0)
    t = np.arange(0, 4, 1 / 15)
    traj = np.stack([t, 2 * t, 0.3 * np.sin(6 * t)], 1)
    kin = [(float(ti), 30 + 5 * ti, float(np.sin(ti))) for ti in t]
    for i, ti in enumerate(t):
        frame = np.full((180, 320, 3), i * 3 % 255, np.uint8)
        rec.push(frame, float(ti))
        if abs(ti - 2.0) < 1e-6:
            rec.trigger(float(ti), "V0007", "car", "red", "WEAVING", "amp=0.4", {"lat_amp": 0.41, "p_WEAVING": 0.93},
                        traj, kin)
    rec.close()
    (d,) = rec.written
    assert {p.name for p in d.iterdir()} == {"clip.mp4", "trajectory.png", "packet.json", "summary.html"}
    meta = json.loads((d / "packet.json").read_text())
    assert meta["metrics"]["lat_amp"] == 0.41 and "not" in meta["disclaimer"].lower()
    cap = cv2.VideoCapture(str(d / "clip.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 28 <= n <= 34                                # ~1 s before + ~1 s after at 15 fps
    assert "data:image/png;base64" in (d / "summary.html").read_text()


def test_close_flushes_unfinished_packet(tmp_path):
    rec = EvidenceRecorder(tmp_path)
    rec.push(np.zeros((90, 160, 3), np.uint8), 0.0)
    rec.trigger(0.0, "V1", "car", None, "OVER LIMIT", "", {}, np.zeros((0, 3)), [])
    rec.close()
    assert len(rec.written) == 1
