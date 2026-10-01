"""
Detection-level recall/precision on VisDrone2019-VID-val vehicles: full-frame vs
sliced inference. Every Nth frame, IoU 0.5, greedy matching by score.
Writes eval/SLICE_RESULTS.md.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from slicing import detect_sliced      # noqa: E402
from visdrone_mot import load_gt, vehicle_ids, iou_dist   # noqa: E402


def match(gt_xywh, boxes, scores):
    """Greedy IoU>=0.5 matching; returns (tp, fp, fn)."""
    if len(gt_xywh) == 0:
        return 0, len(boxes), 0
    if len(boxes) == 0:
        return 0, 0, len(gt_xywh)
    pred = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
    d = iou_dist(gt_xywh, pred)
    used, tp = set(), 0
    for j in np.argsort(-scores):
        col = d[:, j].copy()
        col[list(used)] = np.nan
        if np.all(np.isnan(col)):
            continue
        i = int(np.nanargmin(col)); used.add(i); tp += 1
    return tp, len(boxes) - tp, len(gt_xywh) - tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "Downloads/VisDrone2019-VID-val"))
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    data = Path(a.data)
    model = YOLO(a.model)
    COCO_VEHICLES = vehicle_ids(model)
    modes = {"full frame (imgsz 1280)": None, "sliced 640 tiles + full": True, "sliced 640 tiles only": False}
    tot = {m: [0, 0, 0, 0.0] for m in modes}
    frames = 0
    for seq in sorted(p for p in (data / "sequences").iterdir() if p.is_dir()):
        gt = load_gt(data / "annotations" / f"{seq.name}.txt")
        if not gt:
            continue
        for i, f in enumerate(sorted(seq.glob("*.jpg")), start=1):
            if i % a.every or i not in gt:
                continue
            img = cv2.imread(str(f)); frames += 1
            g = [b[1:] for b in gt[i]]
            for m, mode in modes.items():
                t0 = time.time()
                if mode is None:
                    r = model.predict(img, conf=a.conf, imgsz=1280, classes=COCO_VEHICLES,
                                      device=a.device, verbose=False)[0]
                    b, s = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
                else:
                    b, s, _ = detect_sliced(model, img, COCO_VEHICLES, a.conf, device=a.device, add_full=mode)
                tp, fp, fn = match(np.array(g, float), b, s)
                t = tot[m]; t[0] += tp; t[1] += fp; t[2] += fn; t[3] += time.time() - t0
        print(seq.name, {m: f"R={t[0]/max(t[0]+t[2],1):.3f}" for m, t in tot.items()}, flush=True)
    out = ["# Sliced inference — detection recall on VisDrone2019-VID-val (vehicles)", "",
           f"`{a.model}`, conf {a.conf}, every {a.every}th annotated frame ({frames} frames), IoU 0.5.", "",
           "| Mode | Recall | Precision | F1 | s/frame |", "|---|---|---|---|---|"]
    for m, (tp, fp, fn, t) in tot.items():
        r, p = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
        out.append(f"| {m} | {r:.3f} | {p:.3f} | {2*p*r/max(p+r,1e-9):.3f} | {t/max(frames,1):.2f} |")
    (ROOT / "eval" / "SLICE_RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
