"""
Tracking evaluation on VisDrone2019-VID-val: ByteTrack vs BoT-SORT vs BoT-SORT+ReID.

Ground truth rows: frame,id,x,y,w,h,score,category,truncation,occlusion.
We keep vehicle categories only (car, van, truck, bus) and predict with the
COCO detector's car/bus/truck classes. Matching is IoU >= 0.5 (motmetrics).

Usage:
  python eval/visdrone_mot.py                       # all trackers, all sequences
  python eval/visdrone_mot.py --trackers bytetrack --max-seqs 1 --max-frames 100
Writes eval/RESULTS.md. Note: HOTA is not computed here (motmetrics has no HOTA).
"""
import argparse
import time
from pathlib import Path

import motmetrics as mm
from hota import hota
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
GT_VEHICLE_CATS = {4, 5, 6, 9}          # car, van, truck, bus
EVAL_NAMES = {"car", "van", "truck", "bus"}        # same set as the GT categories we keep


def vehicle_ids(model) -> list[int]:
    """Class ids by name, so COCO and VisDrone-fine-tuned weights are scored on the same classes."""
    return [i for i, n in model.names.items() if n in EVAL_NAMES]
TRACKERS = {
    "ByteTrack": "bytetrack.yaml",
    "BoT-SORT": "botsort.yaml",
    "BoT-SORT+ReID": str(ROOT / "botsort_reid.yaml"),
}


def load_gt(path: Path) -> dict[int, list]:
    gt: dict[int, list] = {}
    for line in path.read_text().split():
        r = [float(x) for x in line.split(",")]
        frame, tid, x, y, w, h, score, cat = r[:8]
        if score == 0 or int(cat) not in GT_VEHICLE_CATS:
            continue
        gt.setdefault(int(frame), []).append((int(tid), x, y, w, h))
    return gt


def iou_dist(a, b, max_iou=0.5):
    """1-IoU cost matrix for (x,y,w,h) boxes; pairs below max_iou become NaN (no match).
    Own implementation: motmetrics' version breaks on NumPy 2."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, :1], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1:2], b[:, 1]), 0, None)
    inter = iw * ih
    iou = inter / (a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter + 1e-9)
    d = 1 - iou
    d[iou < max_iou] = np.nan
    return d


def run_sequence(model, seq: Path, ann: Path, tracker: str, args):
    gt = load_gt(ann)
    frames = sorted(seq.glob("*.jpg"))[: args.max_frames or None]
    acc = mm.MOTAccumulator(auto_id=False)
    gt_h, pr_h = {}, {}                          # frame -> (ids, xywh boxes) for HOTA
    model.predictor = None                      # reset tracker state between sequences
    for i, f in enumerate(frames, start=1):
        res = model.track(str(f), persist=True, conf=args.conf, imgsz=args.imgsz,
                          classes=vehicle_ids(model), tracker=tracker,
                          device=args.device, verbose=False)[0]
        ids, boxes = [], []
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int).tolist()
            for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        g = gt.get(i, [])
        dist = iou_dist([b[1:] for b in g], boxes) if g and boxes \
            else np.empty((len(g), len(boxes)))
        acc.update([b[0] for b in g], ids, dist, frameid=i)
        gt_h[i] = (np.array([b[0] for b in g], int), np.array([b[1:] for b in g], float).reshape(-1, 4))
        pr_h[i] = (np.array(ids, int), np.array(boxes, float).reshape(-1, 4))
    return acc, hota(gt_h, pr_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "Downloads/VisDrone2019-VID-val"))
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--trackers", nargs="+", default=list(TRACKERS), choices=list(TRACKERS))
    ap.add_argument("--imgsz", type=int, default=1280, help="aerial cars are tiny; 640 misses many")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    seqs = sorted(p for p in (data / "sequences").iterdir() if p.is_dir())[: args.max_seqs or None]
    # Some sequences are pedestrian-only; they say nothing about vehicle tracking.
    seqs = [s for s in seqs if load_gt(data / "annotations" / f"{s.name}.txt")]
    rows = []
    for name in args.trackers:
        model = YOLO(args.model)                # fresh model per tracker
        accs, hotas, t0, n = [], [], time.time(), 0
        for s in seqs:
            print(f"[{name}] {s.name}", flush=True)
            acc, h = run_sequence(model, s, data / "annotations" / f"{s.name}.txt", TRACKERS[name], args)
            accs.append(acc); hotas.append(h)
            n += len(accs[-1].mot_events.index.get_level_values(0).unique())
        summ = mm.metrics.create().compute_many(
            accs, metrics=["mota", "idf1", "num_switches", "recall", "precision",
                           "num_misses", "num_false_positives", "num_objects"],
            names=[s.name for s in seqs], generate_overall=True).loc["OVERALL"]
        fps = n / max(time.time() - t0, 1e-6)
        hm = {k: float(np.nanmean([h[k] for h in hotas])) for k in ('HOTA', 'DetA', 'AssA')}
        rows.append((name, summ, fps, hm))
        print(f"[{name}] MOTA {summ.mota:.3f} IDF1 {summ.idf1:.3f} HOTA {hm['HOTA']:.3f} IDsw {int(summ.num_switches)}", flush=True)

    out = ["# Tracking results — VisDrone2019-VID-val (vehicles only)", "",
           f"Detector `{args.model}`, imgsz {args.imgsz}, conf {args.conf}, "
           f"{len(seqs)} sequences{f', first {args.max_frames} frames each' if args.max_frames else ''}. "
           "MOTA/IDF1 at IoU 0.5 (motmetrics); HOTA is the mean over IoU 0.05-0.95 (own implementation of the TrackEval algorithm, averaged over sequences).", "",
           "| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | ID switches | Recall | Precision | FPS |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s, fps, hm in rows:
        out.append(f"| {name} | {hm['HOTA']:.3f} | {hm['DetA']:.3f} | {hm['AssA']:.3f} | {s.mota:.3f} | {s.idf1:.3f} | {int(s.num_switches)} | "
                   f"{s.recall:.3f} | {s.precision:.3f} | {fps:.1f} |")
    (ROOT / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
