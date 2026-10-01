"""
Real vehicle trajectories from VisDrone ground truth, in the stabilizer's world frame.
Runs carwatch.Stabilizer over the frames (GT vehicle boxes masked) so drone drift is
removed, then caches per-vehicle tracks to eval/cache/real_tracks.pkl.
"""
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import carwatch                    # noqa: E402

CACHE = ROOT / "eval" / "cache" / "real_tracks.pkl"
FPS = 30.0      # VisDrone-VID is ~30 fps; assumed, not read from the files


def extract(data: Path):
    out = {}
    for seq in sorted(p for p in (data / "sequences").iterdir() if p.is_dir()):
        gt_raw = {}
        for line in (data / "annotations" / f"{seq.name}.txt").read_text().split():
            r = [float(x) for x in line.split(",")]
            if r[6] != 0 and int(r[7]) in (4, 5, 6, 9):
                gt_raw.setdefault(int(r[0]), []).append((int(r[1]), r[2], r[3], r[4], r[5]))
        if not gt_raw:
            continue
        stab, tracks = carwatch.Stabilizer(), {}
        for i, f in enumerate(sorted(seq.glob("*.jpg")), start=1):
            boxes = gt_raw.get(i, [])
            stab.update(cv2.imread(str(f)), [(x, y, x + w, y + h) for _, x, y, w, h in boxes])
            for tid, x, y, w, h in boxes:
                wx, wy = stab.to_world(x + w / 2, y + h / 2)
                tracks.setdefault(tid, []).append((i, wx, wy, max(w, h)))
        out[seq.name] = {k: np.array(v) for k, v in tracks.items()}
        print(seq.name, len(out[seq.name]), "tracks", flush=True)
    return out


def load(data=Path.home() / "Downloads/VisDrone2019-VID-val"):
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    t = extract(Path(data))
    CACHE.write_bytes(pickle.dumps(t))
    return t


if __name__ == "__main__":
    t = load()
    print({k: len(v) for k, v in t.items()})
