"""
SAHI-style sliced inference: split a large frame into overlapping tiles, run the
detector on each at full resolution, then merge with NMS. From altitude a car is
only ~15-30 px; a single downscaled pass shrinks it further, tiles keep the pixels.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import batched_nms


def tile_origins(size: int, tile: int, overlap: float) -> list[int]:
    if size <= tile:
        return [0]
    step = int(tile * (1 - overlap))
    xs = list(range(0, size - tile, step)) + [size - tile]
    return sorted(set(xs))


def detect_sliced(model, frame: np.ndarray, classes: list[int], conf: float,
                  tile: int = 640, overlap: float = 0.2, iou: float = 0.5,
                  device: str | None = None, add_full: bool = True):
    """Return (boxes xyxy Nx4, scores N, cls N). Optionally also a full-frame pass
    so large nearby vehicles that straddle tile edges are still found whole."""
    h, w = frame.shape[:2]
    boxes, scores, cls = [], [], []

    def run(img, ox, oy, imgsz):
        r = model.predict(img, conf=conf, imgsz=imgsz, classes=classes,
                          device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return
        b = r.boxes.xyxy.cpu().numpy().copy()
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        boxes.append(b)
        scores.append(r.boxes.conf.cpu().numpy())
        cls.append(r.boxes.cls.cpu().numpy())

    for oy in tile_origins(h, tile, overlap):
        for ox in tile_origins(w, tile, overlap):
            run(frame[oy:oy + tile, ox:ox + tile], ox, oy, tile)
    if add_full:
        run(frame, 0, 0, max(h, w))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    b, s, c = np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls)
    # class-aware NMS so a car and an overlapping truck box don't cancel each other
    keep = batched_nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(),
                       torch.from_numpy(c).long(), iou)
    keep = keep.numpy()
    return b[keep], s[keep], c[keep]
