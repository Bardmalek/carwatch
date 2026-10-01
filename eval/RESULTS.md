# Tracking results — VisDrone2019-VID-val (vehicles only)

Detector `yolo26s.pt`, imgsz 1280, conf 0.25, 6 sequences. MOTA/IDF1 at IoU 0.5 (motmetrics); HOTA is the mean over IoU 0.05-0.95 (own implementation of the TrackEval algorithm, averaged over sequences).

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | ID switches | Recall | Precision | FPS |
|---|---|---|---|---|---|---|---|---|---|
| ByteTrack | 0.482 | 0.422 | 0.573 | 0.388 | 0.557 | 329 | 0.572 | 0.765 | 18.1 |
| BoT-SORT | 0.543 | 0.444 | 0.679 | 0.405 | 0.643 | 104 | 0.609 | 0.751 | 11.7 |
| BoT-SORT+ReID | 0.540 | 0.440 | 0.678 | 0.390 | 0.643 | 144 | 0.614 | 0.736 | 7.0 |
