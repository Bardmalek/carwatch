# Sliced inference — detection recall on VisDrone2019-VID-val (vehicles)

`yolo26s.pt`, conf 0.4, every 15th annotated frame (157 frames), IoU 0.5.

| Mode | Recall | Precision | F1 | s/frame |
|---|---|---|---|---|
| full frame (imgsz 1280) | 0.557 | 0.806 | 0.659 | 0.03 |
| sliced 640 tiles + full | 0.681 | 0.572 | 0.622 | 0.43 |
| sliced 640 tiles only | 0.620 | 0.601 | 0.610 | 0.29 |
