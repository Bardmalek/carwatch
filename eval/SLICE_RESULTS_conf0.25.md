# Sliced inference — detection recall on VisDrone2019-VID-val (vehicles)

`yolo26s.pt`, conf 0.25, every 15th annotated frame (157 frames), IoU 0.5.

| Mode | Recall | Precision | F1 | s/frame |
|---|---|---|---|---|
| full frame (imgsz 1280) | 0.615 | 0.682 | 0.646 | 0.03 |
| sliced 640 tiles + full | 0.724 | 0.469 | 0.569 | 0.44 |
| sliced 640 tiles only | 0.678 | 0.512 | 0.583 | 0.29 |
