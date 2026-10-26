"""
Core pipeline:  Image -> detectors -> defects -> severity -> verdict -> database

Two detectors work together ("hybrid"):
  1. YOLO model trained on NEU-DET  (learned, 6 defect types)
  2. Classical scratch detector      (OpenCV, no training, works on any steel photo)

CLI usage (inspect a whole folder):
    python inspector.py --weights runs/neu_det/weights/best.pt --folder some_images/
    python inspector.py --folder some_images/ --no-yolo        # classical detector only
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

import db
from classic_detector import detect_scratches
from severity import bbox_area_pct, classify_severity, overall_verdict

_model_cache = {}

COLORS = {"yolo": (0, 200, 0), "classic": (0, 0, 255)}  # BGR: green = YOLO, red = classical


def load_model(weights):
    """Returns a YOLO model, or None if the weights file does not exist."""
    if not weights or not os.path.exists(weights):
        return None
    from ultralytics import YOLO

    if weights not in _model_cache:
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _yolo_detect(model, image_bgr, conf, as_gray):
    img = image_bgr
    if as_gray:  # NEU-DET is grayscale; this narrows the gap for colour photos
        g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    result = model.predict(img, conf=conf, verbose=False)[0]
    out = []
    for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        out.append({
            "class_name": result.names[int(cls)],
            "confidence": round(float(score), 3),
            "box": tuple(round(v, 1) for v in box),
            "source": "yolo",
        })
    return out


def inspect_image(model, image_bgr, conf=0.25, use_classic=True, classic_z=9.0, yolo_gray=True):
    """
    model may be None (classical detector only).
    Returns (annotated_bgr, defects, verdict).
    """
    h, w = image_bgr.shape[:2]
    detections = []
    if model is not None:
        detections += _yolo_detect(model, image_bgr, conf, yolo_gray)
    if use_classic:
        for c in detect_scratches(image_bgr, z_thresh=classic_z):
            # skip classical boxes that YOLO already found
            if not any(_iou(c["box"], d["box"]) > 0.3 for d in detections if d["source"] == "yolo"):
                detections.append(c)

    defects = []
    annotated = image_bgr.copy()
    thick = max(2, int(round(max(h, w) / 300)))
    scale = max(0.5, max(h, w) / 900)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        area = bbox_area_pct(x1, y1, x2, y2, w, h)
        d["area_pct"] = round(area, 2)
        d["severity"] = classify_severity(d["class_name"], area)
        defects.append(d)
        col = COLORS.get(d["source"], (255, 0, 0))
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), col, thick)
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        cv2.putText(annotated, label, (int(x1), max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, col, max(1, thick - 1), cv2.LINE_AA)

    verdict = overall_verdict([d["severity"] for d in defects])
    return annotated, defects, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/neu_det/weights/best.pt")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-yolo", action="store_true", help="use only the classical detector")
    ap.add_argument("--no-classic", action="store_true", help="use only YOLO")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    db.init_db()
    model = None if args.no_yolo else load_model(args.weights)
    if model is None and not args.no_yolo:
        print(f"[info] weights not found at '{args.weights}' -> using classical detector only")
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    for p in sorted(Path(args.folder).iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        annotated, defects, verdict = inspect_image(model, img, args.conf, use_classic=not args.no_classic)
        db.save_inspection(p.name, verdict, defects)
        cv2.imwrite(str(out_dir / p.name), annotated)
        print(f"{p.name:30s} {verdict:7s} defects={len(defects)}")


if __name__ == "__main__":
    main()
