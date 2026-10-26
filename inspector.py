"""YOLO-only inspector. Every image is converted to grayscale before inference."""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import severity
except ImportError:
    severity = None


def load_model(weights=None):
    """Return a YOLO model, or None if the weights file isn't found."""
    candidates = [weights, "runs/neu_det/weights/best.pt", "best.pt"]
    for c in candidates:
        if c and Path(c).exists():
            return YOLO(str(c))
    return None


def to_grayscale_bgr(img_bgr):
    """BGR image -> grayscale -> 3 channels (YOLO needs 3 channels, values stay gray)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _severity_from_area(area_pct):
    if area_pct >= 5:
        return "HIGH"
    if area_pct >= 1:
        return "MEDIUM"
    return "LOW"


def _verdict(defects):
    if not defects:
        return "PASS"
    strict = getattr(severity, "STRICT_MODE", True) if severity else True
    if strict:
        return "REJECT"
    levels = {d["severity"] for d in defects}
    return "REJECT" if "HIGH" in levels else "REVIEW"


def inspect_image(model, img_bgr, conf=0.25, imgsz=640, grayscale=True, **_ignored):
    """
    Returns (annotated_bgr, defects, verdict)
    defects: list of dicts with class_name, source, confidence, area_pct, severity, bbox
    """
    source = to_grayscale_bgr(img_bgr) if grayscale else img_bgr
    h, w = source.shape[:2]

    defects = []
    annotated = source.copy()
    if model is not None:
        result = model.predict(source, conf=conf, imgsz=imgsz, verbose=False)[0]
        annotated = result.plot()  # BGR
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            area_pct = round(100.0 * (x2 - x1) * (y2 - y1) / (w * h), 2)
            defects.append({
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "source": "YOLO",
                "confidence": round(float(box.conf[0]), 3),
                "area_pct": area_pct,
                "severity": _severity_from_area(area_pct),
                "box": (x1, y1, x2, y2),
                "bbox": [x1, y1, x2, y2],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    return annotated, defects, _verdict(defects)
