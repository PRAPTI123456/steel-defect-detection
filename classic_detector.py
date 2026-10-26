"""
Classical (no-training) surface defect detector for thin scratch-like defects.

Why this exists: the YOLO model is trained on NEU-DET (flat grayscale steel close-ups).
On a normal colour photo of a steel part it may find nothing. This detector uses plain
image processing, so it works on ANY steel photo without training:

  1. Convert to grayscale and remove slow lighting changes (background subtraction)
  2. Find thin line-like structures with a multi-scale Hessian "ridge" filter
  3. Suppress strong object edges (the outline of the part is not a defect)
  4. Keep only elongated, connected, high-contrast components -> scratches / cracks

Tune with the parameters of detect_scratches(). Output format matches the YOLO
detections used elsewhere (class_name, confidence, box, area_pct, severity added later).
"""
import cv2
import numpy as np


def _ridge_response(gray_f, sigmas=(1.5, 2.5)):
    """Max line-like response over scales. Positive for bright OR dark thin lines."""
    best = np.zeros_like(gray_f)
    for s in sigmas:
        g = cv2.GaussianBlur(gray_f, (0, 0), s)
        gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        tmp = np.sqrt(((gxx - gyy) / 2) ** 2 + gxy ** 2)
        l1 = (gxx + gyy) / 2 + tmp   # larger eigenvalue
        l2 = (gxx + gyy) / 2 - tmp   # smaller eigenvalue
        # bright line: strongly negative l2, small |l1|; dark line: strongly positive l1, small |l2|
        bright = np.maximum(-l2, 0) - np.abs(l1) * 0.5
        dark = np.maximum(l1, 0) - np.abs(l2) * 0.5
        resp = np.maximum(bright, dark) * (s ** 2)   # scale normalisation
        best = np.maximum(best, resp)
    return np.maximum(best, 0)


def detect_scratches(
    image_bgr,
    work_size=640,
    z_thresh=9.0,
    z_low=3.0,
    close_px=21,
    side_max=14.0,
    min_len_frac=0.06,
    max_len_frac=0.60,
    min_elongation=4.0,
    edge_suppress=True,
    outline_frac=0.35,
    max_components=8,
    debug=False,
):
    """
    Returns list of dicts: {class_name, confidence, box=(x1,y1,x2,y2) in ORIGINAL pixels,
                            polygon (list of points, original pixels), source}
    min_len_frac / max_len_frac: component length as a fraction of the image's longer side.
    z_thresh: higher = fewer, stronger detections (less false alarms). 10 = top-1% ridge strength.
    """
    h0, w0 = image_bgr.shape[:2]
    scale = work_size / max(h0, w0)
    img = cv2.resize(image_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr.copy()
    scale = img.shape[1] / w0
    h, w = img.shape[:2]
    long_side = max(h, w)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray_f = gray.astype(np.float32)
    # remove slow illumination changes
    gray_f = gray_f - cv2.GaussianBlur(gray_f, (0, 0), 20)

    ridge = _ridge_response(gray_f)

    # robust z-score threshold
    # ridge map is mostly zeros, so scale by a high percentile instead of MAD
    scale_ref = float(np.percentile(ridge, 99)) + 1e-6
    z = ridge / scale_ref * 10.0          # z = 10 means "as strong as the top 1% of pixels"
    # suppress the OUTLINE of the part: only long, connected strong edges are removed,
    # so a short scratch (which is also a strong feature) is kept.
    if edge_suppress:
        sm = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        step = (mag > np.percentile(mag, 94)).astype(np.uint8) * 255
        ns, sl, sst, _ = cv2.connectedComponentsWithStats(step, connectivity=8)
        outline = np.zeros_like(step)
        for j in range(1, ns):
            ext = max(sst[j, cv2.CC_STAT_WIDTH], sst[j, cv2.CC_STAT_HEIGHT])
            thin = sst[j, cv2.CC_STAT_AREA] <= ext * 14   # thin line, not a fat noise blob
            if ext > outline_frac * long_side and thin:
                outline[sl == j] = 255
        outline = cv2.dilate(outline, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        z[outline > 0] = 0

    low = (z > z_low).astype(np.uint8)
    high = (z > z_thresh)
    # hysteresis: keep weak ridge pixels only if their connected group has a strong seed
    nl, ll = cv2.connectedComponents(low, connectivity=8)
    keep = np.zeros(nl, bool)
    keep[np.unique(ll[high])] = True
    keep[0] = False
    mask = keep[ll].astype(np.uint8) * 255

    # join small gaps along a scratch
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 15:
            continue
        ys, xs = np.where(labels == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        length, thick = max(rw, rh), max(min(rw, rh), 1.0)
        if length < min_len_frac * long_side or length > max_len_frac * long_side:
            continue
        if length / thick < min_elongation:
            continue
        # side test: compare brightness on the two sides of the component
        ang_r = np.deg2rad(ang if rw >= rh else ang + 90)
        d = np.array([np.cos(ang_r), np.sin(ang_r)])
        nrm = np.array([-d[1], d[0]])
        sg = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)
        sample = pts[:: max(1, len(pts) // 200)]
        off = 10
        def _mean_at(o):
            q = np.clip((sample + nrm * o).round().astype(int), [0, 0], [w - 1, h - 1])
            return float(sg[q[:, 1], q[:, 0]].mean())
        side_diff = abs(_mean_at(off) - _mean_at(-off))
        if side_diff > side_max:
            continue
        strength = float(z[ys, xs].mean())
        peak = float(z[ys, xs].max())
        conf = float(np.clip(0.45 + 0.02 * (peak - z_thresh), 0.35, 0.90))
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        pad = 6
        box = (
            max(0, (x1 - pad)) / scale, max(0, (y1 - pad)) / scale,
            min(w - 1, (x2 + pad)) / scale, min(h - 1, (y2 + pad)) / scale,
        )
        found.append({"class_name": "scratches", "confidence": round(conf, 3),
                      "box": tuple(round(float(v), 1) for v in box), "source": "classic", "_len": length,
                      "_strength": round(strength, 2), "_side": round(side_diff, 1), "_peak": round(float(z[ys, xs].max()), 1)})
    found.sort(key=lambda d: (d["confidence"], d["_len"]), reverse=True)
    if not debug:
        for d in found:
            for k in ("_len", "_strength", "_side", "_peak"):
                d.pop(k, None)
    return found[:max_components]
