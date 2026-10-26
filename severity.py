"""
Rule-based severity + pass/fail logic.

Severity depends on:
  1. How much of the image the defect covers (area %)
  2. How dangerous the defect type is (critical classes get bumped up one level)
You can tune the numbers below and justify them in your report.
"""

# Defect types that are structurally more serious
CRITICAL_CLASSES = {"crazing", "inclusion", "pitted_surface"}

# Area thresholds in % of image area
LOW_MAX = 2.0
MEDIUM_MAX = 8.0

LEVELS = ["Low", "Medium", "High"]

# STRICT_MODE = True  -> ANY detected defect rejects the product (typical for zero-defect parts)
# STRICT_MODE = False -> graded verdict: PASS / REVIEW / REJECT based on severity and count
STRICT_MODE = True


def bbox_area_pct(x1, y1, x2, y2, img_w, img_h):
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return 100.0 * area / (img_w * img_h)


def classify_severity(class_name: str, area_pct: float) -> str:
    if area_pct <= LOW_MAX:
        level = 0
    elif area_pct <= MEDIUM_MAX:
        level = 1
    else:
        level = 2
    if class_name in CRITICAL_CLASSES:
        level = min(level + 1, 2)
    return LEVELS[level]


def overall_verdict(severities: list, strict=None) -> str:
    """
    strict mode : no defects -> PASS, any defect -> REJECT.
    graded mode : no defects -> PASS; any High defect or 3+ defects -> REJECT; otherwise REVIEW.
    """
    if strict is None:
        strict = STRICT_MODE
    if not severities:
        return "PASS"
    if strict:
        return "REJECT"
    if "High" in severities or len(severities) >= 3:
        return "REJECT"
    return "REVIEW"
