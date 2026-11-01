"""Confidence-weighted road severity index and damage coverage.

Why this exists: the old score averaged (class weight x detector confidence) over
all detections. That mostly measures how confident the model is, and every weak
box dilutes it. This version scores HOW MUCH of the frame each damage type covers,
and lets weak boxes count in proportion to their confidence instead of being
either fully trusted or thrown away.

Use in appv8.py:
    import severity_utils
    sev_score    = severity_utils.severity_score(df, img_w, img_h)
    coverage_pct = severity_utils.coverage_pct(df, img_w, img_h)
    # optional: expected repair count/cost
    cost = sum(REPAIR_COST[r.Class] * severity_utils.evidence_weight(r.Confidence)
               for r in df.itertuples())

All constants are design assumptions, not standards. Calibrate them: ideally measure
precision per confidence band on your validation set (then use that as the weight),
and check that the labels match your own judgement on 10-15 photos.
"""
import numpy as np

# Structural-impact rank (same as the app): D00 longitudinal ... D40 pothole
SEVERITY_WEIGHT = {"D00": 1, "D10": 2, "D20": 3, "D40": 4}

MIN_CONF = 0.05      # below this a box is treated as noise and ignored
CONF_FULL = 0.50     # a box at/above this confidence counts fully; weaker ones count proportionally
DENSITY_REF = 0.30   # confidence-weighted share of the frame one damage type must cover to be "fully severe"
POTHOLE_POINTS = 10  # extra points per (fully trusted) pothole: a hazard even when small
POTHOLE_CAP = 40


def evidence_weight(conf):
    """0..1 weight for a detection: 0 below MIN_CONF, linear up to CONF_FULL, then 1."""
    conf = float(conf)
    return 0.0 if conf < MIN_CONF else min(1.0, conf / CONF_FULL)


def _weight_map(rows, img_w, img_h):
    """Float image: every box paints its evidence weight; overlaps keep the maximum."""
    m = np.zeros((img_h, img_w), dtype=np.float32)
    for r in rows.itertuples():
        w = evidence_weight(r.Confidence)
        x1, y1 = max(int(r.x1), 0), max(int(r.y1), 0)
        x2, y2 = min(int(r.x2), img_w), min(int(r.y2), img_h)
        if w > 0 and x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = np.maximum(m[y1:y2, x1:x2], w)
    return m


def coverage_pct(df, img_w, img_h):
    """Confidence-weighted percent of the frame covered by damage (overlaps counted once)."""
    if df.empty or img_w <= 0 or img_h <= 0:
        return 0.0
    return round(float(_weight_map(df, img_w, img_h).mean()) * 100.0, 2)


def severity_score(df, img_w, img_h):
    """0-100 index: weighted damage extent per type x structural impact, plus a pothole term."""
    if df.empty or img_w <= 0 or img_h <= 0:
        return 0.0
    extent = 0.0
    for cls, g in df.groupby("Class"):
        density = float(_weight_map(g, img_w, img_h).mean())
        extent += (SEVERITY_WEIGHT[cls] / 4) * min(1.0, density / DENSITY_REF)
    score = 100.0 * min(1.0, extent)
    pothole_w = sum(evidence_weight(c) for c in df.loc[df["Class"] == "D40", "Confidence"])
    score += min(POTHOLE_CAP, POTHOLE_POINTS * pothole_w)
    return round(min(100.0, score), 1)
