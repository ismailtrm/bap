#!/usr/bin/env python3
"""
Friend/Foe color classifier.
- Maps HSV histogram/mean color to a discrete label.
- Conservative gate: "unknown" → do not fire.
"""
import numpy as np, cv2

# Define which color is friend vs foe for Stage-2 (you can swap as needed)
FRIEND_COLOR = "green"
FOE_COLOR = "red"

def classify_color_name(bgr_roi):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mean = hsv.mean(axis=(0,1))  # H,S,V means
    H = mean[0]

    # Simple H-based decision; you can refine with S/V thresholds
    if (H < 15) or (H > 170):  # red-ish (wraps around)
        return "red"
    if 40 <= H <= 85:
        return "green"
    if 95 <= H <= 130:
        return "blue"
    return "unknown"

def is_fire_allowed(color_label):
    # Fire only if explicitly considered FOE
    return color_label == FOE_COLOR

