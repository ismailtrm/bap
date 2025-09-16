#!/usr/bin/env python3
"""
Double-check logic for Stage-3: shape + color must both match.
- Uses simple contour-based shape classification + mean color.
"""
import cv2, numpy as np

def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    area = cv2.contourArea(contour)
    if peri == 0: return "unknown"
    circularity = 4*np.pi*(area/(peri*peri))
    if len(approx) >= 8 or circularity > 0.7:
        return "circle"
    elif len(approx) == 3:
        return "triangle"
    elif 4 <= len(approx) <= 6:
        return "square"
    return "unknown"

def mean_color_label(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[...,0].mean()
    if (H < 15) or (H > 170): return "red"
    if 40 <= H <= 85: return "green"
    if 95 <= H <= 130: return "blue"
    return "unknown"

def shape_color_match(contour, roi_bgr, target_shape, target_color):
    s = classify_shape(contour)
    c = mean_color_label(roi_bgr)
    ok = (s == target_shape) and (c == target_color)
    return ok, s, c
