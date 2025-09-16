#!/usr/bin/env python3
"""
Safety gate: veto fire command if predicted aim lies inside a no-fire polygon.
- Polygon is specified in a JSON (normalized coords 0..1).
"""
import json, numpy as np

def load_mask(json_path):
    with open(json_path) as f: m=json.load(f)
    W=m.get("image_width",1280); H=m.get("image_height",720)
    poly = np.array([[int(px*W), int(py*H)] for px,py in m["polygon"]], dtype=np.int32)
    return W,H,poly

def is_inside(point, poly):
    # point: (x,y), poly: Nx2 int
    x,y = point
    res = cv2.pointPolygonTest(poly, (float(x), float(y)), False)
    return res >= 0

try:
    import cv2
except Exception:
    # If OpenCV isn't present here, allow import but note that is_inside needs cv2.
    pass
