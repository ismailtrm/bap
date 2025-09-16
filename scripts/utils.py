#!/usr/bin/env python3
"""
Utility helpers for color ranges, drawing, and bbox conversion.
"""
import numpy as np

def xywh_to_xyxy(x,y,w,h):
    return [x, y, x+w, y+h]

def xyxy_to_cxcywh(x1,y1,x2,y2):
    w = x2-x1; h = y2-y1; cx = x1 + w/2.0; cy = y1 + h/2.0
    return [cx,cy,w,h]
