#!/usr/bin/env python3
"""
Target board detector placeholder:
- For demo purposes, we search for a red triangle region (template-like)
  and assume "A/B/C" platform cue is given externally via UI selection or OCR.
- In real runs, use AprilTag/ArUco or a printed template and a strict ROI.
"""
import cv2, numpy as np

def find_board_roi(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0,80,80), (10,255,255))
    mask2 = cv2.inRange(hsv, (170,80,80), (180,255,255))
    mask = cv2.bitwise_or(mask, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 500: return None
    x,y,w,h = cv2.boundingRect(c)
    return (x,y,w,h)
