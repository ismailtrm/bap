#!/usr/bin/env python3
"""
Streamlit UI for the demo:
- Live webcam feed
- "Engagement Accepted" button
- Draw/edit no-fire mask (simple toggle display for now)
- Live score display (A1/A2/A3)
Note: Streamlit is for demonstration; for low-latency control, a native UI is recommended.
"""
import streamlit as st
import cv2, time, numpy as np, os
from sort_tracker import SortLite
from friend_foe_classifier import classify_color_name, is_fire_allowed

st.set_page_config(page_title="Air-Defense Demo UI", layout="wide")
st.title("Air-Defense Demo UI")

col_left, col_right = st.columns([3,1])
with col_right:
    stage = st.selectbox("Stage", [1,2,3], index=0)
    accept = st.button("Engagement Accepted")
    show_mask = st.checkbox("Show No-Fire Mask", value=True)
    run = st.checkbox("Run", value=False)

with col_left:
    stframe = st.empty()

cap = cv2.VideoCapture(0)
tracker = SortLite(iou_thresh=0.3, max_age=30)

score = 0.0; base=0.0; t0=time.time()
while run:
    ok, frame = cap.read()
    if not ok: st.write("Camera not available."); break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv,(0,90,80),(10,255,255)); m2=cv2.inRange(hsv,(170,90,80),(180,255,255))
    mr = cv2.bitwise_or(m1,m2); mb=cv2.inRange(hsv,(95,80,60),(130,255,255)); mg=cv2.inRange(hsv,(40,60,60),(85,255,255))
    mask = cv2.bitwise_or(cv2.bitwise_or(mr,mb),mg)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),1)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets=[]
    for c in contours:
        if cv2.contourArea(c)<150: continue
        x,y,w,h=cv2.boundingRect(c); dets.append([x,y,x+w,y+h])

    tracks = tracker.update(dets)
    for tid, box, stable in tracks:
        x1,y1,x2,y2 = box.astype(int)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.putText(frame,f"ID {tid}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        # Demo: sample 'hit' logic when accepted and stable
        if accept and stable>=3:
            # In a real system, we'd check friend/foe, shape, and mask, then log the event.
            base += 5.0  # placeholder increment (stage-specific handled in score harness)
            accept = False

    if show_mask:
        overlay = frame.copy()
        H,W=frame.shape[:2]
        pts = np.array([[int(0.4*W),int(0.2*H)],[int(0.6*W),int(0.2*H)],[int(0.5*W),int(0.45*H)]], np.int32)
        cv2.fillPoly(overlay,[pts],(0,0,255))
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

cap.release()
