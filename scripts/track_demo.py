#!/usr/bin/env python3
"""
Tracking demo that can take detections from:
- classic CV ("cv") using color/shape pipeline
- YOLO ("yolo") using ultralytics model

It overlays track IDs and prints simple event logs.
"""
import argparse, time, csv, os
import cv2, numpy as np
from sort_tracker import SortLite
from friend_foe_classifier import classify_color_name

def detect_cv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Simple threshold union of red/blue/green
    ranges = [((0,90,80),(10,255,255)), ((170,90,80),(180,255,255)), ((95,80,60),(130,255,255)), ((40,60,60),(85,255,255))]
    mask = None
    for lo,hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask,m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),1)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets=[]
    for c in contours:
        if cv2.contourArea(c) < 150: continue
        x,y,w,h = cv2.boundingRect(c)
        dets.append([x,y,x+w,y+h])
    return dets

def yolo_predictor(weights, conf=0.25):
    try:
        from ultralytics import YOLO
        model = YOLO(weights)
    except Exception as e:
        print("Ultralytics not available. Install ultralytics or use --det cv"); model=None
    def _pred(frame):
        if model is None: return []
        res = model.predict(frame, conf=conf, verbose=False)[0]
        return res.boxes.xyxy.cpu().numpy().astype(int).tolist()
    return _pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--det", choices=["cv","yolo"], default="cv")
    ap.add_argument("--weights", default="runs/train/yolo_balloon/weights/best.pt")
    ap.add_argument("--log", default="logs/events.csv")
    args = ap.parse_args()

    if args.det=="yolo":
        pred = yolo_predictor(args.weights)
    else:
        pred = detect_cv

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logf = open(args.log,"w", newline=""); writer = csv.writer(logf)
    writer.writerow(["t","event","id","x1","y1","x2","y2"])

    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    if not cap.isOpened(): print("Cannot open source"); return

    tracker = SortLite(iou_thresh=0.3, max_age=30)
    t0=time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        dets = pred(frame)
        tracks = tracker.update(dets)

        # draw
        for (tid, box, stable) in tracks:
            x1,y1,x2,y2 = box.astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
            cv2.putText(frame,f"ID {tid}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
            writer.writerow([f"{time.time()-t0:.3f}","track",tid,x1,y1,x2,y2])

        cv2.imshow("Track Demo", frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows()
    logf.close()

if __name__ == "__main__":
    main()
