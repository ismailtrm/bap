#!/usr/bin/env python3
"""
Run YOLO inference on webcam/video and draw detections.
"""
import argparse, time
import cv2, numpy as np
try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not available. Install with: pip install ultralytics")
    raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/train/yolo_balloon/weights/best.pt")
    ap.add_argument("--source", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    if not cap.isOpened(): print("ERROR opening source"); return

    last=time.time(); frames=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.predict(frame, conf=args.conf, verbose=False)[0]
        for b in res.boxes.xyxy.cpu().numpy().astype(int):
            x1,y1,x2,y2 = b[:4]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,"balloon",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)

        frames+=1; now=time.time()
        if now-last>1.0:
            fps=frames/(now-last); frames=0; last=now
            cv2.putText(frame,f"FPS {fps:.1f}",(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)

        cv2.imshow("YOLO Inference", frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
