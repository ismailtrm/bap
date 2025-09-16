#!/usr/bin/env python3
"""
Classic CV pipeline for detecting colored balloon-like shapes and classifying their shape.
- HSV thresholds for colors (tunable).
- Morphological cleanup and contour analysis.
- Polygon approx for shape classification (circle/square/triangle).
- Area threshold for small/big.
- Designed to work on webcam or video files.
"""
import cv2, argparse, numpy as np, time

# HSV ranges (BGR→HSV) — tune for lighting. Values cover common reds, blues, greens.
HSV_RANGES = {
    "red1": ((0, 90, 80), (10, 255, 255)),     # lower red
    "red2": ((170, 90, 80), (180, 255, 255)),  # upper red
    "blue": ((95, 80, 60), (130, 255, 255)),
    "green": ((40, 60, 60), (85, 255, 255)),
}

def classify_shape(contour):
    # Approximate polygon and use vertex count + circularity to infer shape
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

def mask_color(hsv):
    masks = []
    for k,(lo,hi) in HSV_RANGES.items():
        lo=np.array(lo); hi=np.array(hi)
        masks.append(cv2.inRange(hsv, lo, hi))
    # Note: red split into two parts; 'red1' + 'red2' will both be ORed here with other colors.
    m = masks[0]
    for i in range(1,len(masks)):
        m = cv2.bitwise_or(m, masks[i])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam or path to video")
    ap.add_argument("--area_small", type=float, default=1200.0, help="area threshold for small vs big")
    ap.add_argument("--show_mask", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    if not cap.isOpened():
        print("ERROR: Cannot open source"); return

    last = time.time(); frames=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        m = mask_color(hsv)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 150: continue
            x,y,w,h = cv2.boundingRect(c)
            shape = classify_shape(c)
            size = "small" if area < args.area_small else "big"
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,f"{shape}/{size}",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)

        frames+=1
        now=time.time()
        if now-last>1.0:
            fps=frames/(now-last); frames=0; last=now
            cv2.putText(frame,f"FPS {fps:.1f}",(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)

        if args.show_mask:
            vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            out = np.hstack([frame, vis])
        else:
            out = frame
        cv2.imshow("CV Balloon Detection", out)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
