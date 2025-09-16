#!/usr/bin/env python3
"""
Lightweight SORT-like tracker.
- Maintains a set of tracks with simple constant-velocity model.
- IOU-based association (greedy), minimal deps.
- This is NOT a full Kalman/Hungarian SORT, but keeps similar interfaces.
"""
import numpy as np, time

def iou(a, b):
    # a,b: [x1,y1,x2,y2]
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    w=max(0,x2-x1); h=max(0,y2-y1); inter=w*h
    if inter<=0: return 0.0
    area_a=(a[2]-a[0])*(a[3]-a[1]); area_b=(b[2]-b[0])*(b[3]-b[1])
    return inter/(area_a+area_b-inter+1e-6)

class Track:
    def __init__(self, box, id, ttl=30):
        self.id=id
        self.box=np.array(box, dtype=float)
        self.v=np.zeros(4, dtype=float)  # simple velocity on corners
        self.ttl=ttl
        self.age=0
        self.stable_hits=0

    def predict(self):
        self.box = self.box + self.v
        self.age += 1
        self.ttl -= 1

    def update(self, box):
        new_box=np.array(box, dtype=float)
        self.v = 0.5*self.v + 0.5*(new_box - self.box)  # EMA for velocity
        self.box = new_box
        self.ttl = 30
        self.stable_hits += 1

class SortLite:
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh=iou_thresh
        self.max_age=max_age
        self.tracks=[]
        self._next_id=1

    def update(self, detections):
        # detections: list of [x1,y1,x2,y2]
        for t in self.tracks: t.predict()

        assigned=set()
        # Greedy match highest IOU for each detection
        for di, det in enumerate(detections):
            best_iou=0; best_t=None
            for t in self.tracks:
                i=iou(det, t.box)
                if i>best_iou:
                    best_iou=i; best_t=t
            if best_t is not None and best_iou >= self.iou_thresh:
                best_t.update(det); assigned.add(best_t.id)
            else:
                # new track
                tr=Track(det, self._next_id, ttl=self.max_age)
                self._next_id+=1
                self.tracks.append(tr)

        # remove dead tracks
        alive=[]
        for t in self.tracks:
            if t.ttl>0: alive.append(t)
        self.tracks=alive

        return [(int(t.id), t.box.copy(), t.stable_hits) for t in self.tracks]
