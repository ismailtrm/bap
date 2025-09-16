#!/usr/bin/env python3
"""
Synthetic dataset generator for balloon targets with color & shape.
- Produces simple scenes with random backgrounds, multiple balloons (circle/square/triangle),
  and exports YOLO-format labels for a single class "balloon".
- Shapes & colors are stored in an accompanying metadata JSON for downstream logic tests.
- This is for DEMO/training bootstrap; you can later mix with real frames.
"""
import os, json, argparse, random, math
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

COLORS = {
    "red": (220,50,50),
    "blue": (50,120,220),
    "green": (70,180,90),
}

SHAPES = ["circle","square","triangle"]

def rand_bg(w,h):
    # Soft gradient/noise background
    base = Image.new("RGB",(w,h),(240,240,240))
    arr = np.array(base).astype(np.int16)
    noise = np.random.normal(0, 6, size=(h,w,3))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def draw_shape(draw, shape, cx, cy, r, color):
    if shape == "circle":
        bbox = [cx-r, cy-r, cx+r, cy+r]
        draw.ellipse(bbox, fill=color, outline=(20,20,20))
        return bbox
    elif shape == "square":
        bbox = [cx-r, cy-r, cx+r, cy+r]
        draw.rectangle(bbox, fill=color, outline=(20,20,20))
        return bbox
    else: # triangle
        p1 = (cx, cy - r)
        p2 = (cx - int(0.866*r), cy + int(0.5*r))
        p3 = (cx + int(0.866*r), cy + int(0.5*r))
        draw.polygon([p1,p2,p3], fill=color, outline=(20,20,20))
        xs = [p1[0],p2[0],p3[0]]
        ys = [p1[1],p2[1],p3[1]]
        return [min(xs),min(ys),max(xs),max(ys)]

def bbox_to_yolo(bbox, W, H):
    # bbox in pixel [x1,y1,x2,y2] -> YOLO [cls cx cy w h] normalized
    x1,y1,x2,y2 = bbox
    x = (x1+x2)/2.0; y = (y1+y2)/2.0
    w = (x2-x1); h = (y2-y1)
    return x/W, y/H, w/W, h/H

def gen_image(out_images, out_labels, idx, W=640, H=360, n_shapes=(1,5)):
    im = rand_bg(W,H)
    draw = ImageDraw.Draw(im)
    n = random.randint(*n_shapes)
    ann_lines = []
    meta = {"shapes":[]}
    for _ in range(n):
        shape = random.choice(SHAPES)
        color_name = random.choice(list(COLORS.keys()))
        color = COLORS[color_name]
        r = random.randint(16, 48)  # radius/half-extent
        cx = random.randint(r+5, W-r-5)
        cy = random.randint(r+5, H-r-5)
        bbox = draw_shape(draw, shape, cx, cy, r, color)
        cxn, cyn, wn, hn = bbox_to_yolo(bbox, W, H)
        ann_lines.append(f"0 {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}")
        meta["shapes"].append({"shape":shape,"color":color_name,"bbox":bbox})
    # Light blur/gamma jitter
    if random.random()<0.3:
        im = im.filter(ImageFilter.GaussianBlur(0.8))
    if random.random()<0.4:
        im = ImageOps.autocontrast(im, cutoff=1)

    img_path = os.path.join(out_images, f"{idx:06d}.jpg")
    lbl_path = os.path.join(out_labels, f"{idx:06d}.txt")
    im.save(img_path, quality=90)
    with open(lbl_path,"w") as f: f.write("\n".join(ann_lines))
    return img_path, lbl_path, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth_samples", help="output root")
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--W", type=int, default=640)
    ap.add_argument("--H", type=int, default=360)
    args = ap.parse_args()

    train_img = os.path.join(args.out,"train/images"); os.makedirs(train_img, exist_ok=True)
    train_lbl = os.path.join(args.out,"train/labels"); os.makedirs(train_lbl, exist_ok=True)
    val_img = os.path.join(args.out,"val/images"); os.makedirs(val_img, exist_ok=True)
    val_lbl = os.path.join(args.out,"val/labels"); os.makedirs(val_lbl, exist_ok=True)

    metas = {"train":[], "val":[]}
    for i in range(args.train):
        img,lbl,meta = gen_image(train_img,train_lbl,i,args.W,args.H)
        metas["train"].append({"img":img,"label":lbl,**meta})
    for i in range(args.val):
        img,lbl,meta = gen_image(val_img,val_lbl,i,args.W,args.H)
        metas["val"].append({"img":img,"label":lbl,**meta})
    with open(os.path.join(args.out,"meta.json"),"w") as f:
        json.dump(metas,f,indent=2)
    print(f"Done. Train={args.train}, Val={args.val} → {args.out}")

if __name__ == "__main__":
    main()
