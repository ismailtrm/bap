#!/usr/bin/env python3
"""
Enhanced synthetic dataset generator for balloon targets with improved realism.
- More realistic balloon shapes with gradients and shadows
- Better backgrounds (sky, clouds, outdoor scenes)
- Occlusion handling and motion blur
- Color variations and lighting effects
- Size variations based on distance simulation
"""
import os, json, argparse, random, math
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import numpy as np

# Enhanced color palette with variations
COLORS = {
    "red": [(220,50,50), (200,30,30), (240,70,70), (180,20,20)],
    "blue": [(50,120,220), (30,100,200), (70,140,240), (20,80,180)],
    "green": [(70,180,90), (50,160,70), (90,200,110), (30,140,50)],
}

SHAPES = ["circle", "square", "triangle"]

def create_sky_background(w, h):
    """Create a more realistic sky background with gradient and clouds"""
    # Create sky gradient
    sky = Image.new("RGB", (w, h), (135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(sky)
    
    # Add gradient effect
    for y in range(h):
        ratio = y / h
        # Fade from light blue at top to darker at bottom
        r = int(135 + ratio * 20)
        g = int(206 + ratio * 10)
        b = int(235 + ratio * 15)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    
    # Add some cloud-like noise
    arr = np.array(sky).astype(np.float32)
    noise = np.random.normal(0, 8, size=(h, w, 3))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)

def draw_realistic_balloon(draw, shape, cx, cy, r, color_variants):
    """Draw a more realistic balloon with gradients and shadows"""
    color = random.choice(color_variants)
    
    if shape == "circle":
        # Main balloon
        bbox = [cx-r, cy-r, cx+r, cy+r]
        draw.ellipse(bbox, fill=color, outline=(20,20,20), width=2)
        
        # Add highlight
        highlight_bbox = [cx-r//2, cy-r//2, cx+r//3, cy+r//3]
        draw.ellipse(highlight_bbox, fill=(255,255,255,100), outline=None)
        
        # Add shadow
        shadow_bbox = [cx-r+2, cy-r+2, cx+r+2, cy+r+2]
        draw.ellipse(shadow_bbox, fill=(0,0,0,50), outline=None)
        
        return bbox
        
    elif shape == "square":
        # Main balloon
        bbox = [cx-r, cy-r, cx+r, cy+r]
        draw.rectangle(bbox, fill=color, outline=(20,20,20), width=2)
        
        # Add highlight
        highlight_bbox = [cx-r//2, cy-r//2, cx+r//3, cy+r//3]
        draw.rectangle(highlight_bbox, fill=(255,255,255,100), outline=None)
        
        return bbox
        
    else:  # triangle
        p1 = (cx, cy - r)
        p2 = (cx - int(0.866*r), cy + int(0.5*r))
        p3 = (cx + int(0.866*r), cy + int(0.5*r))
        draw.polygon([p1,p2,p3], fill=color, outline=(20,20,20), width=2)
        
        # Add highlight
        h1 = (cx, cy - r//2)
        h2 = (cx - int(0.4*r), cy)
        h3 = (cx + int(0.4*r), cy)
        draw.polygon([h1,h2,h3], fill=(255,255,255,100), outline=None)
        
        xs = [p1[0],p2[0],p3[0]]
        ys = [p1[1],p2[1],p3[1]]
        return [min(xs),min(ys),max(xs),max(ys)]

def apply_motion_blur(image, intensity=0.3):
    """Apply motion blur to simulate movement"""
    if random.random() < intensity:
        # Random direction blur
        angle = random.uniform(0, 360)
        distance = random.uniform(1, 3)
        return image.filter(ImageFilter.GaussianBlur(radius=distance))
    return image

def apply_lighting_effects(image):
    """Apply various lighting effects"""
    # Random brightness adjustment
    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Random contrast adjustment
    if random.random() < 0.3:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.9, 1.1))
    
    return image

def bbox_to_yolo(bbox, W, H):
    """Convert bbox to YOLO format"""
    x1,y1,x2,y2 = bbox
    x = (x1+x2)/2.0; y = (y1+y2)/2.0
    w = (x2-x1); h = (y2-y1)
    return x/W, y/H, w/W, h/H

def gen_enhanced_image(out_images, out_labels, idx, W=640, H=360, n_shapes=(1,5)):
    """Generate enhanced synthetic image with better realism"""
    # Create realistic background
    im = create_sky_background(W, H)
    draw = ImageDraw.Draw(im)
    
    n = random.randint(*n_shapes)
    ann_lines = []
    meta = {"shapes":[]}
    
    # Generate objects with potential occlusion
    objects = []
    for _ in range(n):
        shape = random.choice(SHAPES)
        color_name = random.choice(list(COLORS.keys()))
        color_variants = COLORS[color_name]
        
        # Size variation based on "distance" simulation
        base_r = random.randint(16, 48)
        distance_factor = random.uniform(0.7, 1.3)
        r = int(base_r * distance_factor)
        
        # Position with overlap handling
        max_attempts = 10
        for attempt in range(max_attempts):
            cx = random.randint(r+5, W-r-5)
            cy = random.randint(r+5, H-r-5)
            
            # Check for overlap with existing objects
            overlap = False
            for obj in objects:
                ox, oy, orad = obj
                dist = math.sqrt((cx-ox)**2 + (cy-oy)**2)
                if dist < (r + orad) * 0.8:  # 80% overlap threshold
                    overlap = True
                    break
            
            if not overlap:
                break
        
        objects.append((cx, cy, r))
        bbox = draw_realistic_balloon(draw, shape, cx, cy, r, color_variants)
        cxn, cyn, wn, hn = bbox_to_yolo(bbox, W, H)
        ann_lines.append(f"0 {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}")
        meta["shapes"].append({
            "shape": shape,
            "color": color_name,
            "bbox": bbox,
            "distance_factor": distance_factor
        })
    
    # Apply post-processing effects
    im = apply_lighting_effects(im)
    im = apply_motion_blur(im)
    
    # Additional augmentation
    if random.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(0.5))
    if random.random() < 0.3:
        im = ImageOps.autocontrast(im, cutoff=1)
    
    # Save image and label
    img_path = os.path.join(out_images, f"{idx:06d}.jpg")
    lbl_path = os.path.join(out_labels, f"{idx:06d}.txt")
    im.save(img_path, quality=95)
    with open(lbl_path,"w") as f: f.write("\n".join(ann_lines))
    
    return img_path, lbl_path, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/enhanced_synth_samples", help="output root")
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
    
    print("Generating enhanced training data...")
    for i in range(args.train):
        if i % 50 == 0:
            print(f"  Progress: {i}/{args.train}")
        img,lbl,meta = gen_enhanced_image(train_img,train_lbl,i,args.W,args.H)
        metas["train"].append({"img":img,"label":lbl,**meta})
    
    print("Generating enhanced validation data...")
    for i in range(args.val):
        img,lbl,meta = gen_enhanced_image(val_img,val_lbl,i,args.W,args.H)
        metas["val"].append({"img":img,"label":lbl,**meta})
    
    with open(os.path.join(args.out,"meta.json"),"w") as f:
        json.dump(metas,f,indent=2)
    
    print(f"Enhanced dataset complete! Train={args.train}, Val={args.val} → {args.out}")

if __name__ == "__main__":
    main()
