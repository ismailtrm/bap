#!/usr/bin/env python3
"""
Train a YOLOv8 nano model for single class 'balloon' using Ultralytics.
Enhanced with early stopping, dropout, and comprehensive training parameters.
"""
import argparse, os, json, sys
try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not available. Install with: pip install ultralytics")
    raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    ap.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    ap.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    ap.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    ap.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    ap.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    ap.add_argument("--cos_lr", action="store_true", help="Use cosine learning rate scheduler")
    ap.add_argument("--save_period", type=int, default=10, help="Save checkpoint every N epochs")
    args = ap.parse_args()

    # Load model
    model = YOLO(args.weights)
    
    # Training configuration with early stopping and regularization
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'project': "runs/train",
        'name': "yolo_balloon",
        
        # Early stopping
        'patience': args.patience,
        
        # Regularization
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        
        # Optimizer settings
        'lr0': args.lr,
        'momentum': args.momentum,
        'warmup_epochs': args.warmup_epochs,
        'cos_lr': args.cos_lr,
        
        # Data augmentation
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation augmentation
        'hsv_v': 0.4,    # HSV-Value augmentation
        'degrees': 0.0,  # Rotation degrees
        'translate': 0.1, # Translation fraction
        'scale': 0.5,    # Scale gain
        'shear': 0.0,    # Shear degrees
        'perspective': 0.0, # Perspective gain
        'flipud': 0.0,   # Flip up-down probability
        'fliplr': 0.5,   # Flip left-right probability
        'mosaic': 1.0,   # Mosaic probability
        'mixup': 0.0,    # Mixup probability
        
        # Model settings
        'save_period': args.save_period,
        'save': True,
        'save_txt': True,
        'save_conf': True,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'visualize': False,
        'augment': True,
        'agnostic_nms': False,
        'retina_masks': False,
        'overlap_mask': True,
        'mask_ratio': 4,
    }
    
    print(f"Starting training with enhanced parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Dropout rate: {args.dropout}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    
    # Train the model
    results = model.train(**train_args)
    
    print("Training complete. See runs/train/yolo_balloon/ for results.")
    print(f"Best model saved at: runs/train/yolo_balloon/weights/best.pt")
    print(f"Last model saved at: runs/train/yolo_balloon/weights/last.pt")

if __name__ == "__main__":
    main()
