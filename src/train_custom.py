# src/train_custom.py
"""
Robust training script for YOLOv8 (Ultralytics).
Usage examples (local CPU small smoke test):
    python src/train_custom.py --data data/data.yaml --weights yolov8n.pt --epochs 5 --imgsz 320 --batch 4 --name smoke_test --device cpu

Recommended for real training on GPU (Colab / NVIDIA):
    python src/train_custom.py --data data/data.yaml --weights yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --name my_experiment --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on custom dataset (Ultralytics)")
    p.add_argument("--data", type=str, default="data/data.yaml", help="path to data.yaml")
    p.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights (yolov8n.pt recommended for quick tests)")
    p.add_argument("--epochs", type=int, default=50, help="number of epochs")
    p.add_argument("--imgsz", type=int, default=640, help="image size for training (square)")
    p.add_argument("--batch", type=int, default=16, help="batch size")
    p.add_argument("--project", type=str, default="runs/train", help="save results to project/name")
    p.add_argument("--name", type=str, default="exp", help="experiment name")
    p.add_argument("--device", type=str, default="", help="device to use, e.g. 'cpu', 'cuda', or '' for auto")
    p.add_argument("--workers", type=int, default=4, help="dataloader workers")
    p.add_argument("--exist_ok", action="store_true", help="overwrite existing runs folder")
    return p.parse_args()

def check_files(args):
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] data file not found: {data_path.resolve()}")
        print("Create data/data.yaml as described in README and try again.")
        sys.exit(1)

    # minimal sanity: check train and val dirs referenced in yaml
    try:
        import yaml
        with open(data_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse {data_path}: {e}")
        sys.exit(1)

    for key in ("train", "val"):
        if key not in cfg:
            print(f"[ERROR] '{key}' not found in {data_path}. Make sure file contains 'train' and 'val' paths.")
            sys.exit(1)
        if not Path(cfg[key]).exists():
            print(f"[ERROR] Path specified in data.yaml does not exist: {cfg[key]}")
            sys.exit(1)

    print("[INFO] data.yaml OK. train and val paths exist.")

def main():
    args = parse_args()

    # Lazy import ultralytics (so script fails clearly if not installed)
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERROR] ultralytics is not installed or failed to import.")
        print("Install via: pip install ultralytics")
        print("Exception:", e)
        sys.exit(1)

    check_files(args)

    # Ensure project dir exists
    os.makedirs(args.project, exist_ok=True)

    print(f"[INFO] Training config:")
    print(f"  data:    {args.data}")
    print(f"  weights: {args.weights}")
    print(f"  epochs:  {args.epochs}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  batch:   {args.batch}")
    print(f"  project: {args.project}/{args.name}")
    print(f"  device:  {'auto' if args.device=='' else args.device}")
    print()

    # Create / load model
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"[ERROR] Failed loading weights '{args.weights}': {e}")
        print("Make sure the file exists or use a pretrained name like 'yolov8n.pt'")
        sys.exit(1)

    # Train
    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device,   # '' = auto, 'cpu' = cpu, 'cuda' = auto GPU
            workers=args.workers,
            exist_ok=args.exist_ok
        )
    except Exception as e:
        print("[ERROR] Training failed with exception:")
        print(e)
        print("If you are on Windows with AMD GPU (ROCm not available), use --device cpu or run on Colab.")
        sys.exit(1)

    print("[INFO] Training finished. Check the runs directory for outputs (runs/train/<name>/weights/best.pt)")

if __name__ == "__main__":
    main()
