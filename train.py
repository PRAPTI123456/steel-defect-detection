"""
Train a YOLOv8 defect detector.

Usage:
    python train.py                      # defaults (yolov8n, 50 epochs)
    python train.py --model yolov8s.pt --epochs 100 --batch 16
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m .pt (downloads automatically)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="neu_det")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name=args.name,
        patience=15,       # stop early if no improvement
        degrees=10,        # light augmentation suited to flat surface images
        flipud=0.5,
        fliplr=0.5,
    )

    metrics = model.val()
    print(f"mAP@0.5     : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"Best weights: runs/{args.name}/weights/best.pt  (check the actual folder printed above)")


if __name__ == "__main__":
    main()
