import os
import argparse
import cv2
from ultralytics import YOLO


def load_model(weights_path="models/best.pt"):
    """Load YOLOv8 model, fallback if best.pt missing."""
    if os.path.exists(weights_path):
        print(f"[INFO] Loading custom model: {weights_path}")
        return YOLO(weights_path)
    else:
        print("[WARNING] best.pt not found. Using yolov8n.pt.")
        return YOLO("yolov8n.pt")


def detect_image(image_path, output_dir="outputs", weights_path="models/best.pt"):
    """Run YOLO on an image and save annotated output."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_model(weights_path)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("[ERROR] Failed to load image.")

    # Run inference
    results = model(image, verbose=False)[0]

    # Annotate result
    annotated = results.plot()

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"annotated_{filename}")

    cv2.imwrite(output_path, annotated)
    print(f"[INFO] Saved annotated image → {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Image Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", default="models/best.pt", help="Model weights")
    parser.add_argument("--output", default="outputs", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_image(args.image, args.output, args.weights)
