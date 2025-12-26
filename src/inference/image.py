import os
import argparse
import cv2
from ultralytics import YOLO


def load_model(weights_path: str) -> YOLO:
    """
    Load a YOLOv8 model.

    Tries to load a custom model from `weights_path`.
    If it doesn't exist, falls back to the default `yolov8n.pt`.
    """
    if weights_path and os.path.exists(weights_path):
        print(f"[INFO] Loading custom model: {weights_path}")
        return YOLO(weights_path)

    print("[WARNING] Custom weights not found. Falling back to yolov8n.pt")
    return YOLO("yolov8n.pt")


def detect_image(
    image_path: str,
    weights_path: str,
    conf: float,
    device: str,
    save: bool,
    output_dir: str,
) -> None:
    """
    Run YOLO object detection on a single image.

    Args:
        image_path: Path to the input image.
        weights_path: Path to YOLO model weights.
        conf: Confidence threshold.
        device: Device to use (cpu, cuda, or auto).
        save: Whether to save annotated image.
        output_dir: Directory to save results.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        print(f"[HINT] Current directory: {os.getcwd()}")
        return

    # Load model
    model = load_model(weights_path)

    # Read image
    image = cv2.imread(image_path)  # pylint: disable=no-member
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return

    # Run inference
    try:
        results = model(
            image,
            conf=conf,
            device=device if device else None,
            verbose=False,
        )[0]
    except Exception as exc: 
        print(f"[ERROR] Inference failed: {exc}")
        return

    # Annotate result
    annotated = results.plot()

    # Save or display
    if save:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"annotated_{os.path.basename(image_path)}"
        )
        cv2.imwrite(output_path, annotated)  
        print(f"[INFO] Saved annotated image → {output_path}")
    else:
        print("[INFO] --save not set, result not written to disk")

    # Summary
    num_objects = len(results.boxes)
    print(f"[INFO] Objects detected: {num_objects}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="VisionGuard - Image Inference"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="Path to model weights (default: models/best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device to use: cpu, cuda, or auto (default)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated image",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory (default: outputs)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    detect_image(
        image_path=args.image,
        weights_path=args.weights,
        conf=args.conf,
        device=args.device,
        save=args.save,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
