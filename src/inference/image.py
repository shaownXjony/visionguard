import os
import argparse
import cv2  # pylint: disable=import-error
from ultralytics import YOLO


def load_model(weights_path: str = "models/best.pt") -> YOLO:
    """
    Load a YOLOv8 model.

    Tries to load a custom model from `weights_path`.
    If it doesn't exist, falls back to the default `yolov8n.pt`.
    
    Args:
        weights_path: Path to YOLO model weights file.
        
    Returns:
        YOLO: Loaded YOLOv8 model instance.
    """
    if os.path.exists(weights_path):
        print(f"[INFO] Loading custom model: {weights_path}")
        return YOLO(weights_path)

    print("[WARNING] Custom weights not found. Falling back to yolov8n.pt")
    return YOLO("yolov8n.pt")


def detect_image(
    image_path: str,
    output_dir: str = "outputs",
    weights_path: str = "models/best.pt"
) -> None:
    """
    Run YOLO object detection on a single image and save the annotated result.

    Args:
        image_path: Path to the input image.
        output_dir: Directory where the annotated image will be saved.
        weights_path: Path to YOLO model weights.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        print(f"[HINT] Current directory: {os.getcwd()}")
        print("[HINT] Make sure the image path is correct")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_model(weights_path)

    # Read image
    image = cv2.imread(image_path)  # pylint: disable=no-member
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return

    # Run inference
    try:
        results = model(image, verbose=False)[0]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[ERROR] Inference failed: {exc}")
        return

    # Annotate result
    annotated = results.plot()

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"annotated_{filename}")

    cv2.imwrite(output_path, annotated)  # pylint: disable=no-member
    print(f"[INFO] Saved annotated image → {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Image Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="Path to model weights (e.g., models/best.pt or yolov8n.pt)",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory to save annotated image",
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    detect_image(args.image, args.output, args.weights)


if __name__ == "__main__":
    main()