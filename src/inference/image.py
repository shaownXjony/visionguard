import os
import argparse
import cv2
from ultralytics import YOLO


def validate_device(device: str) -> str:
    """
    Validate requested device and safely fall back to CPU if needed.
    """
    if not device:
        return ""

    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[WARNING] CUDA requested but not available. Falling back to CPU.")
                return "cpu"
        except ImportError:
            print("[WARNING] PyTorch not available. Falling back to CPU.")
            return "cpu"

    return device


def load_model(weights_path: str) -> YOLO:
    """
    Load a YOLOv8 model with fallback.
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
    verbose: bool,
) -> None:
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    device = validate_device(device)

    if verbose:
        print("[INFO] Verbose mode enabled")

    model = load_model(weights_path)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return

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

    annotated = results.plot()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"annotated_{os.path.basename(image_path)}"
        )
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Saved annotated image → {output_path}")
    else:
        print("[INFO] --save not set, result not written to disk")

    print(f"[INFO] Objects detected: {len(results.boxes)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionGuard - Image Inference"
    )

    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="", help="cpu | cuda | auto")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    detect_image(
        image_path=args.image,
        weights_path=args.weights,
        conf=args.conf,
        device=args.device,
        save=args.save,
        output_dir=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
