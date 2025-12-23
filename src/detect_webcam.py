import os
import time
import argparse
from datetime import datetime
from pathlib import Path
import platform

import cv2
from ultralytics import YOLO


# ----------------- Model loading ----------------- #
def load_model(weights: str) -> YOLO:
    """Load YOLO model, fallback to yolov8n.pt if custom weights not found."""
    if os.path.exists(weights):
        print(f"[INFO] Loading custom model → {weights}")
        return YOLO(weights)

    print(f"[WARNING] Weights not found: {weights}. Falling back to yolov8n.pt")
    return YOLO("yolov8n.pt")


# ----------------- Video writer ----------------- #
def create_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create a MP4 video writer."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


# ----------------- Webcam loop ----------------- #
def run_webcam(
    source,
    weights: str = "models/best.pt",
    flip: bool = False,
    record: bool = False,
    out_dir: str = "outputs",
):
    # figure out source type
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # choose backend (Windows black-screen fix)
    if isinstance(source, int) and platform.system() == "Windows":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera/source: {source}")
        return

    model = load_model(weights)

    # output dirs
    screenshot_dir = Path(out_dir) / "screenshots"
    video_dir = Path(out_dir) / "videos"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps != fps:  # NaN or 0
        fps = 30.0

    writer = None
    ema_fps = fps

    print("[INFO] Press 'q' to quit, 's' for screenshot, 'r' to toggle recording")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame. Exiting.")
            break

        if flip:
            frame = cv2.flip(frame, 1)

        # FPS timing
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            ema_fps = 0.9 * ema_fps + 0.1 * current_fps

        # YOLO inference
        results = model(frame, verbose=False)
        annotated = results[0].plot()

        # overlay FPS text
        cv2.putText(
            annotated,
            f"FPS: {ema_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # recording
        if record:
            if writer is None:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                video_path = video_dir / f"visionguard_{ts}.mp4"
                h, w = annotated.shape[:2]
                writer = create_video_writer(str(video_path), fps, w, h)
                print(f"[INFO] Recording started → {video_path}")
            writer.write(annotated)

        cv2.imshow("VisionGuard Webcam", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            img_path = screenshot_dir / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(img_path), annotated)
            print(f"[INFO] Screenshot saved → {img_path}")

        if key == ord("r"):
            record = not record
            if not record and writer is not None:
                writer.release()
                writer = None
                print("[INFO] Recording stopped")
            elif record:
                # writer will be created on next loop
                print("[INFO] Recording toggled ON")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed. Bye!")


# ----------------- CLI ----------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="VisionGuard Webcam Detection")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument("--weights", default="models/best.pt", help="Path to YOLO weights")
    parser.add_argument("--flip", action="store_true", help="Flip frame horizontally")
    parser.add_argument("--record", action="store_true", help="Record video to file")
    parser.add_argument("--out", default="outputs", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_webcam(
        source=args.source,
        weights=args.weights,
        flip=args.flip,
        record=args.record,
        out_dir=args.out,
    )
