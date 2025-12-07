"""
VisionGuard - Robust Webcam Detection (cleaned & production-ready)

Features:
 - CLI args: --source, --weights, --imgsz, --device, --flip, --record, --out
 - Loads custom weights (falls back to yolov8n.pt)
 - Silent imgsz auto-adjust to multiple-of-32 (no spammy repeated warnings)
 - Model warm-up to reduce first-frame delay
 - Safe inference with fallback to raw frame (no crashes)
 - FPS smoothing (EMA) for stable display
 - Press 's' to save screenshot, 'r' to start/stop recording, 'q' to quit
 - Optional video recording (MJPG) into outputs/videos/
 - Cross-platform path handling, Windows-friendly usage
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ultralytics import
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "ultralytics is required. Install with: pip install ultralytics"
    ) from e


def parse_args():
    p = argparse.ArgumentParser(description="VisionGuard - Webcam Detection")
    p.add_argument("--source", type=int, default=0, help="Webcam source index (default: 0)")
    p.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to model weights (default: models/best.pt). Use yolov8n.pt for small fallback.",
    )
    p.add_argument("--imgsz", type=int, default=736, help="Inference size (must be multiple of 32). Default: 736")
    p.add_argument("--device", type=str, default="", help="Device: '', 'cpu' or 'cuda' (empty = auto)")
    p.add_argument("--flip", action="store_true", help="Horizontally flip webcam frame (mirror view)")
    p.add_argument("--record", action="store_true", help="Record output video to outputs/videos/")
    p.add_argument("--out", type=str, default="outputs", help="Output folder for screenshots/videos")
    p.add_argument("--preview-width", type=int, default=1280, help="Preview window width")
    p.add_argument("--preview-height", type=int, default=720, help="Preview window height")
    return p.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def choose_device(user_device: str = "") -> str:
    """
    Decide which device to use:
    - if user passes --device, use that
    - otherwise: 'cuda' if available, else 'cpu'
    """
    if user_device:  # explicit override
        return user_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except Exception:
        # torch not installed or not usable -> default to cpu
        return "cpu"


def load_model(weights_path: str, device: str = "") -> YOLO:
    weights_path = str(weights_path)
    if Path(weights_path).exists():
        print(f"[INFO] Loading custom model: {weights_path}")
        try:
            model = YOLO(weights_path)
            if device:
                model.to(device)
            return model
        except Exception as e:
            print(f"[WARNING] Failed to load {weights_path}: {e}. Falling back to 'yolov8n.pt'.")
    else:
        print(f"[WARNING] '{weights_path}' not found. Using 'yolov8n.pt' instead.")

    model = YOLO("yolov8n.pt")
    if device:
        try:
            model.to(device)
        except Exception as e:
            print(f"[WARNING] Failed to move fallback model to device '{device}': {e}")
    return model


def adjust_imgsz(imgsz: int) -> int:
    """Return imgsz rounded up to the nearest multiple of 32 (silent single warning)."""
    if imgsz % 32 == 0:
        return imgsz
    new = ((imgsz + 31) // 32) * 32
    print(f"[INFO] imgsz={imgsz} adjusted to {new} (multiple of 32).")
    return new


def warmup_model(model: YOLO, imgsz: int):
    """Run a single dummy inference to avoid long first-frame delay."""
    try:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = model(dummy, verbose=False, imgsz=imgsz)
        print("[INFO] Model warm-up complete.")
    except Exception as e:
        print(f"[WARNING] Model warm-up failed (non-fatal): {e}")


def create_video_writer(path: str, fps: float, width: int, height: int):
    # MJPG is widely supported on Windows; use mp4 on other systems if desired
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(path, fourcc, max(1.0, fps), (width, height))


def draw_fps(frame, fps: float):
    text = f"FPS: {fps:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return frame


def run_webcam(
    source: int,
    weights: str,
    imgsz: int,
    device: str,
    flip: bool,
    record: bool,
    out_dir: str,
    preview_w: int,
    preview_h: int,
):
    # Prepare outputs
    ensure_dir(out_dir)
    screenshots_dir = Path(out_dir) / "screenshots"
    videos_dir = Path(out_dir) / "videos"
    ensure_dir(screenshots_dir)
    ensure_dir(videos_dir)

    # Load model
    model = load_model(weights, device)

    # Adjust imgsz (single info message)
    imgsz_valid = adjust_imgsz(imgsz)

    # Warm-up
    warmup_model(model, imgsz_valid)

    # Open camera
    cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Try another --source index or close other apps using the camera.")
        return

    # Request preview resolution (may be ignored by camera driver)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, preview_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_h)

    win_name = "VisionGuard - Webcam Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, preview_w, preview_h)

    prev_time = time.time()
    ema_fps = 0.0
    alpha = 0.2  # EMA smoothing factor

    recording = False
    video_writer: Optional[cv2.VideoWriter] = None

    print("[INFO] Webcam started. Controls: 's' save screenshot, 'r' start/stop recording, 'q' quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        if flip:
            frame = cv2.flip(frame, 1)

        # Inference with safe fallback
        try:
            results = model(frame, verbose=False, imgsz=imgsz_valid)[0]
            annotated = results.plot() if results is not None else frame.copy()
        except Exception as e:
            # If inference fails for a frame, print once and fallback to raw frame
            print(f"[WARN] Inference failed for a frame: {e}")
            annotated = frame.copy()

        # FPS calculation and smoothing
        now = time.time()
        raw_fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now
        ema_fps = alpha * raw_fps + (1 - alpha) * ema_fps if ema_fps > 0 else raw_fps

        annotated = draw_fps(annotated, ema_fps)

        cv2.imshow(win_name, annotated)

        # If recording, write frame into video (use annotated frame for consistency)
        if recording and video_writer is not None and video_writer.isOpened():
            try:
                # Ensure frame size matches writer; resize if needed
                h, w = annotated.shape[:2]
                writer_frame = cv2.resize(annotated, (w, h))
                video_writer.write(writer_frame)
            except Exception as e:
                print(f"[WARN] Failed to write video frame: {e}")

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            # Save screenshot with timestamp
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = screenshots_dir / f"screenshot_{ts}.jpg"
            try:
                cv2.imwrite(str(out_path), annotated)
                print(f"[INFO] Screenshot saved: {out_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save screenshot: {e}")

        elif key == ord("r"):
            # Toggle recording
            recording = not recording
            if recording:
                ts = time.strftime("%Y%m%d_%H%M%S")
                video_path = videos_dir / f"record_{ts}.avi"
                # Use current EMA fps for writer (fallback to 15)
                writer_fps = max(1.0, ema_fps) if ema_fps > 0 else 15.0
                # Use preview size for writer
                try:
                    video_writer = create_video_writer(str(video_path), writer_fps, preview_w, preview_h)
                    if video_writer.isOpened():
                        print(f"[INFO] Recording started: {video_path} @ {writer_fps:.2f} FPS")
                    else:
                        print(f"[ERROR] Video writer failed to open: {video_path}")
                        recording = False
                        video_writer = None
                except Exception as e:
                    print(f"[ERROR] Failed to start recording: {e}")
                    recording = False
                    video_writer = None
            else:
                # Stop recording
                if video_writer is not None:
                    video_writer.release()
                    print("[INFO] Recording stopped.")
                    video_writer = None

        elif key == ord("q"):
            print("[INFO] Quit signal received.")
            break

    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam detection stopped.")


if __name__ == "__main__":
    args = parse_args()
    # decide device: user choice or auto
    chosen_device = choose_device(args.device)
    print(f"[INFO] Using device: {chosen_device}")

    run_webcam(
        source=args.source,
        weights=args.weights,
        imgsz=args.imgsz,
        device=chosen_device,
        flip=args.flip,
        record=args.record,
        out_dir=args.out,
        preview_w=args.preview_width,
        preview_h=args.preview_height,
    )
