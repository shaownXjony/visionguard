import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import platform

import cv2
from ultralytics import YOLO


# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class WebcamDetector:
    """
    Professional real-time webcam/video object detection system.
    """

    def __init__(
        self,
        source: Union[int, str],
        weights: str,
        conf_threshold: float,
        device: str,
        flip: bool,
        output_dir: str,
    ):
        self.source = self._parse_source(source)
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.device = self._validate_device(device)
        self.flip = flip
        self.output_dir = Path(output_dir)

        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.ema_fps = 30.0
        self.prev_time = time.time()

        self.screenshot_dir = self.output_dir / "screenshots"
        self.video_dir = self.output_dir / "videos"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.model = self._load_model()
        self.capture = self._initialize_capture()

    # -------------------- Helpers --------------------

    def _parse_source(self, source: Union[int, str]) -> Union[int, str]:
        return int(source) if isinstance(source, str) and source.isdigit() else source

    def _validate_device(self, device: str) -> str:
        """
        Validate requested device and safely fall back to CPU if needed.
        """
        if not device:
            return ""

        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning(
                        "CUDA requested but not available. Falling back to CPU."
                    )
                    return "cpu"
            except ImportError:
                logger.warning("PyTorch not available. Falling back to CPU.")
                return "cpu"

        return device

    def _load_model(self) -> YOLO:
        if os.path.exists(self.weights):
            logger.info(f"Loading model: {self.weights}")
            return YOLO(self.weights)

        logger.warning("Weights not found. Falling back to yolov8n.pt")
        return YOLO("yolov8n.pt")

    def _initialize_capture(self) -> cv2.VideoCapture:
        if isinstance(self.source, int) and platform.system() == "Windows":
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            logger.error(f"Cannot open source: {self.source}")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.ema_fps = fps if fps and fps > 1 else 30.0
        return cap

    def _create_video_writer(self, frame_shape):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.video_dir / f"recording_{timestamp}.mp4"
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(path), fourcc, self.ema_fps, (w, h))

    def _calculate_fps(self):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            self.ema_fps = 0.9 * self.ema_fps + 0.1 * (1.0 / dt)

    def _draw_overlay(self, frame):
        cv2.putText(
            frame,
            f"FPS: {self.ema_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if self.is_recording:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "REC",
                (frame.shape[1] - 80, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    def _process_frame(self, frame):
        if self.flip:
            frame = cv2.flip(frame, 1)

        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device if self.device else None,
            verbose=False,
        )[0]

        annotated = results.plot()
        self._calculate_fps()
        self._draw_overlay(annotated)
        return annotated

    # -------------------- Main Loop --------------------

    def run(self):
        logger.info("Press Q or ESC to quit | S: Screenshot | R: Record")

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break

                annotated = self._process_frame(frame)

                if self.is_recording and self.video_writer:
                    self.video_writer.write(annotated)

                cv2.imshow("VisionGuard Webcam", annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = self.screenshot_dir / f"snap_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(path), annotated)
                    logger.info(f"Screenshot saved: {path}")
                if key == ord("r"):
                    self.is_recording = not self.is_recording
                    self.video_writer = (
                        self._create_video_writer(frame.shape)
                        if self.is_recording
                        else None
                    )

                self.frame_count += 1

        finally:
            self.capture.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()

            logger.info(f"Frames processed: {self.frame_count}")
            logger.info("VisionGuard closed")


# -------------------- CLI --------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionGuard - Real-Time Webcam Object Detection"
    )

    parser.add_argument("--source", default="0", help="Camera index or video file")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="", help="cpu | cuda | auto")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    detector = WebcamDetector(
        source=args.source,
        weights=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        flip=args.flip,
        output_dir=args.output,
    )

    if args.record:
        detector.is_recording = True

    detector.run()


if __name__ == "__main__":
    main()
