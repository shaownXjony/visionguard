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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class WebcamDetector:
    """
    Professional webcam object detection system.
    
    Handles real-time object detection from webcam or video files with
    recording, screenshot, and FPS monitoring capabilities.
    """
    
    def __init__(
        self,
        source: Union[int, str],
        weights: str = "models/best.pt",
        conf_threshold: float = 0.5,
        flip: bool = False,
        output_dir: str = "outputs"
    ):
        """
        Initialize the webcam detector.
        
        Args:
            source: Camera index (int) or video file path (str).
            weights: Path to YOLO model weights.
            conf_threshold: Confidence threshold for detections (0-1).
            flip: If True, flip frame horizontally.
            output_dir: Base directory for outputs.
        """
        self.source = self._parse_source(source)
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.flip = flip
        self.output_dir = Path(output_dir)
        
        # State variables
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.ema_fps = 30.0
        self.prev_time = time.time()
        
        # Setup directories
        self.screenshot_dir = self.output_dir / "screenshots"
        self.video_dir = self.output_dir / "videos"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Initialize capture
        self.capture = self._initialize_capture()
        
    def _parse_source(self, source: Union[int, str]) -> Union[int, str]:
        """Parse and validate source input."""
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source
    
    def _load_model(self) -> YOLO:
        """Load YOLO model with fallback."""
        if os.path.exists(self.weights):
            logger.info(f"Loading custom model: {self.weights}")
            try:
                return YOLO(self.weights)
            except Exception as exc:
                logger.error(f"Failed to load custom model: {exc}")
                logger.warning("Falling back to yolov8n.pt")
        else:
            logger.warning(f"Weights not found: {self.weights}")
            logger.info("Using default yolov8n.pt")
        
        return YOLO("yolov8n.pt")
    
    def _initialize_capture(self) -> cv2.VideoCapture:
        """Initialize video capture with platform-specific settings."""
        # Windows CAP_DSHOW fix for black screen issues
        if isinstance(self.source, int) and platform.system() == "Windows":
            cap = cv2.VideoCapture(
                self.source, cv2.CAP_DSHOW
            )
        else:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera/source: {self.source}")
            sys.exit(1)
        
        # Get and validate FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps != fps:  # NaN or invalid
            fps = 30.0
        
        self.ema_fps = fps
        logger.info(f"Camera initialized - FPS: {fps:.1f}")
        
        return cap
    
    def _create_video_writer(self, frame_shape: tuple) -> cv2.VideoWriter:
        """Create video writer for recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.video_dir / f"recording_{timestamp}.mp4"
        
        height, width = frame_shape[:2]
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps != fps:
            fps = 30.0
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(video_path), fourcc, fps, (width, height)
        )
        
        logger.info(f"Recording started: {video_path}")
        return writer
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS using exponential moving average."""
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        
        if dt > 0:
            current_fps = 1.0 / dt
            self.ema_fps = 0.9 * self.ema_fps + 0.1 * current_fps
        
        return self.ema_fps
    
    def _draw_overlay(self, frame) -> None:
        """Draw FPS and recording indicator on frame."""
        # FPS counter
        cv2.putText(
            frame,
            f"FPS: {self.ema_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        
        # Recording indicator (red dot + REC text)
        if self.is_recording:
            width = frame.shape[1]
            cv2.circle(
                frame, (width - 30, 30), 10, (0, 0, 255), -1
            )
            cv2.putText(
                frame,
                "REC",
                (width - 80, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        
        # Frame counter
        cv2.putText(
            frame,
            f"Frame: {self.frame_count}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    
    def _save_screenshot(self, frame) -> None:
        """Save current frame as screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = self.screenshot_dir / f"screenshot_{timestamp}.jpg"
        cv2.imwrite(str(screenshot_path), frame)
        logger.info(f"Screenshot saved: {screenshot_path}")
    
    def _toggle_recording(self, frame_shape: tuple) -> None:
        """Toggle video recording on/off."""
        self.is_recording = not self.is_recording
        
        if self.is_recording:
            self.video_writer = self._create_video_writer(frame_shape)
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            logger.info("Recording stopped")
    
    def _process_frame(self, frame):
        """Process single frame through YOLO detection."""
        if self.flip:
            frame = cv2.flip(frame, 1)
        
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        annotated = results[0].plot()
        
        # Calculate FPS
        self._calculate_fps()
        
        # Draw overlays
        self._draw_overlay(annotated)
        
        return annotated
    
    def _handle_keyboard(self, key: int, frame) -> bool:
        """
        Handle keyboard input.
        
        Returns:
            bool: False if should quit, True to continue.
        """
        if key == ord("q") or key == 27:  # q or ESC
            return False
        
        if key == ord("s"):
            self._save_screenshot(frame)
        
        if key == ord("r"):
            self._toggle_recording(frame.shape)
        
        if key == ord("h"):
            self._print_help()
        
        return True
    
    def _print_help(self) -> None:
        """Print keyboard controls to console."""
        print("\n" + "="*50)
        print("KEYBOARD CONTROLS")
        print("="*50)
        print("  q / ESC  - Quit application")
        print("  s        - Save screenshot")
        print("  r        - Toggle recording")
        print("  h        - Show this help")
        print("="*50 + "\n")
    
    def run(self) -> None:
        """Main detection loop."""
        logger.info("Starting VisionGuard detection system")
        self._print_help()
        
        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    logger.warning("Failed to read frame. Exiting.")
                    break
                
                # Process frame
                annotated = self._process_frame(frame)
                
                # Record if enabled
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(annotated)
                
                # Display
                cv2.imshow("VisionGuard", annotated)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard(key, annotated):
                    break
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
        
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        self.capture.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()  
        
        logger.info("="*50)
        logger.info("SESSION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Average FPS: {self.ema_fps:.2f}")
        logger.info(f"Screenshots saved: {len(list(self.screenshot_dir.glob('*.jpg')))}")
        logger.info(f"Videos saved: {len(list(self.video_dir.glob('*.mp4')))}")
        logger.info("="*50)
        logger.info("VisionGuard closed. Goodbye! 👋")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="VisionGuard - Professional Real-Time Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic webcam detection
  python detect_webcam.py
  
  # Use external webcam (index 1)
  python detect_webcam.py --source 1
  
  # Process video file
  python detect_webcam.py --source path/to/video.mp4
  
  # Flip horizontally (useful for selfie mode)
  python detect_webcam.py --flip
  
  # Start recording immediately
  python detect_webcam.py --record
  
  # Custom model with high confidence threshold
  python detect_webcam.py --weights yolov8m.pt --conf 0.7
  
  # All options combined
  python detect_webcam.py --source 0 --weights models/best.pt --conf 0.6 --flip --record

Keyboard Controls (during runtime):
  q / ESC  - Quit application
  s        - Save screenshot
  r        - Toggle recording
  h        - Show help
        """
    )
    
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (0, 1, ...) or video file path (default: 0)"
    )
    parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="Path to YOLO model weights (default: models/best.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections, 0-1 (default: 0.5)"
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip frame horizontally (mirror mode)"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Start recording video immediately"
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for screenshots and videos (default: outputs)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run detector
    detector = WebcamDetector(
        source=args.source,
        weights=args.weights,
        conf_threshold=args.conf,
        flip=args.flip,
        output_dir=args.output
    )
    
    # Start recording if requested
    if args.record:
        detector.is_recording = True
    
    detector.run()


if __name__ == "__main__":
    main()