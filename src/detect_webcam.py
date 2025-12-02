import time
import os
import cv2
from ultralytics import YOLO


def load_model(weights_path="models/best.pt"):
    """Load YOLOv8, fallback to yolov8n.pt."""
    if os.path.exists(weights_path):
        print(f"[INFO] Loading custom model: {weights_path}")
        return YOLO(weights_path)
    else:
        print("[WARNING] best.pt not found. Using yolov8n.pt instead.")
        return YOLO("yolov8n.pt")


def draw_fps(frame, fps):
    """Draw FPS on frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    return frame


def get_fps(prev_time):
    """FPS calculator."""
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    return fps, current_time


def run_webcam(source=0, weights_path="models/best.pt"):
    model = load_model(weights_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

   
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "VisionGuard – Webcam Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    prev_time = None
    print("[INFO] Webcam detection running at 1280×720. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break
        frame = cv2.flip(frame, 1)
    
        results = model(frame, verbose=False, imgsz=720)[0]

        annotated = results.plot()

        fps, prev_time = get_fps(prev_time)
        annotated = draw_fps(annotated, fps)

  
        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam detection stopped.")
    

if __name__ == "__main__":
    run_webcam()
