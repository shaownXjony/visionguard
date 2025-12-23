# 🛡️ VisionGuard – Real-Time Object Detection App (YOLOv8)

VisionGuard is a real-time object detection system built with **Python**, **YOLOv8**, **OpenCV**, and **Streamlit**.  
This project will detect objects using a webcam, allow image uploads, and support custom YOLO model training.

## 🚀 Features (Coming Soon)
- ✅ Real-time webcam object detection (OpenCV + YOLOv8)
- ✅ Image-based detection script
- ✅ Custom YOLO training script (with sample dataset)
- ✅ Clean, modular Python code with CLI arguments
- 🕒 Optional Streamlit / GUI interface (planned for future)

---

## 📁 Project Structure (Initial Setup)

```bash
visionguard/
│
├── data/
│   ├── dataset/            # training images + labels (not in git)
│   └── samples/            # demo images
│
├── src/
│   ├── detect_webcam.py    # real-time webcam detector
│   ├── detect_image.py     # single-image detector
│   └── train_custom.py     # YOLO training helper
│
├── tools/
│   └── download_samples.py # download small sample dataset
│
├── outputs/                # screenshots + videos (examples)
├── runs/                   # training runs (ignored in git)
├── notebooks/
│   └── 01_experiments.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```


## 🧱 Tech Stack
- Python 3+
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Streamlit

---

## 📌 Note

Detailed documentation will be added as features are developed.  
Stay tuned for updates!

---

## 🎥 Run Real-Time Webcam Detection

## Install dependencies:

```bash
pip install -r requirements.txt
```

```bash
python src/detect_webcam.py --source 0 --flip
```
## Run detection on a single image
```bash
python src/detect_image.py --image data/samples/demo.jpg --weights yolov8n.pt --output outputs
```
## 📸 Demo Screenshots

<div align="center">
  <img src="outputs/annotated_test2.jpg" width="500"/>
  <br/>
  <em>Annotated YOLOv8 Detection Output</em>
  <br/><br/>
  <img src="outputs/screenshots/screenshot.jpg" width="500"/>
  <br/>
  <em>Webcam Snapshot</em>
</div>
