# 🛡️ VisionGuard

### Modular Object Detection & Training Framework (YOLOv8)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)

VisionGuard is a **modular computer vision framework** built on **YOLOv8 (Ultralytics)** and **OpenCV**. It supports image inference, webcam inference, model training, and model comparison, with utilities for logging and reproducibility.

The project is structured to reflect real-world ML pipelines, separating inference, training, utilities, and tools.

---

## 🚀 Features (Implemented)

- ✅ Image-based object detection
- ✅ Real-time webcam detection
- ✅ YOLOv8 model training pipeline
- ✅ Model comparison utilities
- ✅ Reproducibility & logging utilities
- ✅ Clean, scalable project structure

---

## 📁 Project Structure

<!-- TREE START -->
<!-- TREE END -->

---

## 🧱 Tech Stack

- **Python 3.9+**
- **Ultralytics YOLOv8**
- **OpenCV**
- **PyTorch**
- **NumPy**

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/shaownXjony/visionguard.git
cd visionguard
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 YOLOv8 Model Setup (One-Time)

Download YOLOv8 weights once:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

After this, all inference runs **offline**.

---

## ▶️ Usage

### 🖼️ Image Inference

Run object detection on an image:

```bash
python src/inference/image.py --image path/to/image.jpg
```

**📌 Output:**
- Annotated image saved in `outputs/`

---

### 🎥 Webcam Inference

Run real-time webcam detection:

```bash
python src/inference/webcam.py
```

**Controls:**
- Press **`q`** to quit

---

### 🏋️ Model Training

Train a YOLOv8 model:

```bash
python src/training/train.py
```

**📌 Notes:**
- Uses Ultralytics YOLOv8 defaults
- Training outputs are saved under `runs/` (YOLO default)

---

### 📊 Model Comparison

Compare different YOLOv8 model variants:

```bash
python src/tools/compare_models.py
```

**Useful for:**
- Speed vs accuracy trade-offs
- Model benchmarking

---

### 📥 Download Sample Data

Fetch sample images or datasets:

```bash
python tools/download_samples.py
```

---

## 🧩 Utilities

### Logging
- Centralized logging via `utils/logger.py`

### Reproducibility
- Seed control and deterministic behavior via `utils/reproducibility.py`

These utilities help ensure repeatable experiments.

---

## 📂 Outputs

- **Inference results** → `outputs/`
- **Trained models** → `runs/` (YOLO default)
- **Custom weights** (optional) → `models/`

---

## 🛣️ Roadmap (Planned)

- ⬜ Streamlit / GUI interface
- ⬜ Video file inference
- ⬜ Config-file-driven parameters
- ⬜ Experiment tracking dashboard
- ⬜ Docker support

---

## 🤝 Contributing

Contributions are welcome.

1. **Fork** the repository
2. **Create** a feature branch
3. **Commit** your changes
4. **Open** a pull request

Keep changes modular and documented.

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Shaown Jony**

- GitHub: [@shaownXjony](https://github.com/shaownXjony)
- Project: [https://github.com/shaownXjony/visionguard](https://github.com/shaownXjony/visionguard)

---

## ⭐ Final Notes

VisionGuard demonstrates:

- Practical YOLOv8 usage
- Clean ML project structuring
- Separation of inference, training, and utilities
- Reproducibility-aware experimentation

If you find this project useful, consider giving it a **⭐**.