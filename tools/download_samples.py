import os, requests, shutil

# Create dataset folders
os.makedirs("data/dataset/images/train", exist_ok=True)
os.makedirs("data/dataset/images/val", exist_ok=True)
os.makedirs("data/dataset/labels/train", exist_ok=True)
os.makedirs("data/dataset/labels/val", exist_ok=True)

# Sample images (Unsplash royalty-free placeholders)
urls = [
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=1280",
    "https://images.unsplash.com/photo-1517487881594-2787fef5ebf7?w=1280",
    "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?w=1280"
]

print("[INFO] Downloading sample images...")

for i, u in enumerate(urls):
    r = requests.get(u, timeout=30)
    filename = f"data/dataset/images/train/img_{i:03d}.jpg"
    open(filename, "wb").write(r.content)
    print(f"Saved {filename}")

# Copy last image to validation set
shutil.copy(
    "data/dataset/images/train/img_002.jpg",
    "data/dataset/images/val/img_002.jpg"
)

print("[INFO] Copied one image to validation set.")

# Create dummy YOLO labels
dummy_label = "0 0.5 0.5 1.0 1.0\n"

for folder in ["train", "val"]:
    img_folder = f"data/dataset/images/{folder}"
    label_folder = f"data/dataset/labels/{folder}"

    for fname in os.listdir(img_folder):
        if fname.lower().endswith(".jpg"):
            name = fname.split(".")[0]
            out = f"{label_folder}/{name}.txt"
            with open(out, "w") as f:
                f.write(dummy_label)

print("[INFO] Dummy YOLO labels created.")
print("[DONE] Sample dataset is ready 🎉")
