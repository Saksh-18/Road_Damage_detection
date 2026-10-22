"""Test the pretrained RDD2022 YOLOv8m model on one image or a whole folder.

Usage (run from your app folder):
    python test_pretrained.py path/to/photo.jpg
    python test_pretrained.py path/to/photo.jpg 0.10      # lower confidence
    python test_pretrained.py path/to/folder_of_images
"""
import sys

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

if len(sys.argv) < 2:
    sys.exit("Please give an image or folder path, e.g.\n"
             "  python test_pretrained.py my_photo.jpg")

source = sys.argv[1]
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

weights = hf_hub_download("dronefreak/rdd2022-yolov8m", "best.pt")
model = YOLO(weights)
print("Classes:", model.names)

results = model.predict(source, conf=conf, save=True)
for r in results:
    print(f"{r.path}: {len(r.boxes)} detection(s)")
print("Annotated images saved in:", results[0].save_dir)
