from pathlib import Path
import pickle
import numpy as np
from PIL import Image
import pillow_heif
import json

pillow_heif.register_heif_opener()

with open("faces.pkl", "rb") as f:
    data = pickle.load(f)

files = data["files"]
locations = data["locations"]
labels = data["labels"]

output_dir = Path("clusters")
output_dir.mkdir(exist_ok=True)

# для каждого кластера сохраняем по 3 примера
unique_labels = sorted(set(labels))
if -1 in unique_labels:
    unique_labels.remove(-1)

for label in unique_labels:
    cluster_dir = output_dir / str(label)
    cluster_dir.mkdir(exist_ok=True)

    indices = [i for i, l in enumerate(labels) if l == label][:3]

    for idx in indices:
        img = Image.open(files[idx]).convert("RGB")
        top, right, bottom, left = locations[idx]
        # немного расширяем рамку
        pad = 30
        face = img.crop((
            max(0, left - pad),
            max(0, top - pad),
            min(img.width, right + pad),
            min(img.height, bottom + pad)
        ))
        face = face.resize((150, 150))
        face.save(cluster_dir / f"{idx}.jpg")

print("Готово! Открой папку clusters/ и посмотри на кластеры")
print("Потом запустим скрипт чтобы дать имена")