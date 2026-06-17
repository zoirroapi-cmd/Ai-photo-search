from pathlib import Path
import face_recognition
import numpy as np
from PIL import Image
import pillow_heif
import pickle

pillow_heif.register_heif_opener()

people_dir = Path("people_test")

# 1. Загружаем известные лица
known_encodings = []
known_names = []

print("Загружаем образцы лиц...")

for img_path in people_dir.iterdir():
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".heic"]:
        continue

    name = img_path.stem.split("_")[0]  # nika_1.jpg -> nika

    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)

    encodings = face_recognition.face_encodings(img_array)

    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)
        print(f"  {img_path.name} -> {name}")
    else:
        print(f"  {img_path.name} -> лицо не найдено!")

print(f"\nЗагружено образцов: {len(known_encodings)}\n")

# 2. Загружаем все лица из основной базы (уже посчитаны в faces.pkl)
with open("faces.pkl", "rb") as f:
    data = pickle.load(f)

all_encodings = np.array(data["encodings"])
all_files = data["files"]

# 3. Сравниваем каждое известное лицо со всеми
print("Ищем совпадения...\n")

results = {name: [] for name in set(known_names)}

for known_enc, known_name in zip(known_encodings, known_names):
    distances = face_recognition.face_distance(all_encodings, known_enc)

    for i, dist in enumerate(distances):
        if dist < 0.45:  # порог похожести, можно подбирать
            if all_files[i] not in results[known_name]:
                results[known_name].append(all_files[i])

for name, file_list in results.items():
    print(f"{name}: найдено {len(file_list)} фото")
    for f in file_list:
        print(f"  - {Path(f).name}")