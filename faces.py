from pathlib import Path
import face_recognition
import pickle
import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

photos_dir = Path("photos")
encodings_path = Path("faces.pkl")

all_encodings = []  # векторы лиц
all_files = []      # к какому файлу относится
all_locations = []  # где на фото

print("Ищем лица...")

for img_path in photos_dir.iterdir():
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".heic"]:
        continue

    print("processing:", img_path.name)

    try:
        # конвертируем в RGB numpy array
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # находим лица
        locations = face_recognition.face_locations(img_array)
        encodings = face_recognition.face_encodings(img_array, locations)

        for enc, loc in zip(encodings, locations):
            all_encodings.append(enc)
            all_files.append(str(img_path))
            all_locations.append(loc)

        if encodings:
            print(f"  найдено лиц: {len(encodings)}")

    except Exception as e:
        print("error:", img_path.name, e)

print(f"\nВсего лиц найдено: {len(all_encodings)}")

# сохраняем
with open(encodings_path, "wb") as f:
    pickle.dump({
        "encodings": all_encodings,
        "files": all_files,
        "locations": all_locations
    }, f)

print("Сохранено в faces.pkl")