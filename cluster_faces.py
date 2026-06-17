from pathlib import Path
import pickle
import numpy as np
from sklearn.cluster import DBSCAN

with open("faces.pkl", "rb") as f:
    data = pickle.load(f)

encodings = np.array(data["encodings"])
files = data["files"]

print(f"Кластеризуем {len(encodings)} лиц...")

# DBSCAN группирует похожие лица, -1 = не вошло ни в один кластер
clustering = DBSCAN(eps=0.58, min_samples=2, metric="euclidean")
labels = clustering.fit_predict(encodings)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Найдено кластеров (людей): {n_clusters}")
print(f"Лиц без кластера: {list(labels).count(-1)}")

# сохраняем
with open("faces.pkl", "wb") as f:
    pickle.dump({
        "encodings": data["encodings"],
        "files": data["files"],
        "locations": data["locations"],
        "labels": labels.tolist()
    }, f)

print("Готово!")