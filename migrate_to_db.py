import torch
import psycopg2
from pathlib import Path

index_path = Path("index.pt")

print("Загружаем index.pt...")
saved = torch.load(index_path, map_location="cpu")
embeddings = saved["embeddings"]
files = saved["files"]

print(f"Найдено {len(files)} фото для переноса")

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="postgres"
)
cur = conn.cursor()

print("Очищаем старые данные...")
cur.execute("DELETE FROM photos")

print("Переносим...")
for filename, emb in zip(files, embeddings):
    emb_list = emb.tolist()
    cur.execute(
        "INSERT INTO photos (filename, embedding) VALUES (%s, %s)",
        (str(filename), emb_list)
    )

conn.commit()
cur.close()
conn.close()

print("Готово! Все эмбеддинги в базе.")