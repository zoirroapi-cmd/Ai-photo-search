import torch
import open_clip
import psycopg2
from deep_translator import GoogleTranslator

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Загрузка модели...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device)
print("Модель загружена\n")

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="postgres"
)
cur = conn.cursor()

while True:
    query = input("Search: ").strip()

    if not query:
        continue

    if query.lower() == "exit":
        break

    if any("\u0400" <= c <= "\u04FF" for c in query):
        query = GoogleTranslator(source="ru", target="en").translate(query)
        print("Перевод:", query)

    text_query = f"a photo of {query}, high quality, detailed"

    with torch.no_grad():
        text = open_clip.tokenize([text_query]).to(device)
        text_emb = model.encode_text(text)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    emb_list = text_emb.squeeze(0).cpu().tolist()

    # pgvector: оператор <=> это косинусное расстояние, чем меньше тем похожее
    cur.execute(
        """
        SELECT filename, 1 - (embedding <=> %s::vector) AS score
        FROM photos
        ORDER BY embedding <=> %s::vector
        LIMIT 10
        """,
        (emb_list, emb_list)
    )

    results = cur.fetchall()

    print("\nРезультаты:")
    for filename, score in results:
        print(f"  {filename}  score: {score:.3f}")

    print()