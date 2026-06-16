from pathlib import Path
import torch
import open_clip
from PIL import Image
from deep_translator import GoogleTranslator

photos_dir = Path("photos")
index_path = Path("index.pt")

if not photos_dir.exists():
    print("Папка photos не найдена")
    exit()

print("Загрузка модели...")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Устройство:", device)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device)

print("Модель загружена\n")



if index_path.exists():
    print("Загружаем сохранённый индекс...")
   
    saved = torch.load(index_path, map_location="cpu")
   
    all_embeddings = saved["embeddings"].to(device)
    files = [Path(f) for f in saved["files"]]
   
    print(f"Готово! Загружено {len(files)} фото\n")

else:
    files = []
    image_embeddings = []

    print("Индексация фото...")

    for img_path in photos_dir.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        print("processing:", img_path.name)

        try:
            image = preprocess(
                Image.open(img_path).convert("RGB")
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.encode_image(image)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            image_embeddings.append(emb)
            files.append(str(img_path))

            print("indexed:", img_path.name)

        except Exception as e:
            print("skipped:", img_path.name, "—", e)

    if not files:
        print("Нет фото для поиска")
        exit()

    all_embeddings = torch.cat(image_embeddings, dim=0)

    torch.save(
        {
            "embeddings": all_embeddings.cpu(),
            "files": files
        },
        index_path
    )

    files = [Path(f) for f in files]
  
    print(f"\nГотово! Проиндексировано {len(files)} фото")
    print("Индекс сохранён\n")

while True:
    query = input("Search: ").strip()

    if not query:
        continue

    if query.lower() == "exit":
        break

    if any("\u0400" <= c <= "\u04FF" for c in query):
        query = GoogleTranslator(source="ru", target="en").translate(query)
        print("Перевод:", query)

    query = f"a photo of {query}, high quality, detailed"

    with torch.no_grad():
        text = open_clip.tokenize([query]).to(device)
   
        text_emb = model.encode_text(text)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    scores = (text_emb @ all_embeddings.T).squeeze(0)

    top_k = min(10, len(files))
    values, indices = torch.topk(scores, k=top_k)

    print("\nРезультаты:")
    
    for score, idx in zip(values.tolist(), indices.tolist()):
        print(f"  {files[idx].name}  score: {score:.3f}")

    print()