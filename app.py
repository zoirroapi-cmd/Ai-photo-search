from pathlib import Path
import torch
import open_clip
import face_recognition
import numpy as np
import pickle
from PIL import Image
from deep_translator import GoogleTranslator
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pillow_heif
import io

pillow_heif.register_heif_opener()

app = FastAPI()
app.mount("/photos", StaticFiles(directory="photos"), name="photos")

photos_dir = Path("photos")
index_path = Path("index.pt")
people_dir = Path("people")

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- CLIP ---
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device)

saved = torch.load(index_path, map_location="cpu")
all_embeddings = saved["embeddings"].to(device)
files = [Path(f) for f in saved["files"]]

# --- Лица: образцы известных людей ---
print("Загружаем образцы лиц...")
known_encodings = []
known_names = []

for img_path in people_dir.iterdir():
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".heic"]:
        continue
    name = img_path.stem.split("_")[0].lower()
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    encodings = face_recognition.face_encodings(img_array)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)

print(f"Загружено образцов: {len(known_encodings)} ({set(known_names)})")

# --- Лица: все найденные на всех фото ---
with open("faces.pkl", "rb") as f:
    faces_data = pickle.load(f)

all_face_encodings = np.array(faces_data["encodings"])
all_face_files = faces_data["files"]

# --- Связываем: для каждого имени список файлов где оно есть ---
FACE_THRESHOLD = 0.45
person_files = {name: set() for name in set(known_names)}

for known_enc, known_name in zip(known_encodings, known_names):
    distances = face_recognition.face_distance(all_face_encodings, known_enc)
    for i, dist in enumerate(distances):
        if dist < FACE_THRESHOLD:
            person_files[known_name].add(all_face_files[i])

for name, file_set in person_files.items():
    print(f"  {name}: {len(file_set)} фото")


@app.get("/image/{filename}")
def get_image(filename: str):
    path = photos_dir / filename
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>AI Photo Search</title>
        <style>
            body { font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }
            input { width: 70%; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            .results { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
            .item { text-align: center; }
            .item img { width: 200px; height: 200px; object-fit: cover; border-radius: 8px; }
            .item p { font-size: 12px; color: #666; margin: 4px 0; }
        </style>
    </head>
    <body>
        <h1>🔍 AI Photo Search</h1>
        <input id="query" placeholder="liza улыбается, дерево, праздник..." />
        <button onclick="search()">Найти</button>
        <div class="results" id="results"></div>
        <script>
            async function search() {
                const q = document.getElementById("query").value;
                const res = await fetch("/search?q=" + encodeURIComponent(q));
                const data = await res.json();
                const div = document.getElementById("results");
                div.innerHTML = data.results.map(r => `
                    <div class="item">
                        <img src="/image/${r.name}" />
                        <p>${r.name}</p>
                        <p>score: ${r.score}</p>
                    </div>
                `).join("");
            }
            document.getElementById("query").addEventListener("keydown", e => {
                if (e.key === "Enter") search();
            });
        </script>
    </body>
    </html>
    """


@app.get("/search")
def search(q: str):
    raw_query = q.strip().lower()
    print("RAW QUERY:", raw_query)

    # сначала ищем известное имя в ОРИГИНАЛЬНОМ запросе (до перевода)
    matched_person = None
    remaining_query = raw_query

    for name in person_files.keys():
        if name in raw_query:
            matched_person = name
            remaining_query = raw_query.replace(name, "").strip()
            break

    print("MATCHED PERSON:", matched_person, "REMAINING:", remaining_query)

    # переводим только остаток запроса, если есть кириллица
    if remaining_query and any("\u0400" <= c <= "\u04FF" for c in remaining_query):
        remaining_query = GoogleTranslator(source="ru", target="en").translate(remaining_query)
        remaining_query = remaining_query.lower()
        print("TRANSLATED REMAINING:", remaining_query)

    # --- Если есть имя: фильтруем по человеку ---
    if matched_person:
        candidate_files = person_files[matched_person]
        candidate_indices = [i for i, f in enumerate(files) if str(f) in candidate_files]

        if not candidate_indices:
            return {"results": []}

        if remaining_query:
            text_query = f"a photo of {remaining_query}, high quality, detailed"
            with torch.no_grad():
                text = open_clip.tokenize([text_query]).to(device)
                text_emb = model.encode_text(text)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            candidate_embeddings = all_embeddings[candidate_indices]
            scores = (text_emb @ candidate_embeddings.T).squeeze(0)
            top_k = min(20, len(candidate_indices))
            values, top_local_idx = torch.topk(scores, k=top_k)

            results = [
                {"name": files[candidate_indices[idx]].name, "score": f"{v:.3f}"}
                for v, idx in zip(values.tolist(), top_local_idx.tolist())
            ]
        else:
            results = [{"name": files[i].name, "score": "1.000"} for i in candidate_indices]

        return {"results": results}

    # --- Без имени: обычный CLIP поиск ---
    query_for_clip = remaining_query if remaining_query else raw_query

    if any("\u0400" <= c <= "\u04FF" for c in query_for_clip):
        query_for_clip = GoogleTranslator(source="ru", target="en").translate(query_for_clip)
        query_for_clip = query_for_clip.lower()

    text_query = f"a photo of {query_for_clip}, high quality, detailed"
    with torch.no_grad():
        text = open_clip.tokenize([text_query]).to(device)
        text_emb = model.encode_text(text)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    scores = (text_emb @ all_embeddings.T).squeeze(0)
    top_k = min(20, len(files))
    values, indices = torch.topk(scores, k=top_k)

    results = [
        {"name": files[i].name, "score": f"{v:.3f}"}
        for v, i in zip(values.tolist(), indices.tolist())
    ]

    return {"results": results}