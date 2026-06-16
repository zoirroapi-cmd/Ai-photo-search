from pathlib import Path
import torch
import open_clip
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

device = "mps" if torch.backends.mps.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device)

saved = torch.load(index_path, map_location="cpu")
all_embeddings = saved["embeddings"].to(device)
files = [Path(f) for f in saved["files"]]

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
        <input id="query" placeholder="ребёнок, дерево, праздник..." />
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
    query = q.strip()

    if any("\u0400" <= c <= "\u04FF" for c in query):
        query = GoogleTranslator(source="ru", target="en").translate(query)

    query = f"a photo of {query}, high quality, detailed"

    with torch.no_grad():
        text = open_clip.tokenize([query]).to(device)
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