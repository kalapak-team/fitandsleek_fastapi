import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from PIL import Image

# Lightweight image embedding service using a smaller CLIP model
# Model: clip-ViT-B-32 (approx. 150-200MB). For smaller footprint, you can swap to
# a quantized variant if available in your environment.

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
AI_SERVICE_KEY = os.getenv("AI_SERVICE_KEY")
model = SentenceTransformer('clip-ViT-B-32')

@app.post('/embed')
async def get_embedding(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    if AI_SERVICE_KEY and x_api_key != AI_SERVICE_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    embedding = model.encode(image)
    return {'embedding': embedding.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
