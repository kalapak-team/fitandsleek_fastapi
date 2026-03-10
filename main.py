import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import Optional  # បន្ថែមសម្រាប់ការកំណត់ Type Hint ឱ្យត្រូវជាមួយ Python 3.9

# សេវាកម្ម Embedding រូបភាពដោយប្រើ CLIP Model
app = FastAPI()

# អនុញ្ញាតឱ្យ Laravel តភ្ជាប់មកកាន់ AI Service នេះតាមរយៈ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ទាញយក API Key ពី Environment Variables (Hugging Face Secrets)
AI_SERVICE_KEY = os.getenv("AI_SERVICE_KEY")

# ដំឡើង Model (វានឹងទាញយកដោយស្វ័យប្រវត្តិពេល Build លើកដំបូង)
model = SentenceTransformer('clip-ViT-B-32')

@app.get("/")
async def root():
    return {"message": "AI Embedding Service is running!"}

@app.post('/embed')
async def get_embedding(
    file: UploadFile = File(...),
    # កែសម្រួលត្រង់នេះ៖ ប្រើ Optional[str] ជំនួសឱ្យ str | None ដើម្បីឱ្យដើរលើ Python 3.9
    x_api_key: Optional[str] = Header(default=None),
):
    # ពិនិត្យសុវត្ថិភាព API Key
    if AI_SERVICE_KEY and x_api_key != AI_SERVICE_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized access")

    try:
        # អានទិន្នន័យរូបភាព
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # បង្កើត Vector Embedding
        embedding = model.encode(image)
        
        return {'embedding': embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    # ប្រើ Port 7860 សម្រាប់ Hugging Face
    uvicorn.run(app, host='0.0.0.0', port=7860)
