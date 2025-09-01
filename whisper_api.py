from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper

app = FastAPI()
model = whisper.load_model("base")

# CORS許可（Reactからアクセスするため）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では特定のドメインに限定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with open(file.filename, "wb") as f:
        f.write(await file.read())

    result = model.transcribe(file.filename, language="ja", fp16=False)
    return {"text": result["text"]}
