# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000



from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
from app.api.gemini.entAnalysis import analyze_sentiment

app = FastAPI()
model = whisper.load_model("base")

# CORS許可（React Native からアクセスするため）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では限定したほうが良い
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Whisper で文字起こし
        result = model.transcribe(tmp_path, language="ja", fp16=False)
        analyze = analyze_sentiment(result["text"])
        return {"text": result["text"], "analyze": analyze}
    finally:
        # 処理後に削除
        os.remove(tmp_path)
