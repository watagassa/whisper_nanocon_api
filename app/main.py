# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
from shutil import copyfile
from pydub import AudioSegment
from app.api.gemini.entAnalysis import analyze_sentiment
from app.services.audio_preprocess import preprocess_audio

app = FastAPI()
model = whisper.load_model("medium")  # "base" → "medium"

# 保存先ディレクトリ
SOUNDS_DIR = "sounds"
os.makedirs(SOUNDS_DIR, exist_ok=True)

# CORS許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 一時ファイル (mp4) に保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # wav への変換先を作る
    wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"

    try:
        # --- mp4 → wav 変換 ---
        audio = AudioSegment.from_file(tmp_path)
        audio.export(wav_path, format="wav")

        # --- 前処理 (ノイズ除去・正規化) ---
        preprocess_audio(wav_path, wav_path)

        # --- Whisper で文字起こし ---
        result = model.transcribe(wav_path, language="ja", fp16=False)

        # --- 感情分析 ---
        analyze = analyze_sentiment(result["text"])

        # --- sounds フォルダに保存 ---
        mp4_filename = os.path.join(SOUNDS_DIR, os.path.basename(tmp_path))
        wav_filename = os.path.join(SOUNDS_DIR, os.path.basename(wav_path))
        copyfile(tmp_path, mp4_filename)
        copyfile(wav_path, wav_filename)

        return {
            "text": result["text"],
            "analyze": analyze,
            "saved_mp4": mp4_filename,
            "saved_wav": wav_filename
        }

    finally:
        # 一時ファイル削除
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
