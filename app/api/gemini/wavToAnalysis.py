import json
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_audio_emotion(audio_path: str):
    """
    音声ファイルを渡すと感情分析してJSONで返す
    """
    # 1. 音声ファイルをアップロード
    with open(audio_path, "rb") as f:
        audio_file = client.files.upload(
            file=f,
        )

    # 2. プロンプトで感情分類を依頼
    prompt = """
    あなたは音声感情分析器です。
    次の感情カテゴリのいずれかを返してください:
    ["happy", "sad", "angry", "fear", "hate"]

    JSON形式で返してください。形式は以下:
    {
      "emotion": <ラベル>,
      "confidence": <0〜1の数値>,
      "ths": [t, h, s]
    }

    - "ths" はLED制御用の配列
        - "t": 点滅周期 (0〜1)
        - "h": 色相 (0〜1)
        - "s": 彩度 (0〜1)
    """

    # 3. Geminiにリクエスト
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=[prompt, audio_file]
    )
    print("Gemini Response:", response.text)

    # 4. JSONとして返す
    return json.loads(response.output_text)

