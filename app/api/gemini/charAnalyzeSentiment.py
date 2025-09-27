import json
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def char_analyze_sentiment(text, backgroundSetting="default"):
    """
    文章を渡すと感情分析してJSONで返す
    感情応対としてLEDの光らせ方も含む
    """
    prompt = f"""
    You are an AI assistant that analyzes sentiment of text and responds
    according to a character's personality described by "backgroundSetting".
    Based on the sentiment and the character's personality, suggest how a
    LED should react (blink, hue, saturation).  

    Return JSON with the following structure:
    {{
        "sentiment": one of  ["happy", "sad", "angry", "fear", "hate"],
        "score": float between 0 and 1,
        "character_reaction": "A brief description of how the character would react",
        "personality_adjustment": "A brief description of how the character's personality influences the reaction",
        "ths": [t, h, s]  // LED blinking period (t) float between 0 and 5, hue (h) float between 0 and 1, saturation (s) float between 0 and 1
    }}

    Background Setting (character personality):
    {backgroundSetting}

    Text:
    "{text}"
    """


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    print("Gemini Response:", response.text)

    return json.loads(response.text)
