import json
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_sentiment(text):
    """
    文章を渡すと感情分析してJSONで返す
    """
    prompt = f"""
    Analyze the sentiment of the following text and return JSON.
    Use the following mapping between sentiment and color (hue, saturation):
    - happy: yellow  (h≈0.15, s≈1.0)
    - sad: blue      (h≈0.58, s≈1.0)
    - angry: red     (h≈0.0,  s≈1.0)
    - fear: purple   (h≈0.75, s≈1.0)
    - love: pink  (h≈0.94,  s≈0.6)
    The JSON must contain:
    - "sentiment" : one of  ["happy", "sad", "angry", "fear", "love"]
        
    - "score" : a float between 0 and 1, representing the intensity of the sentiment
    - "ths" : an array [t, h, s]
        - "t" : LED blinking period, When emotions are strong, the value is low; when they are weak, the value is high. (float, between 0.1 and 2) 
        - "h" : hue (float, between 0 and 1)
        - "s" : saturation (float, between 0 and 1)

    Text: "{text}"
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    print("Gemini Response:", response.text)

    return json.loads(response.text)  # JSONをPythonのdictに変換して返す
