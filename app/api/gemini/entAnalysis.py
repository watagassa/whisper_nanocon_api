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
    The JSON must contain:
    - "sentiment" : one of ["positive","neutral","negative"]
    - "score" : a float between 0 and 1
    - "hsv" : an array [h, s, v], each value between 0 and 1

    Text: "{text}"
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    return json.loads(response.text)  # JSONをPythonのdictに変換して返す
