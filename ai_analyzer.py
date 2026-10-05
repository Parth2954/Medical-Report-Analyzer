import os
import json
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Gemini model
MODEL_NAME = "gemini-3.5-flash"


def analyze_report(report_text):

    prompt = f"""
You are an expert medical AI assistant.

Analyze the following medical report.

Return ONLY valid JSON.
Do not use Markdown.
Do not use ```json or ```.

Use this exact structure:

{{
    "health_score": 92,
    "summary": "",
    "abnormal_values": [
        {{
            "name": "",
            "status": "",
            "reason": ""
        }}
    ],
    "recommendations": [
        "",
        "",
        ""
    ]
}}

Rules:

1. health_score must be a number from 0 to 100.
2. summary must briefly explain the overall report.
3. abnormal_values must contain only values that appear abnormal or potentially concerning.
4. If there are no abnormal values, return an empty array.
5. recommendations should contain practical general recommendations.
6. Do not invent laboratory values that are not present in the report.
7. Do not diagnose diseases.
8. Clearly state that medical professionals should be consulted for diagnosis or treatment.
9. Return valid JSON only.

Medical Report:

{report_text}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        clean_text = response.text.strip()

        # Remove Markdown code fences if returned
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]

        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]

        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        clean_text = clean_text.strip()

        print("========== CLEAN JSON ==========")
        print(clean_text)
        print("================================")

        # Convert JSON string into Python dictionary
        result = json.loads(clean_text)

        return result

    except json.JSONDecodeError as e:
        print("Gemini returned invalid JSON:")
        print(clean_text)
        raise ValueError(f"Invalid JSON returned by Gemini: {e}")

    except Exception as e:
        print(f"Gemini API error: {e}")
        raise