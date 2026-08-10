import os
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from fastapi import HTTPException

from app.system_prompt import SYSTEM_PROMPT

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
# Pinned model id. If Google retires it, calls fail with a 502 "Gemini is
# not reachable" — re-pin from https://ai.google.dev/gemini-api/docs/models
# (Google emails a ~2-week deprecation notice before removing a model).
GEMINI_MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=GEMINI_API_KEY)
generate_config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)


def call_gemini(question: str) -> str:
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=question,
            config=generate_config,
        )
        return response.text
    except genai_errors.APIError:
        raise HTTPException(status_code=502, detail="Gemini is not reachable.")
