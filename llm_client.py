"""Small wrapper around the OpenAI API: send a question, get the answer back."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT", "You are a helpful assistant. Answer clearly and concisely."
)

_client = OpenAI()  # reads OPENAI_API_KEY from the environment


def ask_llm(question: str) -> str:
    """Send one question to OpenAI and return the text answer."""
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()
