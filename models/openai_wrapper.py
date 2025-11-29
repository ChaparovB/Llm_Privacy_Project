import os

from dotenv import load_dotenv

# ✅ Force .env load from correct path
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class GPT4oWrapper:
    def query(self, prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error] {str(e)}"
