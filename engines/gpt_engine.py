import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Çalıştırma komutu: uvicorn app:app --reload

# --- ENV ---
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
print("API KEY VAR MI:", bool(api_key))

if not api_key:
    raise RuntimeError("OPENAI_API_KEY bulunamadı.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    print("USING OPENAI KEY:", api_key[:20])
    return OpenAI(api_key=api_key)

# --- PROMPT BUILDER ---
def build_player_prompt(player: dict) -> str:
    return f"""
Player Name: {player.get('player_name')}
Position: {player.get('main_position')}
Age: {player.get('age')}
Market Value: {player.get('market_value')} €
Final Score: {player.get('final_score')}
Risk Score: {player.get('risk_score')}
"""

# --- GPT ANALYSIS ---
def analyze_player_with_gpt(player: dict, language: str = "English") -> str:
    try:
        client = get_openai_client()  # 🔥 HER ÇAĞRIDA TAZE CLIENT

        prompt = build_player_prompt(player)

        system_prompt = (
            "You are a professional football scout and transfer analyst."
            if language == "English"
            else "Sen profesyonel bir futbol scout ve transfer analistisin."
        )

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=400
        )

        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        return response.output[0]["content"][0]["text"]

    except Exception as e:
        return f"GPT ERROR: {str(e)}"


