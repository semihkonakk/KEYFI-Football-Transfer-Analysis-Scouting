from fastapi import FastAPI
from engines.gpt_engine import analyze_player_with_gpt

app = FastAPI()

@app.get("/")
def root():
    return {"status": "KEYFI API running"}

@app.post("/analyze-player")
def analyze_player(payload: dict):
    player = payload["player"]
    language = payload.get("language", "English")

    analysis = analyze_player_with_gpt(player, language)
    return {"analysis": analysis}
