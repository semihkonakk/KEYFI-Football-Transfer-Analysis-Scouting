import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import ImageDraw
import re


# --------------------------------------------------
# FORMASYON KOORDİNATLARI (MINIMAL & DENGELİ)
# --------------------------------------------------
POSITION_COORDS = {
    "Goalkeeper": [(50, 10)],
    "Defender": [(25, 30), (40, 30), (60, 30), (75, 30)],
    "Midfield": [(30, 52), (50, 55), (70, 52)],
    "Attack": [(30, 75), (50, 78), (70, 75)]
}

# --------------------------------------------------
# IDEAL 11 SEÇİMİ
# --------------------------------------------------
FORMATIONS = {
    "4-3-3": {
        "Goalkeeper": 1,
        "Defender": 4,
        "Midfield": 3,
        "Attack": 3
    },
    "4-2-3-1": {
        "Goalkeeper": 1,
        "Defender": 4,
        "Midfield": 5,
        "Attack": 1
    }
}

def line_up(df, score_col="ensemble_score", formation_name="4-3-3"):

    formation = FORMATIONS[formation_name]
    squad = []

    for pos, n in formation.items():
        top_players = (
            df[df["main_position"] == pos]
            .sort_values(score_col, ascending=False)
            .head(n)
        )
        squad.append(top_players)

    ideal_11 = pd.concat(squad).reset_index(drop=True)

    return ideal_11[[
        "player_name",
        "main_position",
        score_col,
        "player_image_url"
    ]]

# --------------------------------------------------
# KOORDİNAT EKLE
# --------------------------------------------------
FORMATION_COORDS = {

    "4-3-3": {
        "Goalkeeper": [(50, 12)],
        "Defender": [(17, 35), (38, 35), (62, 35), (84, 35)],
        "Midfield": [(25, 60), (50, 55), (75, 60)],
        "Attack": [(25, 82), (50, 85), (75, 82)]
    },

    "4-2-3-1": {
        "Goalkeeper": [(50, 12)],
        "Defender": [(17, 30), (38, 30), (62, 30), (84, 30)],
        "Midfield": [
            (30, 48), (70, 48),
            (20, 70), (50, 63), (80, 70)
        ],
        "Attack": [(50, 85)]
    }
}
def add_coordinates(df, formation_name):
    rows = []

    coords_map = FORMATION_COORDS[formation_name]

    for pos, coords in coords_map.items():
        players = df[df["main_position"] == pos].copy()

        for i, (_, row) in enumerate(players.iterrows()):
            if i >= len(coords):
                continue  # 🔒 ekstra güvenlik

            row["x"], row["y"] = coords[i]
            rows.append(row)

    return pd.DataFrame(rows)


# FUTBOL SAHASI (MINIMAL & ORANLI)

def draw_pitch():
    # 👇 daha minimal (4:8 korunuyor)
    fig, ax = plt.subplots(figsize=(3, 6))

    fig.patch.set_facecolor("#2E7D32")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Oranı KORU
    ax.set_aspect("equal")

    ax.set_facecolor("#2E7D32")
    line_color = "white"

    # Dış çizgiler
    ax.plot([5, 95], [5, 5], color=line_color, lw=1)
    ax.plot([5, 95], [95, 95], color=line_color, lw=1)
    ax.plot([5, 5], [5, 95], color=line_color, lw=1)
    ax.plot([95, 95], [5, 95], color=line_color, lw=1)

    # Orta çizgi
    ax.plot([5, 95], [50, 50], color=line_color, lw=0.8)

    # Orta yuvarlak
    ax.add_patch(plt.Circle((50, 50), 7, fill=False, color=line_color, lw=0.8))

    # Ceza sahaları
    ax.add_patch(plt.Rectangle((25, 5), 50, 15, fill=False, color=line_color, lw=0.8))
    ax.add_patch(plt.Rectangle((25, 80), 50, 15, fill=False, color=line_color, lw=0.8))

    # Kale alanları
    ax.add_patch(plt.Rectangle((37, 5), 26, 6, fill=False, color=line_color, lw=0.8))
    ax.add_patch(plt.Rectangle((37, 89), 26, 6, fill=False, color=line_color, lw=0.8))

    # Penaltı noktaları
    ax.scatter(50, 17, color=line_color, s=6)
    ax.scatter(50, 83, color=line_color, s=6)

    ax.axis("off")
    return fig, ax

# FOTOĞRAF GETİR

def get_player_image(url, zoom=0.12):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGBA")

        w, h = img.size
        size = int(min(w, h) * 1.1)

        left = (w - size) // 2
        top = (h - size) // 2
        right = left + size
        bottom = top + size

        img = img.crop((left, top, right, bottom))

        # ---- DAİRESEL MASKE ----
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        img.putalpha(mask)

        return OffsetImage(img, zoom=zoom)

    except Exception:
        return None

# IDEAL 11 ÇİZİMİ (MINIMAL & ŞIK)

def plot_ideal_11(df):
    fig, ax = draw_pitch()

    for _, row in df.iterrows():
        img = get_player_image(row["player_image_url"])

        if img:
            ab = AnnotationBbox(
                img,
                (row["x"], row["y"]),
                frameon=False
            )
            ax.add_artist(ab)

        ax.text(
            row["x"],
            row["y"]  - 7,
            format_player_name(row["player_name"]),
            ha="center",
            va="top",
            fontsize=3.5,
            color="black",
            linespacing=0.85,
            weight="medium"
        )

    return fig

def format_player_name(name):
    # Parantez ve içini kaldır
    clean_name = re.sub(r"\s*\(.*?\)", "", name).strip()

    parts = clean_name.split()

    if len(parts) == 1:
        return parts[0]

    return f"{parts[0]}\n{parts[-1]}"