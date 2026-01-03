import io
import streamlit as st
import pandas as pd
import base64
from pathlib import Path
import requests
from dotenv import load_dotenv
from pathlib import Path
import os


# engine importları
try:
    from engines.xl import line_up, add_coordinates, plot_ideal_11
    from engines.values import transfer_recommendation, value_for_money_parametric
    from engines.player import get_metric_type
    from engines.pdf import generate_player_report
    from engines.translations import (TEXT,POSITION_LABELS,FOOT_LABELS,FLAGS,
        CLUB_PROFILE_LABELS,SCOUT_METRICS,CLUB_PROFILES_FOR_FP,FORMATIONS,get_column_translations)
    from engines.gpt_engine import analyze_player_with_gpt,build_player_prompt

except ImportError as e:
    st.error(f"Import hatası: {e}")

@st.cache_data
def load_data():
    return pd.read_pickle("../pkl/app.pkl")

df = load_data()

display_df = df.copy()
display_df["foot"] = (display_df["foot"].astype(str).str.strip().str.capitalize())
display_df.insert(0,"Photo",display_df["player_image_url"])


# SESSION STATE

if "started" not in st.session_state:
    st.session_state.started = False

if "lang" not in st.session_state:
    st.session_state.lang = "en"

if "query_results" not in st.session_state:
    st.session_state.query_results = None

if "search_name" not in st.session_state:
    st.session_state.search_name = ""

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(page_title="KEYFI Scout Engine",layout="wide",initial_sidebar_state="collapsed")

def render_metric_input(cfg, key):
    if cfg["type"] == "numeric":
        return st.slider(
            cfg["label"],
            cfg["min"],
            cfg["max"],
            (cfg["min"], cfg["max"])
        )

    elif cfg["type"] == "categorical":
        return st.multiselect(
            cfg["label"],
            cfg["options"]
        )

    elif cfg["type"] == "date":
        return st.date_input(
            cfg["label"],
            (
                pd.to_datetime("2000-01-01"),
                pd.Timestamp.today()
            )
        )
def format_market_value(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except:
        return x

# ==================================================
# LANDING PAGE
# ==================================================

if not st.session_state.started:
    # Tüm Streamlit elementlerini gizle + Scroll engelle
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Scroll'u tamamen engelle - Landing page için */
        body {
            overflow: hidden !important;
            position: fixed !important;
            width: 100% !important;
            height: 100% !important;
        }

        [data-testid="stAppViewContainer"] {
            overflow: hidden !important;
            height: 100vh !important;
        }

        .main {
            overflow: hidden !important;
            height: 100vh !important;
        }

        section[data-testid="stVerticalBlock"] {
            overflow: hidden !important;
        }

        /* Tüm sayfayı viewport'a sığdır */
        .stApp {
            max-height: 100vh !important;
            overflow: hidden !important;
        }

        /* İçeriği ortala */
        [data-testid="stAppViewContainer"] > section {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 100vh !important;
            max-height: 100vh !important;
        }

        /* Buton stilini tamamen kaldır */
        .stButton > button {
            background-color: transparent !important;
            border: none !important;
            color: rgba(255, 255, 255, 0.6) !important;
            font-size: 18px !important;
            font-weight: 400 !important;
            padding: 0 !important;
            cursor: pointer !important;
            box-shadow: none !important;
            transition: color 0.3s ease !important;
        }

        .stButton > button:hover {
            background-color: transparent !important;
            border: none !important;
            color: rgba(255, 255, 255, 0.8) !important;
            box-shadow: none !important;
        }

        .stButton > button:focus {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }

        .stButton > button:active {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* Sayfa arkaplanı */
        .stApp {
            background: radial-gradient(circle at top, #1a1f2e, #0f1419) !important;
        }

        /* Dil butonu stili */
        .lang-button > button {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            font-size: 24px !important;
            min-width: 50px !important;
        }
        .lang-button > button:hover {
            background-color: rgba(255, 255, 255, 0.2) !important;
        }
        /* Slider track */
        div[data-baseweb="slider"] > div > div {
            background-color: #e5e7eb !important;  /* kirli beyaz */
        }
        /* Slider thumb */
        div[data-baseweb="slider"] span {
            background-color: #f9fafb !important;
            border: 2px solid #9ca3af !important;
        }
        /* Slider active range */
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: #d1d5db !important;
        }
        </style>
        <script>
        // JavaScript ile scroll'u engelle
        document.body.style.overflow = 'hidden';
        window.addEventListener('wheel', function(e) {
            e.preventDefault();
        }, { passive: false });

        window.addEventListener('touchmove', function(e) {
            e.preventDefault();
        }, { passive: false });
        
        </script>
    """, unsafe_allow_html=True)

    # Dil seçici - Sol üst köşe (yan yana, bayraklar büyük)
    col_left, col_center, col_right = st.columns([1.5, 15 , 1.5])
    with col_left:
        lang_col1, lang_col2 = st.columns([1, 1])
        with lang_col1:
            st.markdown('<div class="lang-button">', unsafe_allow_html=True)
            if st.button("🇹🇷", key="lang_tr", help="Türkçe"):
                st.session_state.lang = "tr"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with lang_col2:
            st.markdown('<div class="lang-button">', unsafe_allow_html=True)
            if st.button("🇬🇧", key="lang_en", help="English"):
                st.session_state.lang = "en"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    t = TEXT[st.session_state.lang]

    # Logo ve başlık - ortalanmış
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Tam ekran şeffaf buton */
    .fullscreen-enter {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: transparent;
        border: none;
        cursor: pointer;
        z-index: 999;
    }

    /* Streamlit button görünmez */
    .fullscreen-enter button {
        opacity: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([5, 10 , 5])
    with col2:
        # Logo - ortalanmış ve PNG arka plan şeffaf
        col_a, col_b, col_c = st.columns([5, 10, 1])
        with col_b:
            st.image("assets/keyfi.png", width=180, use_container_width=False)
        st.markdown(f"""
            <div style='text-align: center; margin-top: 15px;'>
                <p style='color: rgba(255, 255, 255, 0.7); font-size: 22px; margin-top: 0px; margin-bottom: 60px; font-weight: 300;'>
                    {t["subtitle"]}
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Gizli buton - sadece text gibi görünecek
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_b:
            if st.button(t["enter"]):
                st.session_state.started = True
                st.rerun()

    st.stop()

# ANA UYGULAMA

st.markdown("""
<style>
    header {
        visibility: visible !important;
    }
</style>
""", unsafe_allow_html=True)

# Dil seçici - Sidebar
with st.sidebar:
    st.markdown("### 🌍 " + TEXT[st.session_state.lang]["lang"])
    cols = st.columns(2)
    with cols[0]:
        if st.button(f"{FLAGS['tr']}", key="sidebar_lang_tr", use_container_width=True):
            st.session_state.lang = "tr"
            st.rerun()
    with cols[1]:
        if st.button(f"{FLAGS['en']}", key="sidebar_lang_en", use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()

t = TEXT[st.session_state.lang]
lang = st.session_state.lang

# -------------------------------
# SIDEBAR MENÜ
# -------------------------------
action = st.sidebar.radio(
    t["what"],
    [
        t["search"],
        t["Query"],
        t["top10"],
        t["fp"],
        t["pdf"],
        t["ideal11"]
    ])

# ===============================
# PLAYER SEARCH PAGE
# ===============================

if action == t["search"]:

    st.set_page_config(layout="wide")

    # HEADER
    col1, col2 = st.columns([2, 18])
    with col1:
        st.image("assets/keyfi.png", width=140)
    with col2:
        st.title(t["player_search_title"])

    st.divider()

    # SEARCH INPUT
    search_name = st.text_input(
        f"🔎 {t['search']}",
        placeholder=t["search_placeholder"]
    )

    if search_name:
        results = df[
            df["player_name"]
            .str.contains(search_name, case=False, na=False)
        ]
    else:
        results = df.copy()

    st.subheader(f"📋 {t['results']} ({len(results)})")

    if results.empty:
        st.warning(t["no_player_found"])
        st.stop()

    # RESULT TABLE
    display_df = results.copy()
    display_df.insert(0, t["photo"], display_df["player_image_url"])

    display_df["main_position"] = display_df["main_position"].apply(
        lambda x: POSITION_LABELS.get(x, {}).get(lang, x)
    )

    col_trans = get_column_translations(lang)
    col_trans_without_image = {
        k: v for k, v in col_trans.items()
        if k != "player_image_url"
    }

    display_df_renamed = display_df.rename(columns=col_trans_without_image)

    st.dataframe(
        display_df_renamed[
            [
                t["photo"],
                col_trans["player_name"],
                col_trans["main_position"],
                col_trans["age"],
                col_trans["market_value"]
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            t["photo"]: st.column_config.ImageColumn("", width="small")
        }
    )

    # PLAYER PROFILE
    st.divider()
    st.subheader(f"👤 {t['player_profile']}")

    profile_options = (
        results[["player_id", "player_name"]]
        .drop_duplicates("player_id")
        .copy()
    )

    selected_player_id = st.selectbox(
        t["select_player"],
        options=profile_options["player_id"].tolist(),
        format_func=lambda pid: (
            profile_options.loc[
                profile_options.player_id == pid, "player_name"
            ].values[0]
        )
    )

    player = df[df["player_id"] == selected_player_id].iloc[0]

    st.divider()

    # PLAYER DETAIL + ANALYZE BUTTON
    c1, c2 = st.columns([1, 3])

    with c1:
        # FOTOĞRAF
        st.image(player["player_image_url"], width=160)

        # 🔽 ANALİZ BUTONU (EN ALTA)
        st.markdown("<div style='margin-top: 30px'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button(
            "🤖 " + t["analyze_player_btn"],
            use_container_width=True
        )

    with c2:
        st.markdown(f"## {player['player_name']}")
        pos_raw = player["main_position"]
        pos_label = POSITION_LABELS.get(pos_raw, {}).get(lang, pos_raw)
        st.write(f"**{t['positionn']}:** {pos_label}")
        st.write(f"**{t['age']}:** {player['age']}")
        st.write(f"**{t['market_value']}:** €{player['market_value']:,.0f}")
        st.write(f"**{t['club']}:** {player.get('current_club_name', '-')}")

    # ANALYSIS CALL
    if analyze_clicked:
        with st.spinner(t["analyzing"]):
            try:
                player_payload = {
                    "player_name": player["player_name"],
                    "main_position": player["main_position"],
                    "age": int(player["age"]),
                    "market_value": int(player["market_value"]),
                    "final_score": float(player["final_score"]),
                    "risk_score": float(player["risk_score"])
                }

                response = requests.post(
                    "http://127.0.0.1:8000/analyze-player",
                    json={
                        "player": player_payload,
                        "language": "Turkish" if lang == "tr" else "English"
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    analysis = response.json().get("analysis", "")
                    if analysis:
                        st.markdown(analysis)
                    else:
                        st.warning("Empty analysis result.")
                else:
                    st.error(t["analysis_error"])

            except requests.exceptions.RequestException:
                st.error(t["analysis_error"])

# ===============================
# DETAILED PLAYER QUERY
# ===============================

if action == t["Query"]:

    st.set_page_config(layout="wide")

    if "query_results" not in st.session_state:
        st.session_state.query_results = None

    # HEADER
    col1, col2 = st.columns([2, 18])
    with col1:
        st.image("assets/keyfi.png", width=140)
    with col2:
        st.title(t["query_title"])

    st.divider()

    # METRIC SELECTION
    lang = st.session_state.lang

    metric_options = {
        cfg["label"][lang]: key
        for key, cfg in SCOUT_METRICS.items()
    }

    selected_metric_labels = st.multiselect(
        f"🎯 {t['select_criteria']}",
        list(metric_options.keys())
    )

    selected_metrics = [metric_options[label] for label in selected_metric_labels]

    # METRIC INPUTS (AUTO 2x GRID)
    user_inputs = {}

    if selected_metrics:
        st.subheader(f"⚙️ {t['criteria_details']}")
        cols = st.columns(2)
        col_idx = 0
        for key in selected_metrics:
            cfg = SCOUT_METRICS[key]
            label = cfg["label"][lang]
            with cols[col_idx]:
                # NUMERIC
                if cfg["type"] == "numeric":
                    user_inputs[key] = st.slider(
                        label,
                        cfg["min"],
                        cfg["max"],
                        (cfg["min"], cfg["max"])
                    )
                # CATEGORICAL
                elif cfg["type"] == "categorical":
                    if key == "main_position":
                        user_inputs[key] = st.multiselect(
                            label,
                            cfg["options"],
                            format_func=lambda x: POSITION_LABELS.get(x, {}).get(lang, x)
                        )
                    elif key == "foot":
                        foot_options = cfg["options"]
                        foot_ui_options = [
                            FOOT_LABELS[v.lower()][lang] for v in foot_options
                        ]
                        selected_ui = st.multiselect(label, foot_ui_options)
                        reverse_map = {
                            FOOT_LABELS[v.lower()][lang]: v.lower()
                            for v in foot_options
                        }

                        user_inputs[key] = [reverse_map[v] for v in selected_ui]
                # DATE
                elif cfg["type"] == "date":
                    user_inputs[key] = st.date_input(
                        label,
                        (pd.to_datetime("2000-01-01"), pd.Timestamp.today())
                    )
            col_idx += 1
            if col_idx == 2:
                cols = st.columns(2)
                col_idx = 0
    st.divider()

    # QUERY BUTTON
    if st.button(f"🎯 {t['get_players']}"):

        filtered_df = df.copy()

        for key, value in user_inputs.items():

            if SCOUT_METRICS[key]["type"] == "numeric":
                filtered_df = filtered_df[
                    (filtered_df[key] >= value[0]) &
                    (filtered_df[key] <= value[1])
                ]

            elif SCOUT_METRICS[key]["type"] == "categorical" and value:
                filtered_df = filtered_df[
                    filtered_df[key].isin(value)
                ]

            elif SCOUT_METRICS[key]["type"] == "date":
                filtered_df[key] = pd.to_datetime(filtered_df[key])
                filtered_df = filtered_df[
                    (filtered_df[key] >= pd.to_datetime(value[0])) &
                    (filtered_df[key] <= pd.to_datetime(value[1]))
                ]

        st.session_state.query_results = filtered_df

    # ===============================
    # RESULTS
    # ===============================
    filtered_df = st.session_state.query_results

    if filtered_df is not None:

        base_cols = ["player_name", "main_position", "age", "market_value"]
        result_cols = list(dict.fromkeys(base_cols + selected_metrics))

        st.subheader(f"📋 {t['results']} ({len(filtered_df)})")

        display_df = filtered_df.copy()

        # Önce fotoğrafı ekle (rename'den ÖNCE)
        if "player_image_url" in display_df.columns:
            display_df.insert(0, t["photo"], display_df["player_image_url"])

        # Kolon çevirisi (player_image_url HARİÇ)
        col_trans = get_column_translations(lang)
        col_trans_without_image = {k: v for k, v in col_trans.items() if k != "player_image_url"}

        display_df_renamed = display_df.rename(columns=col_trans_without_image)

        # Çevrilmiş kolon isimlerini oluştur
        translated_result_cols = [col_trans.get(col, col) for col in result_cols]

        if t["photo"] in display_df_renamed.columns:
            st.dataframe(
                display_df_renamed[[t["photo"]] + translated_result_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    t["photo"]: st.column_config.ImageColumn("", width="small")
                }
            )
        else:
            st.dataframe(
                display_df_renamed[translated_result_cols],
                use_container_width=True,
                hide_index=True
            )

# -------------------------------
# TOP 10 BY POSITION
# -------------------------------

if action == t["top10"]:

    st.set_page_config(layout="wide")

    # DATA
    required_cols = [
        "player_name",
        "main_position",
        "age",
        "market_value",
        "ensemble_score"
    ]

    if not all(col in df.columns for col in required_cols):
        st.error("Gerekli sütunlardan biri DataFrame'de yok!")
        st.stop()

    # SIDEBAR – FILTERS
    st.sidebar.header(f"🔍 {t['filters']}")

    position_keys = sorted(df["main_position"].dropna().unique())

    position_ui_options = [t["all"]] + [
        POSITION_LABELS.get(pos, {}).get(lang, pos)
        for pos in position_keys
    ]

    selected_position_ui = st.sidebar.selectbox(
        t["position"],
        position_ui_options
    )

    reverse_position_map = {
        POSITION_LABELS.get(pos, {}).get(lang, pos): pos
        for pos in position_keys
    }
    min_age, max_age = st.sidebar.slider(
        t["age_range"],
        16, 42,
        (16, 42)
    )
    min_budget, max_budget = st.sidebar.slider(
        t["budget_range"],
        100_000,
        500_000_000,
        (100_000, 500_000_000),
        step=100_000
    )

    top_n = st.sidebar.selectbox(
        t["top_n"],
        [5, 10, 15, 20],
        index=1
    )

    # FILTERING
    filtered_df = df.copy()

    if selected_position_ui != t["all"]:
        selected_position = reverse_position_map[selected_position_ui]
        filtered_df = filtered_df[
            filtered_df["main_position"] == selected_position
            ]

    filtered_df = filtered_df[
        (filtered_df["age"] >= min_age) &
        (filtered_df["age"] <= max_age) &
        (filtered_df["market_value"] >= min_budget) &
        (filtered_df["market_value"] <= max_budget)
        ]

    result_df = (
        filtered_df
        .sort_values("final_score", ascending=False)
        .head(top_n)
    )

    # COLUMN SELECTION
    st.sidebar.header(f"📊 {t['table_columns']}")

    default_columns = [
        "player_name",
        "main_position",
        "age",
        "market_value",
        "ensemble_score"
    ]

    optional_columns = [
        col for col in result_df.columns
        if col not in default_columns
    ]

    col_trans = get_column_translations(lang)
    optional_columns_translated = {
        col_trans.get(col, col): col
        for col in optional_columns
        if col != "player_image_url"
    }

    extra_columns_ui = st.sidebar.multiselect(
        t["add_column"],
        list(optional_columns_translated.keys())
    )

    extra_columns = [optional_columns_translated[col] for col in extra_columns_ui]
    display_columns = default_columns + extra_columns
    display_columns = [col for col in display_columns if col in result_df.columns]

    # RESULTS
    if result_df.empty:
        st.warning(t["no_match"])
    else:
        col1, col2 = st.columns([2.5, 20])
        with col1:
            st.image("assets/keyfi.png", width=140)

        with col2:
            st.title(t["transfer_system_title"])

        display_df = result_df.copy()

        if "player_image_url" in display_df.columns:
            display_df.insert(0, t["photo"], display_df["player_image_url"])

        display_df["main_position"] = display_df["main_position"].apply(
            lambda x: POSITION_LABELS.get(x, {}).get(lang, x)
        )

        col_trans_without_image = {k: v for k, v in col_trans.items() if k != "player_image_url"}

        display_df_renamed = display_df.rename(columns=col_trans_without_image)

        if "player_image_url" in display_df_renamed.columns:
            display_df_renamed = display_df_renamed.drop(columns=["player_image_url"])

        translated_display_columns = [col_trans.get(col, col) for col in display_columns]

        st.dataframe(
            display_df_renamed[[t["photo"]] + translated_display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                t["photo"]: st.column_config.ImageColumn("", width="small")
            }
        )

    # DOWNLOAD SECTION
    st.divider()
    st.subheader(f"📄 {t['export']}")

    file_type = st.radio(
        t["download_format"],
        ["CSV", "Excel (XLSX)"],
        horizontal=True
    )

    if not result_df.empty:
        export_df = result_df.copy()

        export_df["main_position"] = export_df["main_position"].apply(
            lambda x: POSITION_LABELS.get(x, {}).get(lang, x)
        )

        col_trans = get_column_translations(lang)
        export_df_renamed = export_df.rename(columns=col_trans)

        if col_trans["player_image_url"] in export_df_renamed.columns:
            export_df_renamed = export_df_renamed.drop(columns=[col_trans["player_image_url"]])

        translated_display_columns = [col_trans.get(col, col) for col in display_columns]
        export_df_final = export_df_renamed[translated_display_columns]

        if file_type == "CSV":
            csv = export_df_final.to_csv(index=False).encode("utf-8")

            st.download_button(
                label=f"📥 {t['download']}",
                data=csv,
                file_name="keyfi_transfer_raporu.csv",
                mime="text/csv"
            )
        else:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                export_df_final.to_excel(
                    writer,
                    index=False,
                    sheet_name="Transfer Raporu"
                )

            st.download_button(
                label=f"📥 {t['download']}",
                data=buffer.getvalue(),
                file_name="keyfi_transfer_raporu.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# -------------------------------
# BEST VALUE PLAYERS (F/P)
# -------------------------------

if action == t["fp"]:

    col1, col2 = st.columns([2, 20])
    with col1:
        st.image("assets/keyfi.png", width=140)
    with col2:
        st.header(t["fp_title"])

    # SIDEBAR – F/P FILTERS
    st.sidebar.header(f"⚙️ {t['fp_filters']}")

    lang = st.session_state.lang

    position_values = sorted(df["main_position"].dropna().unique())

    position_options = [t["all"]] + position_values

    position = st.sidebar.selectbox(
        t["position"],
        position_options,
        format_func=lambda x: (
            t["all"] if x == t["all"]
            else POSITION_LABELS.get(x, {}).get(lang, x)
        )
    )

    profile_key = st.sidebar.selectbox(
        t["club_strategy"],
        list(CLUB_PROFILES_FOR_FP.keys()),
        format_func=lambda x: CLUB_PROFILE_LABELS[x][st.session_state.lang]
    )
    min_budget, max_budget = st.sidebar.slider(
        t.get("budget_range", "Budget Range (€)"),
        min_value=0,
        max_value=500_000_000,
        value=(0, 20_000_000),
        step=100_000
    )

    top_n = st.sidebar.selectbox(
        t["top_n_players"],
        [5, 10, 15, 20],
        index=1
    )

    weights = CLUB_PROFILES_FOR_FP[profile_key]

    # F/P CALCULATION
    fp_df = value_for_money_parametric(
        df=df,
        position=None if position == t["all"] else position,
        min_budget=min_budget,
        max_budget=max_budget,
        weights=weights,
        top_n=top_n
    )

    # RESULTS
    st.subheader(f"💰 {t['best_value_players']}")

    if fp_df.empty:
        st.warning(t.get("no_player_found", "No players found."))
    else:
        display_df = fp_df.copy()

        # Önce fotoğrafı ekle
        if "player_image_url" in display_df.columns:
            display_df.insert(0, t["photo"], display_df["player_image_url"])

        col_trans = get_column_translations(lang)
        col_trans_without_image = {k: v for k, v in col_trans.items() if k != "player_image_url"}

        display_df_renamed = display_df.rename(columns=col_trans_without_image)

        if "player_image_url" in display_df_renamed.columns:
            display_df_renamed = display_df_renamed.drop(columns=["player_image_url"])

        st.dataframe(
            display_df_renamed,
            use_container_width=True,
            hide_index=True,
            column_config={
                t["photo"]: st.column_config.ImageColumn("", width="small")
            }
        )

    # DOWNLOAD
    st.divider()
    st.subheader(f"📄 {t['export']}")

    file_type = st.radio(
        t["download_format"],
        ["CSV", "Excel (XLSX)"],
        horizontal=True
    )

    if not fp_df.empty:
        export_df = fp_df.copy()
        col_trans = get_column_translations(lang)
        export_df_renamed = export_df.rename(columns=col_trans)

        if col_trans["player_image_url"] in export_df_renamed.columns:
            export_df_renamed = export_df_renamed.drop(columns=[col_trans["player_image_url"]])

        if file_type == "CSV":
            csv = export_df_renamed.to_csv(index=False).encode("utf-8")

            st.download_button(
                label=f"📥 {t['download_csv']}",
                data=csv,
                file_name="keyfi_best_value_players.csv",
                mime="text/csv"
            )
        else:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                export_df_renamed.to_excel(
                    writer,
                    index=False,
                    sheet_name="Best Value Players"
                )

            st.download_button(
                label=f"📥 {t['download_excel']}",
                data=buffer.getvalue(),
                file_name="keyfi_best_value_players.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# -------------------------------
# SCOUT PDF RAPORU
# -------------------------------

if action == t["pdf"]:

    st.set_page_config(layout="wide")

    col1, col2 = st.columns([2, 20])
    with col1:
        st.image("assets/keyfi.png", width=140)
    with col2:
        st.header(t["pdf_title"])

    # PLAYER SELECT
    player_options = (
        df[["player_id", "player_name", "player_image_url"]]
        .drop_duplicates("player_id")
        .sort_values("player_name")
    )

    selected_player_id = st.selectbox(
        t["select_player"],
        options=player_options["player_id"].tolist(),
        format_func=lambda pid: player_options.loc[
            player_options.player_id == pid, "player_name"
        ].values[0]
    )

    player_row = df[df["player_id"] == selected_player_id].iloc[0]

    # SCOUT COMMENT
    scout_comment = st.text_area(
        t["scout_comment"],
        placeholder=t["scout_placeholder"]
    )

    player_row = player_row.copy()
    player_row["scout_comment"] = scout_comment

    # GENERATE PDF
    if st.button(t["generate_pdf"]):

        file_name = f"scout_report_{player_row['player_name']}.pdf"

        generate_player_report(
            row=player_row,
            file_name=file_name
        )

        with open(file_name, "rb") as f:
            pdf_bytes = f.read()

        st.success(t["pdf_ready"])

        st.download_button(
            label=t["download_pdf"],
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf"
        )

# -------------------------------
# XL
# -------------------------------

if action == t["ideal11"]:

    st.sidebar.header(f"⚙️ {t['ideal11_settings']}")

    # SIDEBAR CONTROLS
    formation_name = st.sidebar.selectbox(
        t["formation"],
        list(FORMATIONS.keys())
    )

    score_type = st.sidebar.selectbox(
        t["build_based_on"],
        [t["performance_score"], t["value_fp"]]
    )

    # SCORE SOURCE
    if score_type == t["value_fp"]:
        profile_key = st.sidebar.selectbox(
            t["recruitment_profile"],
            options=list(CLUB_PROFILES_FOR_FP.keys()),
            format_func=lambda k: CLUB_PROFILE_LABELS[k][st.session_state.lang]
        )

        weights = CLUB_PROFILES_FOR_FP[profile_key]

        fp_full_df = value_for_money_parametric(
            df=df,
            min_budget=df["market_value"].min(),
            max_budget=df["market_value"].max(),
            weights=weights,
            top_n=len(df)
        )

        base_df = df.merge(
            fp_full_df[["player_name", "value_score"]],
            on="player_name",
            how="left"
        )

        score_col = "value_score"

    else:
        base_df = df.copy()
        score_col = "ensemble_score"

    # BUILD IDEAL 11
    ideal_11_df = line_up(
        base_df,
        score_col=score_col,
        formation_name=formation_name
    )

    ideal_11_df = add_coordinates(ideal_11_df, formation_name)

    # VISUAL
    col1, col2 = st.columns([2, 20])

    with col1:
        st.image("assets/keyfi.png", width=500)

    with col2:
        st.title(f"{t['ideal11_title']} – {formation_name}")

    fig = plot_ideal_11(ideal_11_df)
    st.pyplot(fig, clear_figure=True)

    # TABLE
    with st.expander(f"📋 {t['squad_table']}"):
        # Kolon çevirisi
        col_trans = get_column_translations(lang)

        # Score kolonunu çevir
        score_col_name = t["score"]

        ideal_11_display = ideal_11_df[["player_name", "main_position", score_col]].copy()
        ideal_11_display_renamed = ideal_11_display.rename(columns={
            **col_trans,
            score_col: score_col_name
        })

        st.dataframe(
            ideal_11_display_renamed,
            use_container_width=True,
            hide_index=True
        )


