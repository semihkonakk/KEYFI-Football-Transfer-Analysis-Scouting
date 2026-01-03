
# DETAYLI TRANSFER ARAMASI
def transfer_recommendation(
    df,position=None,min_budget=None,max_budget=None,
    score_col=None,value_col="market_value",risk_col="risk_score",injury_col="injury_days_per_season",
    min_age=None,max_age=None,max_risk=None,max_injury_days=60,top_n=10):

    data = df.copy()

    if position is not None:
        data = data[data["main_position"] == position]

    # Bütçe filtresi
    if max_budget is not None:
        data = data[data[value_col] <= max_budget]
        data = data[data[value_col] >= min_budget]

    # Risk filtresi
    if max_risk is not None:
        data = data[data[risk_col] <= max_risk]

    # Sakatlık filtresi
    if max_injury_days is not None:
        data = data[data[injury_col] <= max_injury_days]

    # Yaş filtresi
    if min_age is not None:
        data = data[data["age"] >= min_age]

    if max_age is not None:
        data = data[data["age"] <= max_age]

    # Skora göre sırala
    data = data.sort_values("ensemble_score", ascending=False)

    return data.head(top_n)[[
        "player_name",
        "main_position",
        "age",
        "current_club_name",
        "market_value",
        "final_score",
        "ml_pred",
        "ensemble_score",
        "risk_score",
        "injury_days_per_season"
    ]]

# F/P
def normalize_series(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def value_for_money_parametric(df,position=None,min_budget=None,max_budget=None,weights=None,min_age=16,max_age=42,top_n=10):

    data = df.copy()
    data = data[data["market_value"] >= min_budget]
    data = data[data["market_value"] <= max_budget]
    data = data[data["age"].between(min_age, max_age)]

    if position is not None:
        data = data[data["main_position"] == position]

    data["_mv_norm"] = normalize_series(data["market_value"])
    data["_inj_norm"] = normalize_series(data["injury_days_per_season"])
    data["ensemble_norm"] = normalize_series(data["ensemble_score"])
    data["risk_norm"] = normalize_series(data["risk_score"])

    data["value_score"] = (
        weights["score"] * data["ensemble_score"]
        - weights["value"] * data["_mv_norm"]
        - weights["risk"]  * data["risk_score"]
        - weights["injury"] * data["_inj_norm"]
        + weights["age"]   * (1 / (data["age"] + 1))*100).round(2)

    return data.sort_values("value_score", ascending=False).head(top_n)[[
        "player_name",
        "main_position",
        "current_club_name",
        "age",
        "market_value",
        "ensemble_score",
        "risk_score",
        "value_score",
        "player_image_url"
    ]]
