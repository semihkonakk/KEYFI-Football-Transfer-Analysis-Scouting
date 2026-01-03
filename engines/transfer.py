def transfer_recommendation(
        df, position, max_budget,min_age, max_age,
        max_risk, max_injury_days,
        score_col="ensemble_score", top_n=10):
    data = df[df["main_position"] == position].copy()

    data = data[
        (data["market_value"] <= max_budget) &
        (data["age"] >= min_age) &
        (data["age"] <= max_age) &
        (data["risk_score"] <= max_risk) &
        (data["injury_days_per_season"] <= max_injury_days)
    ]

    return data.sort_values(score_col, ascending=False).head(top_n)
