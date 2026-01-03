import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display_functions import display

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

dv = pd.read_pickle("datasets/forDV.pkl")

score_cols = [
    "player_id",
    "player_name",
    "main_position",
    "final_score",
    "performance_score",
    "age_score",
    "risk_score",
    "availability_score",
    "market_score",
    "transfer_score"
]

score_df = dv[score_cols].copy()
score_df = score_df.merge(dv[["player_id", "age"]],on="player_id",how="left")

# Histogram – Final Score Dağılımı
plt.figure()
plt.hist(dv["final_score"], bins=30)
plt.xlabel("Final Score")
plt.ylabel("Oyuncu Sayısı")
plt.title("Final Score Dağılımı")
plt.show()

# Boxplot – Outlier Kontrolü
plt.figure()
plt.boxplot(dv["final_score"])
plt.ylabel("Final Score")
plt.title("Final Score Boxplot")
plt.show()

# Yaş - skor ilişkisi
plt.figure()
plt.scatter(dv["age"], dv["final_score"])
plt.xlabel("Yaş")
plt.ylabel("Final Score")
plt.title("Yaş vs Final Score")
plt.show()

# Pozisyona Göre Top 20
def top_players_by_position(df, position, n=20):
    return (
        df[df["main_position"] == position]
        .sort_values("final_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

top20_attack = top_players_by_position(score_df, "Attack")
top20_midfield = top_players_by_position(score_df, "Midfield")
top20_defender = top_players_by_position(score_df, "Defender")
top20_goalkeeper = top_players_by_position(score_df, "Goalkeeper")

# Pozisyonların dağılımı

score_df.groupby("main_position")["final_score"].agg(
    ["mean", "median", "std", "min", "max"]
).sort_values("mean", ascending=False)

# Pozisyona Göre Top 5 – Risk / Performans Dengesi

top5_pos = (score_df.sort_values("final_score", ascending=False).groupby("main_position").head(5).
            loc[:, ["player_name","main_position","final_score","performance_score","risk_score"]]
            .sort_values(["main_position", "final_score"], ascending=[True, False]))

# Yaşa Göre En İyi Oyuncular (U23 / 23–28 / 29+)

import pandas as pd

bins = [0, 22, 28, 100]
labels = ["U23", "Prime (23-28)", "29+"]

age_group_top = (
    score_df.assign(age_group=pd.cut(score_df["age"],bins=bins,labels=labels)).
    sort_values("final_score", ascending=False).groupby("age_group").head(5)
    [["player_name", "age", "main_position", "final_score", "age_group"]])

# Risk Skoru En Düşük Top 20
score_df.sort_values("risk_score")[["player_name", "main_position", "risk_score", "final_score"]].head(20)

# Balanced
score_df["balance_score"] = (
    score_df["performance_score"]
    - score_df["risk_score"]
    + score_df["availability_score"])

(score_df.sort_values("balance_score", ascending=False)
[["player_name", "main_position", "balance_score", "final_score"]].head(20))

# Örnek scout raporları

for pos in top5_pos["main_position"].unique():
    print(f"\n🔹 {pos.upper()} – TOP 5")
    display(top5_pos[top5_pos["main_position"] == pos].drop(columns="main_position"))

scout_report = (
    score_df.loc[score_df["age"] <= 25].sort_values("final_score", ascending=False).groupby("main_position").head(5)
    .loc[:, ["player_name","main_position","final_score","performance_score","risk_score"]].
    sort_values(["main_position", "final_score"], ascending=[True, False]))


scout_report.to_excel("outputs/scout_report_top5_by_position.xlsx",index=False)

# Low Risk – High Performance

perf_threshold = dv["performance_score"].quantile(0.75)
risk_threshold = dv["risk_score"].quantile(0.25)

dv["scout_label"] = "Normal"
mask = ((dv["performance_score"] >= perf_threshold)&
        (dv["risk_score"] <= risk_threshold))

dv.loc[mask, "scout_label"] = "Low Risk / High Performance"

dv["scout_label"].value_counts()

elite_scouts = (dv.query("scout_label == 'Low Risk / High Performance'").sort_values("final_score", ascending=False)
    .loc[:, ["player_name","age","main_position","final_score","performance_score","risk_score"]])

elite_scouts.head(10)

def top_low_risk_high_perf_by_position(df, position, top_n=10):
    return (
        df[
            (df["main_position"] == position) &
            (df["scout_label"] == "Low Risk / High Performance")
        ]
        .sort_values("final_score", ascending=False)
        .loc[:, [
            "player_name",
            "age",
            "current_club_name",
            "final_score",
            "performance_score",
            "risk_score"
        ]]
        .head(top_n)
    )

top_attack = top_low_risk_high_perf_by_position(dv, "Attack")
top_midfield = top_low_risk_high_perf_by_position(dv, "Midfield")
top_defender = top_low_risk_high_perf_by_position(dv, "Defender")
top_goalkeeper = top_low_risk_high_perf_by_position(dv, "Goalkeeper")


