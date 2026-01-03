import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

final_df = pd.read_pickle("../pkl/forDV.pkl")

target = "final_score"

feature_cols = ["goals_per90", "assists_per90",
                "goal_contribution_per90","minutes_per_season",
                "injury_count", "injury_days_per_season", "injury_games_per_season",
                "market_value","on_pitch_ratio","transfer_fee", "value_at_transfer",
                "overpay","yellow_per90","clean_sheets_per90", "conceded_per90","age_score",
                "season_count"]

X = final_df[feature_cols]
y = final_df[target]

# FİNAL SKOR TAHMİNİ

# Verisetimizi Train/Test ile bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)

y_pred = lr.predict(X_test)

def regression_metrics(y_test, y_pred, verbose=True):
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    if verbose:
        for k, v in metrics.items():
            print(f"{k:<5}: {v:.4f}")

    return metrics

regression_metrics(y_test,y_pred)

# Scatter plot
plt.figure(figsize=(7, 7))
plt.scatter(
    y_test,
    y_pred,
    alpha=0.35,
    s=35,
    color="#1f77b4",   # soft blue
    edgecolor="none"
)
# Referans çizgisi (Perfect Prediction)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=2,
    color="gray",
    alpha=0.8,
    label="Perfect Prediction"
)
plt.xlabel("Final Score (Expert)", fontsize=11)
plt.ylabel("Predicted Score (ML)", fontsize=11)
plt.title("Expert Score vs ML Prediction", fontsize=14, fontweight="bold")
plt.legend(frameon=False)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# Model overfitting yapmış mı inceleyelim.
cv_r1 = cross_val_score(lr, X, y, cv=5, scoring="r2")

print("CV R2:", cv_r1)
print("CV mean:", cv_r1.mean())

# LightGBM

lgbm = LGBMRegressor(verbosity=-1,random_state=42).fit(X_train, y_train)

y_pred = lgbm.predict(X_test)

regression_metrics(y_test,y_pred)

cv_r2 = cross_val_score(lgbm,X,y,cv=5,scoring="r2")

print("CV R2:", cv_r2)
print("CV mean:", cv_r2.mean())
cv_r2.std()

# HİPERPARAMETRE OPTİMİZASYONU

lgbm.get_params()

params = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.1, 0.01, 0.001],
    "max_depth": [5, 7, -1],
    "num_leaves": [31, 63, 90],
    "colsample_bytree": [0.4,0.8,1.0]
}

lgbm_bestgird = GridSearchCV(lgbm,params,cv=5,scoring="r2",n_jobs=-1, verbose=True).fit(X, y)

lgbm_bestgird.best_params_

lgbm_final = lgbm.set_params(**lgbm_bestgird.best_params_, random_state=17).fit(X_train, y_train)

y_pred = lgbm_final.predict(X_test)

regression_metrics(y_test,y_pred)

#Cross Validation
cv_r3 = cross_val_score(lgbm_final,X,y,cv=5,scoring="r2")
print("CV R2:", cv_r3)
print("CV mean:", cv_r3.mean())

# Feature Importance
def plot_importance(model, features, top_n=10, save=False):
    feature_imp = pd.DataFrame({
        "Value": model.feature_importances_,
        "Feature": features.columns
    })

    feature_imp = (
        feature_imp
        .sort_values(by="Value", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))

    # Yumuşak ve profesyonel renk paleti
    palette = sns.color_palette("Blues_r", n_colors=top_n)

    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp,
        palette=palette
    )

    plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save:
        plt.savefig("importances.png", dpi=300)

    plt.show()

plot_importance(lgbm_final, X)

# EXPERT FINAL SCORE VS ML FINAL SCORE #

final_df["ml_pred"] = lgbm_final.predict(X)
final_df["score_diff"] = final_df["ml_pred"] - final_df["final_score"]
final_df["abs_diff"] = final_df["score_diff"].abs()


def expert_ml_gap(df, position=None,top_n=10,direction="ml_over"):
    """
    - 'ml_over'  -> ML > Expert (underrated by expert)
    - 'expert_over' -> Expert > ML (overrated by expert)
    """

    temp = df.copy()
    if position:
        temp = temp[temp["main_position"] == position]

    if direction == "ml_over":
        temp = temp[temp["score_diff"] > 0]
    else:
        temp = temp[temp["score_diff"] < 0]

    return (
        temp
        .assign(abs_diff=lambda x: x["score_diff"].abs())
        .sort_values("abs_diff", ascending=False)
        .head(top_n)
        [["player_name", "main_position", "final_score", "ml_pred", "score_diff"]].reset_index(drop=True))

# ML > Expert (underrated)
expert_ml_gap(final_df,position=None, direction="ml_over")
# Expert > ML (overrated)
expert_ml_gap(final_df,position=None , direction="expert_over")

# Expert vs ML scatter
def plot_expert_vs_ml(df, save=False):
    plt.figure(figsize=(7, 7))

    # Scatter points – soft blue, clean look
    plt.scatter(
        df["final_score"],
        df["ml_pred"],
        alpha=0.35,
        s=35,
        color="#1f77b4",
        edgecolor="none"
    )

    # Perfect prediction reference line
    min_val = min(df["final_score"].min(), df["ml_pred"].min())
    max_val = max(df["final_score"].max(), df["ml_pred"].max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2,
        color="gray",
        alpha=0.8,
        label="Perfect Prediction"
    )

    # Labels & title
    plt.xlabel("Expert Final Score", fontsize=11)
    plt.ylabel("ML Predicted Score", fontsize=11)
    plt.title(
        "Expert vs Machine Learning Score Comparison",
        fontsize=14,
        fontweight="bold"
    )

    # Style tweaks
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    if save:
        plt.savefig("expert_vs_ml.png", dpi=300)

    plt.show()

plot_expert_vs_ml(final_df)

# Expert vs ML scatter with position
def plot_expert_vs_ml_by_position(df,expert_col="final_score", ml_col="ml_pred", pos_col="main_position"):
    plt.figure(figsize=(7,7))
    for pos in df[pos_col].unique():
        subset = df[df[pos_col] == pos]
        plt.scatter(
            subset[expert_col],
            subset[ml_col],
            alpha=0.5,
            label=pos)
    min_val = min(df[expert_col].min(), df[ml_col].min())
    max_val = max(df[expert_col].max(), df[ml_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Expert Final Score")
    plt.ylabel("ML Predicted Score")
    plt.title("Expert vs ML Score Comparison by Position")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_expert_vs_ml_by_position(final_df)

# Ensemble score
def ensemble_score(df, expert_col="final_score", ml_col="ml_pred", alpha=0.6):
    """
    alpha = expert ağırlığı
    (1-alpha) = ML ağırlığı
    """
    return alpha * df[expert_col] + (1 - alpha) * df[ml_col]

final_df["ensemble_score"] = ensemble_score(final_df,expert_col="final_score",ml_col="ml_pred",alpha=0.6)

final_df.sort_values("ensemble_score", ascending=False).head(10)[["player_name", "main_position", "final_score", "ml_pred", "ensemble_score"]]


def plot_top10_ensemble(final_df, save=False):
    top10 = (
        final_df
        .sort_values("ensemble_score", ascending=False)
        .head(10)
        .sort_values("ensemble_score")  # alttan üste doğru güzel görünüm
    )

    plt.figure(figsize=(10, 6))

    # Soft, professional color
    plt.barh(
        top10["player_name"],
        top10["ensemble_score"],
        color="#2c7fb8",
        alpha=0.85
    )

    plt.xlabel("Ensemble Score", fontsize=11)
    plt.ylabel("Player", fontsize=11)
    plt.title(
        "Top 10 Players by Ensemble Score",
        fontsize=14,
        fontweight="bold"
    )

    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    if save:
        plt.savefig("top10_ensemble_score.png", dpi=300)

    plt.show()

plot_top10_ensemble(final_df)

to_keyfi = final_df.copy()
to_keyfi.to_pickle("pkl/keyfi.pkl")