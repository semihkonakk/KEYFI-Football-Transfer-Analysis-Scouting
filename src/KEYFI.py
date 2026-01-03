# **************************************
# ** TRANSFER RECOMMENDATION SYSTEM **
# **************************************
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

keyfi = pd.read_pickle("../pkl/keyfi.pkl")
m = pd.read_pickle("../pkl/master_fe.pkl")

# Eski market değelerini getiriyoruz.
def restore_original_columns(target_df,source_df,cols,id_col="player_id",strict=True):
    """
    target_df : normalize edilmiş dataframe (keyfi)
    source_df : orijinal değerlerin olduğu dataframe (master_fe)
    cols      : geri yüklenecek kolon listesi
    id_col    : satır eşleşmesi için id kolonu
    strict    : True ise shape ve id birebir kontrol edilir
    """

    if strict:
        assert target_df.shape[0] == source_df.shape[0], "Row sayıları eşleşmiyor"
        assert (target_df[id_col].values == source_df[id_col].values).all(), \
            f"{id_col} sıralaması eşleşmiyor"

    for col in cols:
        if col in source_df.columns:
            target_df[col] = source_df[col].values
        else:
            print(f"⚠️ {col} source_df içinde yok, atlandı.")

    return target_df

cols= [
    "goals_per90",
    "assists_per90",
    "goal_contribution_per90",
    "conceded_per90",
    "clean_sheets_per90",
    "yellow_per90",
    "red_per90",
    "minutes_per_season",
    "on_pitch_ratio",
    "injury_count",
    "injury_days_per_season",
    "injury_games_per_season",
    "market_value",
    "transfer_fee",
    "value_at_transfer",
    "overpay"]

x = restore_original_columns(target_df=keyfi,source_df=m,cols=cols).copy()

score_col= ['ensemble_score','attack_score', 'midfield_score',
            'defense_score', 'gk_score', 'performance_score',
            'age_score', 'risk_score', 'availability_score',
            'market_score', 'transfer_score', 'final_score','ml_pred']

x[score_col] = x[score_col] * 100
keyfi[score_col].describe().T
keyfi = x.copy()
keyfi[score_col] = keyfi[score_col].round(2)
keyfi.to_pickle("pkl/app.pkl")

