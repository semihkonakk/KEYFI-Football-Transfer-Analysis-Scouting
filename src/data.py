import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

########## MASTER DATASETS SETUP ##########

# Datasets read
p_inj = pd.read_csv("../datasets/player_injuries.csv")
p_values = pd.read_csv("../datasets/player_latest_market_value.csv")
p_profil = pd.read_csv("../datasets/player_profiles.csv")
p_teamcomp = pd.read_csv("../datasets/team_competitions_seasons.csv")
p_performance = pd.read_csv("../datasets/player_performances.csv")
p_transfer = pd.read_csv("../datasets/transfer_history.csv")

# Edit data types
p_inj["from_date"] = pd.to_datetime(p_inj["from_date"], errors="coerce")
p_profil["date_of_birth"] = pd.to_datetime(p_profil["date_of_birth"], errors="coerce")
p_transfer["transfer_date"] = pd.to_datetime(p_transfer["transfer_date"], errors="coerce")
cols_perf_numeric = ["nb_on_pitch", "minutes_played", "goals", "assists",
                    "yellow_cards", "second_yellow_cards", "direct_red_cards",
                    "clean_sheets", "goals_conceded"]
for col in cols_perf_numeric:
    p_performance[col] = pd.to_numeric(p_performance[col], errors="coerce")

###############################################
####### MASTER DATASETS SETUP #######
###############################################

master = p_profil[[
    "player_id",
    "player_name",
    "player_image_url",
    "date_of_birth",
    "height",
    "main_position",
    "foot",
    "current_club_id",
    "current_club_name"
]].copy()

# Futbocuların toplam performanslarını alıyoruz.
perf_agg = p_performance.groupby("player_id").agg({
    "nb_on_pitch": "sum",
    "minutes_played": "sum",
    "goals": "sum",
    "assists": "sum",
    "yellow_cards": "sum",
    "second_yellow_cards": "sum",
    "direct_red_cards": "sum",
    "clean_sheets": "sum",
    "goals_conceded": "sum"
})

# Futbolcuların toplam oynadığı sezon sayısına bakıyoruz.
season_count = p_performance.groupby("player_id")["season_name"].nunique().rename("season_count")

# Hesapladığımız performans verilerini master tablosuna ekliyoruz.
master = master.merge(perf_agg, on="player_id", how="left")
master = master.merge(season_count, on="player_id", how="left")

# Futbocuların toplam Sakatlık değerlerini alıyoruz.
inj_agg = p_inj.groupby("player_id").agg({
    "days_missed": "sum",
    "games_missed": "sum",
    "injury_reason": "count"
}).rename(columns={"injury_reason": "injury_count"})

# Son sakatlanma tarihi
p_inj["from_date"] = pd.to_datetime(p_inj["from_date"])
last_injury = p_inj.groupby("player_id")["from_date"].max().rename("last_injury_date")

# Hesapladığımız sakatlık verilerini master tablosuna ekliyoruz.
master = master.merge(inj_agg, on="player_id", how="left")
master = master.merge(last_injury, on="player_id", how="left")

# Market değerlerini master tabloya ekliyoruz.
master = master.merge(p_values[["player_id", "value"]], on="player_id", how="left")
master.rename(columns={"value": "market_value"}, inplace=True)

# Futbolcunun geçmişte yaptığı transferlerin metriklerini hesaplıyoruz.
tr_agg = p_transfer.groupby("player_id").agg({
    "transfer_fee": "sum",
    "value_at_transfer": "sum",
    "transfer_date": "max"
})

# Toplam transfer sayısını hesaplıyoruz. (overpay = oyuncunun market değeri - ödenen bonservis)
tr_agg["transfer_count"] = p_transfer.groupby("player_id").size()
tr_agg["overpay"] = tr_agg["transfer_fee"] - tr_agg["value_at_transfer"]

# Transfer adedini ve overpay değerlerini master tabloya ekliyoruz.
master = master.merge(tr_agg, on="player_id", how="left")

###############################################
####### EDA (KEŞİFÇİ VERİ ANALİZİ) #######
###############################################

def check_df(dataframe, head=5):
    print("############### Shape ###############")
    print(dataframe.shape)
    print("############### Types ###############")
    print(dataframe.dtypes)
    print("############### Head ###############")
    print(dataframe.head(head))
    print("############### Tail ###############")
    print(dataframe.tail(head))
    print("############### Na ###############")
    print(dataframe.isnull().sum())
    print("############### Quantiles ###############")
    print(dataframe.describe().T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    cat_th: numeric ama az sayıda unique değeri olanı categorical say
    car_th: categorical ama çok sayıda unique değeri olanı cardinal say
    """

    # 1) Kategorik kolonlar
    cat_cols = [col for col in dataframe.columns
                if dataframe[col].dtype == "object"]

    # Sayısal olup kategorik gibi davranan kolonlar
    num_but_cat = [col for col in dataframe.columns
                   if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "object"]

    # Cardinal kolonlar (çok fazla kategori)
    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].dtype == "object" and dataframe[col].nunique() > car_th]

    # Temiz kategorik liste
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # 2) Sayısal kolonlar
    num_cols = [col for col in dataframe.columns
                if dataframe[col].dtype != "object" and col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)} ')
    print(f'num_cols: {len(num_cols)} ')
    print(f'cat_but_car: {len(cat_but_car)} ')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(master)
print(cat_cols)
print(num_cols)
print(cat_but_car)

remove_cols = [
    'player_id',
    'date_of_birth',
    'current_club_id',
    'last_injury_date',
    'transfer_date'
]

num_cols = [col for col in num_cols if col not in remove_cols]

# EKSİK DEĞERLER
def missing_values_table(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (n_miss / len(dataframe) * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print("Missing Values Summary:")
    print(missing_df)

check_df(master)
missing_values_table(master)

# Bu çıktıya göre bir düzenleme yapıyoruz.

# Master table'de işe yaramaz verileri kaldıralım.
# Emekli ve futboculu bırakmış kişileri listeden cıkarıyoruz.
master["current_club_name"].value_counts()
invalid_status = ["Retired", "Unknown", "Career break", "---"]
master = master[~master["current_club_name"].isin(invalid_status)]
master.shape
# Market Value
master["market_value"] = master["market_value"].fillna(0)
# market_value değeri olmayanları kaldırıyoruz.
master = master[master["market_value"] > 0]
# Doğum tarihi olmayanları silelim.
master = master[master["date_of_birth"].notna()]
# İsmi olmayanları silelim
master = master[master["player_name"].notna()]
master.reset_index(drop=True, inplace=True)

# Sakıncalı Kolonlar
master["days_missed"] = master["days_missed"].fillna(0)
master["games_missed"] = master["games_missed"].fillna(0)
master["injury_count"] = master["injury_count"].fillna(0)
master["season_count"] = master["season_count"].fillna(0)

# Transfer Kolonları
transfer_cols = ["transfer_fee", "value_at_transfer", "transfer_count", "overpay"]
for col in transfer_cols:
    master[col] = master[col].fillna(0)

# Performans Kolonları
perf_cols = [
    "minutes_played", "nb_on_pitch", "goals", "assists",
    "yellow_cards", "second_yellow_cards", "direct_red_cards",
    "clean_sheets", "goals_conceded"]

for col in perf_cols:
    master[col] = master[col].fillna(0)

# Foot Kolonu (Sağ/Sol Ayak)
master["foot"] = master["foot"].fillna(master["foot"].mode()[0])

# Height
master["height"] = master["height"].fillna(master["height"].median())
master.loc[master["height"] == 0, "height"] = 180

missing_values_table(master)
check_df(master)

master_eda = master.copy()
master_eda.to_pickle("pkl/master_eda.pkl")
# BU KISMA DÖNMEK İSTİYORSAK
# master = pd.read_pickle("pkl/master_eda.pkl")

###############################################
############# FEATURE ENGINEERING #############
###############################################

# Bazı veri tiplerini tekrar düzenliyoruz.
master["season_count"] = master["season_count"].astype("Int64")
master["transfer_count"] = master["transfer_count"].astype("Int64")

# 1) Güvenli dakika katsayısı
master["minutes_factor"] = master["minutes_played"] / 90

# 2) Maç başına gol & asist
master["goals_per90"] = master["goals"] / master["minutes_factor"]
master["assists_per90"] = master["assists"] / master["minutes_factor"]

# 3) Kartlar per90
master["yellow_per90"] = master["yellow_cards"] / master["minutes_factor"]
master["red_per90"] = (master["second_yellow_cards"] + master["direct_red_cards"]) / master["minutes_factor"]

# 4) Defans performansı per90
master["clean_sheets_per90"] = master["clean_sheets"] / master["minutes_factor"]
master["conceded_per90"] = master["goals_conceded"] / master["minutes_factor"]

# 5) Oyuna katkı metriği
master["goal_contribution_per90"] = (
    (master["goals"] + master["assists"]) / master["minutes_factor"])

# 6) Oynama oranı (availability ratio)
master["minutes_per_season"] = master["minutes_played"] / (master["season_count"] + 1e-9)

# 7) Sakatlık etkisi (injury impact)
master["injury_days_per_season"] = master["days_missed"] / (master["season_count"] + 1e-9)
master["injury_games_per_season"] = master["games_missed"] / (master["season_count"] + 1e-9)

# 8) Performans yoğunluğu (intensity metrics)
master["on_pitch_ratio"] = master["nb_on_pitch"] / (master["season_count"] + 1e-9)

# 9) Oyuncu yaşı
# Referans tarih: 2026-12-30
reference_date = pd.Timestamp("2025-12-30")
# Yaş hesaplama
master["age"] = (reference_date - master["date_of_birth"]).dt.days // 365
# Eksik doğum tarihlerini median yaş ile dolduruyoruz.
master["age"] = master["age"].fillna(master["age"].median())
# 42 yaşından büyük futbocuları kaldırıyoruz.
master = master[master["age"] <= 42].reset_index(drop=True)

# 10) Son sakatlıktan bu yana gün sayısı
master["injury_recency"] = (reference_date - master["last_injury_date"]).dt.days
# Hiç sakatlığı olmayan oyuncular → NaN → -1 ile işaretliyoruz
master["injury_recency"] = master["injury_recency"].fillna(-1)

# 11) Son transferden bu yana geçen gün
master["transfer_recency"] = (reference_date - master["transfer_date"]).dt.days
# Hiç transfer olmamış oyuncular → NaN → -1
master["transfer_recency"] = master["transfer_recency"].fillna(-1)

# 12) Son düzenlemeler
# PER90 kolonlarında değerleri düzeltme işlemi
per90_cols = [
    "goals_per90", "assists_per90", "yellow_per90", "red_per90",
    "clean_sheets_per90", "conceded_per90", "goal_contribution_per90"]
master.loc[:, "goals_per90"] = master["goals_per90"].clip(0, 3)
master.loc[:, "goal_contribution_per90"] = master["goal_contribution_per90"].clip(0, 4)
master.loc[:, "yellow_per90"] = master["yellow_per90"].clip(0, 1.5)
# nb_on_pitch(oyuncunun oynadığı maç sayısı) = 0 olanlarda per90 değerlerini 0 yap
master.loc[master["nb_on_pitch"] == 0, per90_cols] = 0
master.loc[master["minutes_played"] == 0, per90_cols] = 0
# toplam 0 dakika oynayanları listeden çıkart.
weird_mask = ((master["minutes_played"] == 0)&(
        (master["goals"] > 0) |
        (master["assists"] > 0) |
        (master["yellow_cards"] > 0) |
        (master["second_yellow_cards"] > 0) |
        (master["direct_red_cards"] > 0)))
master = master[~weird_mask]
# Sezon kolonlarında 0'a bölme hatalarını düzeltelim
season_cols = ["injury_days_per_season", "injury_games_per_season"]
# season_count < 1 olanlarda değerleri 0 yap
master.loc[master["season_count"] < 1, season_cols] = 0
# Maximum mantıklı aralık (clip)
master["injury_days_per_season"] = master["injury_days_per_season"].clip(0, 365)
master["injury_games_per_season"] = master["injury_games_per_season"].clip(0, 100)
# Negatif değerleri düzeltme
# injury_recency: son sakatlık yoksa → 9999
master.loc[master["last_injury_date"].isna(), "injury_recency"] = 9999
# transfer_recency: hiç transfer yoksa → 9999
master.loc[master["transfer_date"].isna(), "transfer_recency"] = 9999
# Negatif recency değerlerini sıfırlama
master["injury_recency"] = master["injury_recency"].clip(lower=0)
master["transfer_recency"] = master["transfer_recency"].clip(lower=0)
# Tip değişikliği
cols = ["injury_days_per_season", "injury_games_per_season"]
master[cols] = np.ceil(master[cols]).astype(int)
# Minumun dakika ve maç sayısı
min_minutes = 1000
min_games = 10
mask = (master["minutes_played"] >= min_minutes) & (master["nb_on_pitch"] >= min_games)
master = master[mask]

check_df(master)
missing_values_table(master)
master.describe().T

master_fe = master.copy()
master_fe.to_pickle("pkl/master_fe.pkl")
# BU KISMA DÖNMEK İSTİYORSAK
master = pd.read_pickle("../pkl/master_fe.pkl")

#örnek bir futbolcu verisi
Samu_Almagro = master[master["player_name"].str.contains("Samu Almagro", na=False)]
print(Samu_Almagro[["player_name","minutes_played","goals","goals_per90",
              "assists_per90","minutes_per_season","injury_count"]].T)

###############################################
################### SCORING ###################
###############################################

scaler = MinMaxScaler()

positional_cols = [
    "goals_per90",
    "assists_per90",
    "goal_contribution_per90",
    "conceded_per90",
    "clean_sheets_per90",
    "yellow_per90",
    "red_per90"
]
global_cols = [
    "minutes_per_season",
    "on_pitch_ratio",
    "injury_count",
    "injury_days_per_season",
    "injury_games_per_season",
    "market_value",
    "transfer_fee",
    "value_at_transfer",
    "overpay"
]

def positional_minmax(df, group_col, cols):
    df = df.copy()
    for col in cols:
        df[col] = (
            df
            .groupby(group_col)[col]
            .transform(lambda x: MinMaxScaler().fit_transform(
                x.values.reshape(-1, 1)
            ).ravel())
        )
    return df

master = positional_minmax(master,group_col="main_position",cols=positional_cols)
master[global_cols] = scaler.fit_transform(master[global_cols])

### 1) PERFORMANCE SCORE

# Attack
master["attack_score"] = (master["goals_per90"] * 0.45 +
                          master["goal_contribution_per90"] * 0.30 +
                          master["minutes_per_season"] * 0.15 -
                          master["injury_games_per_season"] * 0.10)
# Midfield
master["midfield_score"] = (master["goal_contribution_per90"] * 0.30 +
                            master["assists_per90"] * 0.25 +
                            master["minutes_per_season"] * 0.30 -
                            master["injury_games_per_season"] * 0.05 -
                            master["yellow_per90"] * 0.05 -
                            master["red_per90"] * 0.05)
# Defender
master["defense_score"] = (master["minutes_per_season"] * 0.45 -
                           master["conceded_per90"] * 0.25 -
                           master["injury_games_per_season"] * 0.10 -
                           master["red_per90"] * 0.10 -
                           master["yellow_per90"] * 0.10)
# Goalkeeper
master["gk_score"] = (master["clean_sheets_per90"] * 0.65 -
                      master["conceded_per90"] * 0.25 +
                      master["minutes_per_season"] * 0.10)

score_map = {
    "Attack": "attack_score",
    "Midfield": "midfield_score",
    "Defender":"defense_score",
    "Goalkeeper": "gk_score"}

master["performance_score"] = master.apply(lambda row: row[score_map[row["main_position"]]],axis=1)

# master[["player_name","main_position","performance_score"]].sort_values(ascending=False, by="performance_score").head(10)

### 2) AGE SCORE

def compute_age_score(age):
    if age < 30:
        return 1 + (30 - age) * 0.08
    elif 30 <= age < 33:
        return 1.0
    else:
        return  1 - (age - 33) * 0.05

master["age_score"] = master["age"].apply(compute_age_score)

# master[["player_name","age","performance_score","age_score"]].
# sort_values(ascending=True, by=["age_score","performance_score"]).head(10)

### 3) RISK SCORE

risk_value = 1 - master["minutes_per_season"]

master["risk_score"] = (
    master["injury_count"] * 0.20 +
    master["injury_days_per_season"] * 0.10 +
    risk_value * 0.20 +
    (1 / master["age_score"]) * 0.50)

(master[["player_name","age","injury_count","age_score","injury_days_per_season","risk_score"]].
 sort_values(ascending=False, by=["risk_score"]).head(10))

### 4)AVAILABILITY SCORE

master["availability_score"] = (master["minutes_per_season"] * 0.70 +
                                (master["season_count"] / master["season_count"].mean()) * 0.30)

(master[["player_name","age","performance_score","age_score","risk_score","availability_score"]].
 sort_values(ascending=False, by=["availability_score","performance_score"]).head(10))

### 5) MARKET SCORE

master["market_score"] = (
        (1 - master["transfer_fee"]) * 0.20 +
        master["market_value"] * 0.60 +
        (1 - master["overpay"]) * 0.20)

# (master[["player_name","age","performance_score","age_score","risk_score","availability_score","market_score"]].
# sort_values(ascending=False, by="market_score").head(10))

### 7) TRANSFER SCORE

master["transfer_score"] = (
        master["performance_score"] * 0.30 +
        master["availability_score"] * 0.20 +
        master["market_score"] * 0.15 +
        (1 - master["risk_score"]) * 0.35)

(master[["player_name","performance_score","age_score","risk_score","availability_score","market_score","transfer_score"]].
sort_values(ascending=False, by="transfer_score").head(10))

all_scores = ["performance_score", "risk_score", "availability_score",
              "transfer_score", "market_score"]

master[all_scores] = scaler.fit_transform(master[all_scores])

######### FINAL SCORE ###########

weights = {
    "Attack" :     {"risk": 0.10, "avail": 0.05, "transfer": 0.10, "market": 0.25, "perf": 0.50},
    "Midfield" :   {"risk": 0.10, "avail": 0.30, "transfer": 0.05, "market": 0.05, "perf": 0.50},
    "Defender" :   {"risk": 0.15, "avail": 0.25, "transfer": 0.05, "market": 0.05, "perf": 0.50},
    "Goalkeeper" : {"risk": 0.15, "avail": 0.25, "transfer": 0.05, "market": 0.05, "perf": 0.50}
    }

def calc_final_score(row):
    w = weights.get(row["main_position"])
    return (
        - row["risk_score"] * w["risk"] +
        row["availability_score"] * w["avail"] +
        row["transfer_score"] * w["transfer"] +
        row["market_score"] * w["market"] +
        row["performance_score"] * w["perf"]
    )

master["final_score"] = master.apply(calc_final_score, axis=1)

top20 = master.sort_values("final_score", ascending=False)
top20[["player_name", "main_position", "age","performance_score",
       "market_score","risk_score","availability_score", "transfer_score","final_score"]].head(20)

for_DV = master.copy()
for_DV.to_pickle("pkl/forDV.pkl")
