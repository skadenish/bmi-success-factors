
"""
BMI Adoption Success Analysis
Uses:
1. games_march2025_cleaned.csv
2. steamcharts.csv

Outputs:
- bmi_game_metrics.csv
- feature_importance.csv
- adoption_model_summary.txt

Research idea:
AdoptionScale = max monthly players
AdoptionSpeed = months to reach 50% of peak
Retention = latest players / peak players

Predictors:
price, reviews, playtime, DLCs, F2P, live service, multiplayer, etc.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

METADATA_FILE = "games_march2025_cleaned.csv"
STEAMCHARTS_FILE = "steamcharts.csv"

print("Loading data...")

games = pd.read_csv(METADATA_FILE)
charts = pd.read_csv(STEAMCHARTS_FILE)

games.columns = [c.lower() for c in games.columns]
charts.columns = [c.lower() for c in charts.columns]

# -----------------------------
# BMI FEATURES
# -----------------------------

games["free_to_play"] = (
    pd.to_numeric(games["price"], errors="coerce")
    .fillna(0)
    .eq(0)
    .astype(int)
)

games["multiplayer"] = games["categories"].astype(str).str.contains(
    "multi-player",
    case=False,
    na=False
).astype(int)

games["competitive"] = games["tags"].astype(str).str.contains(
    "competitive|pvp|e-sports|esports",
    case=False,
    regex=True,
    na=False
).astype(int)

LIVE_TERMS = [
    "mmorpg",
    "massively multiplayer",
    "battle royale",
    "online co-op",
    "pvp"
]

def live_service_detector(row):

    text = (
        str(row.get("genres", ""))
        + " "
        + str(row.get("tags", ""))
    ).lower()

    score = 0

    for term in LIVE_TERMS:
        if term in text:
            score += 1

    if row["free_to_play"] == 1:
        score += 1

    return int(score >= 2)

games["live_service"] = games.apply(
    live_service_detector,
    axis=1
)

# -----------------------------
# STEAMCHARTS METRICS
# -----------------------------

charts["avg_players"] = pd.to_numeric(
    charts["avg_players"],
    errors="coerce"
)

charts = charts.dropna(subset=["avg_players"])

metrics = []

for appid, grp in charts.groupby("steam_appid"):

    grp = grp.sort_values("month")

    players = grp["avg_players"].values

    if len(players) < 6:
        continue

    peak = np.max(players)

    if peak <= 0:
        continue

    latest = players[-1]

    half_peak = peak * 0.5

    months_to_half = None

    for i, val in enumerate(players):
        if val >= half_peak:
            months_to_half = i + 1
            break

    metrics.append({
        "appid": appid,
        "adoption_scale": peak,
        "adoption_speed": months_to_half,
        "retention": latest / peak
    })

metrics = pd.DataFrame(metrics)

# -----------------------------
# MERGE
# -----------------------------

merged = games.merge(
    metrics,
    left_on="appid",
    right_on="appid",
    how="inner"
)

merged.to_csv(
    "bmi_game_metrics.csv",
    index=False
)

print("Merged games:", len(merged))

# -----------------------------
# RANDOM FOREST
# -----------------------------

if "release_date" in merged.columns:

    merged["release_date"] = pd.to_datetime(
        merged["release_date"],
        errors="coerce"
    )

    merged["game_age_years"] = (
        pd.Timestamp.today() - merged["release_date"]
    ).dt.days / 365.25
else:
    merged["game_age_years"] = 0

candidate_features = [
    "price",
    "dlc_count",
    "pct_pos_total",
    "num_reviews_total",
    "average_playtime_forever",
    "free_to_play",
    "multiplayer",
    "competitive",
    "live_service",
    "game_age_years"
]

features = [
    c for c in candidate_features
    if c in merged.columns
]

TARGET = "adoption_scale"

rf_df = merged[features + [TARGET]].copy()

rf_df = rf_df.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

X = rf_df[features]
y = np.log1p(rf_df[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

with open(
    "adoption_model_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(f"R2 score: {r2}\n\n")
    f.write("Feature importance:\n")
    f.write(importance.to_string(index=False))

print()
print("R2 =", round(r2, 4))
print()
print(importance.head(15))

print()
print("Files created:")
print("- bmi_game_metrics.csv")
print("- feature_importance.csv")
print("- adoption_model_summary.txt")
