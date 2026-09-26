
"""
BMI Research Models
Model A: Adoption Scale
Model B: Adoption Speed
Model C: Retention

IMPORTANT:
- num_reviews_total removed (data leakage)
- average_playtime_forever removed (likely consequence of success)

Outputs:
feature_importance_adoption_scale.csv
feature_importance_adoption_speed.csv
feature_importance_retention.csv
model_results_summary.txt
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

METADATA_FILE = "games_march2025_cleaned.csv"
STEAMCHARTS_FILE = "steamcharts.csv"

# -----------------------------
# LOAD
# -----------------------------

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
    "competitive|pvp|esports|e-sports",
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
# AGE
# -----------------------------

games["release_date"] = pd.to_datetime(
    games["release_date"],
    errors="coerce"
)

games["game_age_years"] = (
    pd.Timestamp.today() - games["release_date"]
).dt.days / 365.25

# -----------------------------
# ADOPTION METRICS
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

    months_to_half = np.nan

    for i, value in enumerate(players):
        if value >= half_peak:
            months_to_half = i + 1
            break

    metrics.append({
        "appid": appid,
        "adoption_scale": peak,
        "adoption_speed": months_to_half,
        "retention": latest / peak
    })

metrics = pd.DataFrame(metrics)

merged = games.merge(
    metrics,
    on="appid",
    how="inner"
)

print("Merged games:", len(merged))

# -----------------------------
# FEATURES
# -----------------------------

candidate_features = [
    "price",
    "dlc_count",
    "pct_pos_total",
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

# -----------------------------
# MODEL FUNCTION
# -----------------------------

def run_model(target, output_file):

    df = merged[features + [target]].copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    X = df[features]
    y = df[target]

    if target != "retention":
        y = np.log1p(y)

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
        output_file,
        index=False
    )

    return r2, importance

# -----------------------------
# MODEL A
# -----------------------------

r2_scale, imp_scale = run_model(
    "adoption_scale",
    "feature_importance_adoption_scale.csv"
)

# -----------------------------
# MODEL B
# -----------------------------

r2_speed, imp_speed = run_model(
    "adoption_speed",
    "feature_importance_adoption_speed.csv"
)

# -----------------------------
# MODEL C
# -----------------------------

r2_ret, imp_ret = run_model(
    "retention",
    "feature_importance_retention.csv"
)

# -----------------------------
# REPORT
# -----------------------------

with open(
    "model_results_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("MODEL A: ADOPTION SCALE\n")
    f.write(f"R2 = {r2_scale}\n\n")
    f.write(imp_scale.to_string(index=False))

    f.write("\n\n")
    f.write("=" * 60)
    f.write("\n\n")

    f.write("MODEL B: ADOPTION SPEED\n")
    f.write(f"R2 = {r2_speed}\n\n")
    f.write(imp_speed.to_string(index=False))

    f.write("\n\n")
    f.write("=" * 60)
    f.write("\n\n")

    f.write("MODEL C: RETENTION\n")
    f.write(f"R2 = {r2_ret}\n\n")
    f.write(imp_ret.to_string(index=False))

print("\nMODEL A (ADOPTION SCALE)")
print("R2 =", round(r2_scale, 4))
print(imp_scale)

print("\nMODEL B (ADOPTION SPEED)")
print("R2 =", round(r2_speed, 4))
print(imp_speed)

print("\nMODEL C (RETENTION)")
print("R2 =", round(r2_ret, 4))
print(imp_ret)

print("\nFiles created:")
print("feature_importance_adoption_scale.csv")
print("feature_importance_adoption_speed.csv")
print("feature_importance_retention.csv")
print("model_results_summary.txt")
