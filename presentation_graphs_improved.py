
"""
Presentation Quality Graphs for BMI Research
Improved version with:
- Color coding
- Value labels
- Better formatting
- Publication-ready styling

Required:
    bmi_game_metrics.csv

Outputs:
    01_bmi_comparison.png
    02_feature_importance.png
    03_model_performance.png
    04_f2p_vs_premium.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

df = pd.read_csv("bmi_game_metrics.csv")

# -----------------------------
# Helper
# -----------------------------

def human_format(num):

    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"

    if num >= 1_000:
        return f"{num/1_000:.1f}K"

    return f"{num:.0f}"


# -----------------------------
# GRAPH 1
# BMI COMPARISON
# -----------------------------

premium = df[df["free_to_play"] == 0]["adoption_scale"].mean()
f2p = df[df["free_to_play"] == 1]["adoption_scale"].mean()

single = df[df["multiplayer"] == 0]["adoption_scale"].mean()
multi = df[df["multiplayer"] == 1]["adoption_scale"].mean()

other = df[df["live_service"] == 0]["adoption_scale"].mean()
live = df[df["live_service"] == 1]["adoption_scale"].mean()

labels = [
    "Premium",
    "F2P",
    "Singleplayer",
    "Multiplayer",
    "Other",
    "Live Service"
]

values = [
    premium,
    f2p,
    single,
    multi,
    other,
    live
]

colors = [
    "#4C72B0",
    "#55A868",
    "#7F7F7F",
    "#8172B2",
    "#A0A0A0",
    "#DD8452"
]

plt.figure()

bars = plt.bar(
    labels,
    values,
    color=colors
)

for bar in bars:

    h = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        h,
        human_format(h),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.title(
    "Average Adoption Scale by Business Model Characteristics",
    fontweight="bold"
)

plt.ylabel("Average Players")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    "01_bmi_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -----------------------------
# GRAPH 2
# FEATURE IMPORTANCE
# -----------------------------

features = [
    "Game Age",
    "Review Score",
    "DLC Count",
    "Price",
    "Live Service",
    "Multiplayer",
    "Competitive",
    "Free-to-Play"
]

importance = [
    0.284083,
    0.235009,
    0.208423,
    0.170259,
    0.055731,
    0.030320,
    0.009953,
    0.006220
]

colors = [
    "#4C72B0",
    "#4C72B0",
    "#4C72B0",
    "#4C72B0",
    "#DD8452",
    "#DD8452",
    "#DD8452",
    "#DD8452"
]

feature_df = pd.DataFrame({
    "feature": features,
    "importance": importance,
    "color": colors
})

feature_df = feature_df.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

bars = plt.barh(
    feature_df["feature"],
    feature_df["importance"],
    color=feature_df["color"]
)

for bar in bars:

    w = bar.get_width()

    plt.text(
        w + 0.003,
        bar.get_y() + bar.get_height()/2,
        f"{w:.3f}",
        va="center"
    )

plt.title(
    "Factors Influencing BMI Adoption Success",
    fontweight="bold"
)

plt.xlabel("Feature Importance")

plt.tight_layout()

plt.savefig(
    "02_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# GRAPH 3
# MODEL PERFORMANCE
# -----------------------------

models = [
    "Adoption Scale",
    "Adoption Speed",
    "Retention"
]

scores = [
    0.3831,
    0.0429,
    0.0069
]

colors = [
    "#55A868",
    "#DD8452",
    "#C44E52"
]

plt.figure()

bars = plt.bar(
    models,
    scores,
    color=colors
)

for bar in bars:

    h = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        h,
        f"{h:.3f}",
        ha="center",
        va="bottom"
    )

plt.title(
    "Model Performance (R²)",
    fontweight="bold"
)

plt.ylabel("R² Score")

plt.tight_layout()

plt.savefig(
    "03_model_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# GRAPH 4
# F2P VS PREMIUM
# -----------------------------

plt.figure()

bars = plt.bar(
    ["Premium", "F2P"],
    [premium, f2p],
    color=["#4C72B0", "#55A868"]
)

for bar in bars:

    h = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        h,
        human_format(h),
        ha="center",
        va="bottom"
    )

ratio = f2p / premium if premium > 0 else 0

plt.title(
    f"F2P Games Reach {ratio:.1f}× Larger Audience",
    fontweight="bold"
)

plt.ylabel("Average Players")

plt.tight_layout()

plt.savefig(
    "04_f2p_vs_premium.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Created:")
print("01_bmi_comparison.png")
print("02_feature_importance.png")
print("03_model_performance.png")
print("04_f2p_vs_premium.png")
