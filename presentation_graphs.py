
"""
Presentation Graphs for BMI Research
Creates publication/presentation quality figures.

Required files:
- bmi_game_metrics.csv

Outputs:
- graph_f2p_vs_premium.png
- graph_live_service.png
- graph_multiplayer.png
- graph_retention.png
- graph_feature_importance.png
- graph_model_r2.png
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv("bmi_game_metrics.csv")

# --------------------------
# F2P vs Premium
# --------------------------

tmp = df.groupby("free_to_play")["adoption_scale"].mean()

plt.figure()
plt.bar(
    ["Premium", "F2P"],
    [tmp.get(0, 0), tmp.get(1, 0)]
)
plt.title("Average Adoption Scale: Premium vs F2P")
plt.ylabel("Average Players")
plt.tight_layout()
plt.savefig("graph_f2p_vs_premium.png", dpi=300)
plt.close()

# --------------------------
# Live Service
# --------------------------

tmp = df.groupby("live_service")["adoption_scale"].mean()

plt.figure()
plt.bar(
    ["Other", "Live Service"],
    [tmp.get(0, 0), tmp.get(1, 0)]
)
plt.title("Average Adoption Scale: Live Service")
plt.ylabel("Average Players")
plt.tight_layout()
plt.savefig("graph_live_service.png", dpi=300)
plt.close()

# --------------------------
# Multiplayer
# --------------------------

tmp = df.groupby("multiplayer")["adoption_scale"].mean()

plt.figure()
plt.bar(
    ["Singleplayer", "Multiplayer"],
    [tmp.get(0, 0), tmp.get(1, 0)]
)
plt.title("Average Adoption Scale: Multiplayer")
plt.ylabel("Average Players")
plt.tight_layout()
plt.savefig("graph_multiplayer.png", dpi=300)
plt.close()

# --------------------------
# Retention
# --------------------------

tmp = df.groupby("free_to_play")["retention"].mean()

plt.figure()
plt.bar(
    ["Premium", "F2P"],
    [tmp.get(0, 0), tmp.get(1, 0)]
)
plt.title("Average Retention by Business Model")
plt.ylabel("Retention")
plt.tight_layout()
plt.savefig("graph_retention.png", dpi=300)
plt.close()

# --------------------------
# Feature Importance
# --------------------------

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

feature_df = pd.DataFrame({
    "feature": features,
    "importance": importance
})

feature_df = feature_df.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(9, 6))
plt.barh(
    feature_df["feature"],
    feature_df["importance"]
)
plt.title("Factors Influencing BMI Adoption Success")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("graph_feature_importance.png", dpi=300)
plt.close()

# --------------------------
# Model R²
# --------------------------

plt.figure()

plt.bar(
    [
        "Adoption Scale",
        "Adoption Speed",
        "Retention"
    ],
    [
        0.3831,
        0.0429,
        0.0069
    ]
)

plt.ylabel("R²")
plt.title("Model Performance Comparison")
plt.tight_layout()
plt.savefig("graph_model_r2.png", dpi=300)
plt.close()

print("All presentation graphs created.")
