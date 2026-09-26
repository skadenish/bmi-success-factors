# Business Model Innovation in the Video Game Industry
### Identifying Success Factors Using Steam Platform Data

**Author:** Kulakov Anatolii
**Supervisor:** LÃ¡szlÃ³ MolnÃ¡r

## ð¥ Data

> **[â¬ï¸ Download the initial datasets (RAR)](https://www.dropbox.com/scl/fi/qk97lokkh3wow6i3ssb52/initial_data.rar?rlkey=j9ydwka68fm0h3gpifh1efc51&st=pklb6vhc&e=1&dl=0)**

## Overview

Modern gaming increasingly relies on innovative business models â Free-to-Play (F2P), Live-Service, and multiplayer ecosystems. This project investigates **which factors drive the adoption success** of these business models on the Steam platform, combining game metadata with real-world player activity data.

## Objectives

- Identify Business Model Innovation (BMI) characteristics (F2P, Live-Service, Multiplayer, Competitive)
- Measure adoption success across three dimensions: **scale**, **speed**, and **retention**
- Build predictive models using game and business-model features
- Determine which factors matter most for adoption

## Data

- **Steam Games Dataset 2025** (~6,700 games) â price, reviews, DLCs, release date, BMI classification
- **SteamCharts Dataset** â monthly player activity, used to derive adoption metrics

After cleaning, merging, and filtering out games with less than 6 months of player history, the final analytical dataset contained **~5,000 games**.

## Methodology

1. Feature engineering from Steam metadata (Price, Review Score, DLC Count, Game Age, Free-to-Play, Multiplayer, Competitive, Live Service)
2. Derivation of adoption metrics from player activity: **Adoption Scale**, **Adoption Speed**, **Retention**
3. Random Forest regression (500 trees) trained separately for each target variable
4. Feature importance analysis and model performance evaluation (RÂ²)

## Key Findings

- **Product characteristics outweigh BMI indicators**: Game Age (28.4%), Review Score (23.5%), DLC Count (20.8%), and Price (17.0%) were the strongest predictors of adoption â well ahead of Live-Service, Multiplayer, Competitive, or F2P flags.
- **F2P games reach ~3.8Ã larger audiences** than premium (paid) titles on average.
- **Live-Service games achieve the highest average audience** (~11.2K players), ahead of F2P (~5.3K) and Multiplayer (~4.5K).
- Model performance varied sharply by target: **Adoption Scale** was moderately predictable (RÂ² = 0.383), while **Adoption Speed** (RÂ² = 0.043) and **Retention** (RÂ² = 0.007) were very difficult to predict from these features alone.

## Conclusion

Business model choice (e.g., F2P, Live-Service) clearly affects a game's market reach, but **long-term success depends more on product-level factors** â game maturity, review quality, and continuous content support â than on the business model itself. Retention and long-term engagement appear to be driven by gameplay quality and community factors not captured in this dataset.

## Limitations

- Analysis limited to the Steam platform
- Limited set of BMI indicators
- Retention and adoption speed likely require richer behavioral data to model well

## Tech Stack

- Python (pandas, scikit-learn)
- Random Forest Regression
- Data visualization (matplotlib)
