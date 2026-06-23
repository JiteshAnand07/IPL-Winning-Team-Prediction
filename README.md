<div align="center">

# 🏏 IPL Winning Team Prediction System

**An end-to-end machine learning project predicting IPL match outcomes — before the toss, and live during the chase.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#-project-overview) •
[Demo](#%EF%B8%8F-live-demo) •
[Features](#-features) •
[Setup](#%EF%B8%8F-installation--setup) •
[Model Performance](#-model-performance-summary) •
[Limitations](#%EF%B8%8F-limitations--honest-disclosures)

</div>

---

## 📌 Project Overview

This project builds **two separate prediction systems** from real IPL ball-by-ball data (2008–2024):

| | What it predicts | When it's usable |
|---|---|---|
| 🏏 **Pre-Match Predictor** | Match winner before a single ball is bowled | Pre-toss: teams, venue, city, toss |
| 📈 **Live-Match Predictor** | Run-chase winner in real time | Mid-innings: current score, wickets, overs, target |

Rather than chasing an inflated headline accuracy number, this project deliberately reports **honest, held-out test performance** for both systems:

| Predictor | Final Model | Test ROC-AUC | Why |
|---|---|---|---|
| Pre-Match | Logistic Regression *(tuned)* | **≈ 0.55** | T20 outcomes are genuinely hard to call before play begins — a small, real edge over chance, not a modeling failure. |
| Live-Match | Gradient Boosting | **≈ 0.90–0.92** | Live match state (required run rate, wickets, overs) is a strong, direct predictor once the chase is underway. |

> **The gap between these two numbers *is* the finding.** Pre-match cricket prediction sits close to a coin flip with a real but modest edge; live-match prediction is highly reliable. Full methodology — including the point-in-time feature engineering and the match-level train/test split used to prevent leakage — is in the [notebook](notebooks/IPL_Winning_Team_Prediction.ipynb).

---

## 🖼️ Live Demo

<div align="center">
<img src="assets/screenshot_prematch.png" alt="Pre-Match Predictor screenshot" width="800">
<br><em>Pre-Match Predictor — win probability bar, confidence score, and plain-language explanation</em>
<br><br>
<img src="assets/screenshot_live.png" alt="Live Match Predictor screenshot" width="800">
<br><em>Live Match Predictor — real-time win probability as the chase unfolds</em>
</div>

> Add your own screenshots to `assets/` and update the paths above — see [Adding Screenshots](#adding-screenshots) below.

---

## ✨ Features

### 🏏 Pre-Match Predictor
- **Inputs:** Team 1, Team 2, Venue, City, Toss Winner, Toss Decision
- **Engineered features:** team win %, recent form (last 5 matches), head-to-head win %, venue/city-specific win %, toss impact — all computed **point-in-time** to avoid future-data leakage
- **Output:** predicted winner, win probability for both teams, model confidence, plain-language explanation of the key drivers

### 📈 Live-Match Predictor
- **Inputs:** Batting Team, Bowling Team, Venue, City, Current Score, Wickets Lost, Overs, Target Score
- **Auto-calculated:** runs/balls/wickets remaining, current & required run rate, run-rate differential
- **Context features:** venue average first-innings score, venue chase-success rate, team chase/defend success %
- **Built-in match-logic guards:**
  - Target already reached → batting team declared the winner outright, no model call needed
  - All out without reaching the target → bowling team declared the winner
  - Overs exhausted without reaching the target → bowling team declared the winner
  - Required run rate display is capped (`>36.0`) rather than showing distorted extrapolated numbers on the final ball or two
- **Output:** predicted winner, live win probability, dynamic progress bar, automated match-situation insights

### 📊 Analytics Dashboard
All-time team win %, toss-decision trends, toss-impact breakdown, venue chase-success rankings, season-over-season match volume.

---

## 📂 Dataset

| File | Rows | Description |
|---|---|---|
| `matches.csv` | 1,095 | One row per match — teams, venue, city, toss, winner, result margin |
| `deliveries.csv` | 260,920 | One row per ball — batting/bowling team, over, ball, runs, wickets, dismissals |

**Cleaning applied before modeling:**
- Team names standardized across franchise rebrands (e.g. Delhi Daredevils → Delhi Capitals, Kings XI Punjab → Punjab Kings)
- Venue names consolidated (several grounds were recorded under 2–3 different string variants)
- No-result/abandoned matches and rows with a null winner dropped (no usable target)
- Super-over balls excluded from live-match feature engineering (outside normal 1st/2nd-innings chase logic)

---

## 🏗️ Project Structure

```
IPL_Winning_Team_Prediction/
│
├── notebooks/
│   └── IPL_Winning_Team_Prediction.ipynb   # Full pipeline, executed end-to-end with outputs
│
├── data/
│   ├── matches.csv
│   └── deliveries.csv
│
├── models/
│   ├── pre_match_model.pkl                 # Full sklearn pipeline (preprocessing + classifier)
│   ├── live_match_model.pkl                # Full sklearn pipeline (preprocessing + classifier)
│   └── encoders/
│       ├── pre_match_lookups.pkl           # Team/venue/city/H2H stat lookup tables
│       └── live_match_lookups.pkl          # Venue/team chase-stat lookup tables
│
├── assets/                                 # Screenshots, logo (optional, see below)
├── app.py                                  # Multi-page Streamlit application
├── requirements.txt
└── README.md
```

Each model is saved as a **single joblib artifact** containing the entire `ColumnTransformer` + classifier pipeline, so the app calls `.predict_proba()` directly on a raw feature row — no separate encoder/scaler bookkeeping at inference time. The `lookups.pkl` files store the point-in-time statistics needed to construct that feature row from whatever the user selects in the UI.

---

## 🔬 Modeling Approach

Rather than mechanically running every algorithm available, the notebook trains **5–6 complementary, well-established models** per predictor (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM for the live model), compares them on Accuracy / Precision / Recall / F1 / ROC-AUC, and tunes the best one with `RandomizedSearchCV`. CatBoost and Extra Trees were intentionally left out — on this dataset size they converge to performance very close to the models already included, adding training time and dependency surface without new insight.

**Leakage prevention:**
- Pre-match features (team win %, recent form, H2H, venue/city win %) are computed **point-in-time** — only from matches that occurred *before* the match being featurized.
- The live-match model is split into train/test **by `match_id`**, not by row, so the model never sees other balls from a match it's being tested on.

**A real bug caught during manual testing, fixed and documented rather than hidden:** the pre-match model was initially deployed as a `DecisionTreeClassifier` (the leaderboard's top-AUC pick). Its shallow tree produced only 25 leaves / 12 unique probability outputs across the entire input space — many genuinely different team/venue/toss combinations returned identical predicted probabilities, which surfaced as a "stuck" demo. Logistic Regression was deployed instead: statistically tied on AUC (0.527 vs 0.530), but smooth and properly input-sensitive. Full writeup in notebook Section 6.3.1.

---

## 🖥️ Installation & Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/IPL_Winning_Team_Prediction.git
cd IPL_Winning_Team_Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 📓 Running the Notebook

```bash
cd notebooks
jupyter notebook IPL_Winning_Team_Prediction.ipynb
```

Runs top-to-bottom without modification (paths are relative to `notebooks/`, e.g. `../data/matches.csv`) and regenerates everything under `models/` if re-run.

> ⏱️ Full notebook runtime (hyperparameter search across both predictors + permutation importance + SHAP) is roughly 5–7 minutes on a typical machine.

## 🚀 Running the Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the sidebar to switch between the Pre-Match Predictor, Live Match Predictor, and Analytics Dashboard.

<details>
<summary><strong>Adding screenshots</strong></summary>

1. Run the app locally and capture screenshots of each page
2. Save them to `assets/screenshot_prematch.png` and `assets/screenshot_live.png`
3. Commit and push — the [Live Demo](#%EF%B8%8F-live-demo) section above will render them automatically on GitHub

</details>

<details>
<summary><strong>Running on Google Colab</strong></summary>

1. Upload `IPL_Winning_Team_Prediction.ipynb`, `matches.csv`, and `deliveries.csv` to a Colab session or Google Drive folder
2. Mount Drive (if using Drive) and `os.chdir()` into the folder, or just update the `pd.read_csv(...)` paths to wherever you uploaded the CSVs
3. Add `os.makedirs('models/encoders', exist_ok=True)` before the model-saving cells
4. `!pip install -q xgboost lightgbm shap joblib plotly`
5. `Runtime → Run all`

</details>

---

## 📊 Model Performance Summary

**Pre-Match Predictor** (5-model comparison, test set):

| Model | Accuracy | F1 | ROC-AUC | Deployed? |
|---|---|---|---|---|
| Decision Tree | 0.514 | 0.495 | 0.530 *(leaderboard top)* | ❌ — coarse probability output, see [Limitations](#%EF%B8%8F-limitations--honest-disclosures) |
| **Logistic Regression** | 0.514 | 0.527 | 0.527 untuned → **0.549 tuned** | ✅ **Deployed** |
| XGBoost | 0.477 | 0.509 | 0.513 | ❌ |
| Random Forest | 0.482 | 0.519 | 0.481 | ❌ |
| Gradient Boosting | 0.450 | 0.478 | 0.446 | ❌ |

**Live-Match Predictor** (6-model comparison, test set, match-level split):

| Model | Accuracy | F1 | ROC-AUC | Deployed? |
|---|---|---|---|---|
| **Gradient Boosting** | 0.828 | 0.846 | **0.919** | ✅ **Deployed** |
| LightGBM | 0.818 | 0.837 | 0.914 | ❌ |
| Random Forest | 0.800 | 0.822 | 0.901 | ❌ |
| Logistic Regression | 0.780 | 0.805 | 0.884 | ❌ |
| XGBoost | 0.788 | 0.812 | 0.871 | ❌ |
| Decision Tree | 0.776 | 0.795 | 0.804 | ❌ |

*(Exact numbers shift slightly on re-run due to the random train/test split — the **gap** between pre-match and live-match performance is the stable, expected finding.)*

**Top features driving the live-match model** (via feature importance + permutation importance + SHAP): required run rate, runs remaining, balls remaining, and current run rate dominate — matching cricketing intuition directly. Venue and team chase/defend statistics provide secondary signal. Toss plays no role here, since it's already "priced in" by the time a chase is underway.

---

## 🔮 Future Improvements

- Player-level features (current batters' strike rate, bowler economy) for a sharper live-match model
- Chronological (not random) train/test split for the pre-match model, to simulate predicting genuinely unseen future seasons
- `GroupKFold` cross-validation (grouped by `match_id`) for the live model — a more robust leakage-safe CV estimate beyond the single match-level holdout used here
- Phase-aware weighting (powerplay / middle overs / death overs) for the live model, since run-rate dynamics differ meaningfully across these
- CatBoost as an additional comparison point if categorical-native boosting becomes a priority

---

## ⚠️ Limitations & Honest Disclosures

- **Pre-match ROC-AUC (~0.55) is only modestly better than chance (0.50).** Reported transparently — this reflects genuine T20 variance, not a modeling failure.
- **The dataset is ~1,090 matches after cleaning** — small enough that high-cardinality categorical features (venue, city) carry real overfitting risk, part of why the pre-match ceiling is what it is.
- **A real "stuck probability" bug was caught and fixed during manual app testing** (see [Modeling Approach](#-modeling-approach) above) — documented rather than silently patched.
- **Models were trained on a single random split.** Production deployment should use time-based validation (train on past seasons, test on the most recent one) for a realistic forward-looking estimate.
- **Extreme/invalid live-match inputs are explicitly guarded** (all-out, target-already-reached, overs-exhausted) rather than letting the model extrapolate into match states it was never trained on.

---

## 📜 License & Trademark Notice

This project is released under the [MIT License](LICENSE) for the code, notebook, and app logic.

The "IPL" name and any associated logos/trademarks are the property of the BCCI and are **not** owned by or affiliated with this project. They are referenced here purely for educational, non-commercial analysis of publicly available match data. If you fork this project for a public-facing deployment, please avoid using the official IPL logo/branding and consider an original or generic cricket-themed visual instead.

---

<div align="center">

Built as an end-to-end ML project — feedback and PRs welcome.

</div>
