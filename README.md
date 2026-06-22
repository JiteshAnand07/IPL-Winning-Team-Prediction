# 🏏 IPL Winning Team Prediction System

An end-to-end machine learning project that predicts IPL match outcomes in two distinct ways:

1. **Pre-Match Winner Predictor** — predicts the winner using only information available before the toss (teams, venue, city, toss).
2. **Live Match Win Predictor** — predicts the winner in real time during a run chase, using the live match situation (score, wickets, overs, target).

Built from `matches.csv` and `deliveries.csv` (IPL 2008–2024, ball-by-ball data). Includes a full Jupyter notebook (cleaning → EDA → feature engineering → modeling → tuning → SHAP explainability) and a multi-page Streamlit app for interactive predictions.

---

## 📌 Project Overview

This project deliberately reports **honest model performance** rather than inflated numbers:

| Predictor | Final Model | Test ROC-AUC | Why |
|---|---|---|---|
| Pre-Match | **Logistic Regression (tuned)**, deployed over the leaderboard-leading Decision Tree | **≈ 0.55** | T20 cricket outcomes are genuinely hard to call before a ball is bowled — this is a real, modest signal, not a bug. Decision Tree scored marginally higher on the leaderboard (0.530 vs 0.527 untuned) but its shallow tree produces only 25 leaves / 12 unique probability outputs across the *entire* feature space — many distinct team/venue/toss combinations collapsed onto identical predicted probabilities, which surfaced as a "stuck" Streamlit demo. Logistic Regression was deployed instead: statistically tied on AUC, but smooth and genuinely input-sensitive, and tuning improved it further to 0.549. See Section 6.3.1 of the notebook. |
| Live-Match | Gradient Boosting | **≈ 0.90–0.92** | Live match state (required run rate, wickets, overs) is a strong, direct predictor of outcome. |

This gap is the headline insight of the project: **pre-match prediction in T20 cricket is close to a coin flip with a real but small edge, while live-match prediction is highly reliable.** Both are reported as measured on a held-out test set — see the notebook for full methodology, including the match-level (not row-level) train/test split used for the live model to prevent data leakage between balls of the same match.

---

## 📂 Dataset Description

- **`matches.csv`** — 1,095 matches (2008–2024): teams, venue, city, toss, winner, result margin, etc.
- **`deliveries.csv`** — 260,920 ball-by-ball records: batting/bowling team, over, ball, runs, wickets, dismissals.

Both required cleaning before use:
- Team names standardized across franchise rebrands (e.g. Delhi Daredevils → Delhi Capitals, Kings XI Punjab → Punjab Kings).
- Venue names consolidated (many grounds were recorded under 2–3 different string variants).
- No-result/abandoned matches and rows with a null winner were dropped (no usable target).
- Super-over balls (innings 3–6 in `deliveries.csv`) excluded from live-match feature engineering, since they fall outside normal first/second-innings chase logic.

---

## 🏗️ Architecture

```
IPL_Winning_Team_Prediction/
│
├── notebooks/
│   └── IPL_Winning_Team_Prediction.ipynb   # Full pipeline, fully executed with outputs
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
├── app.py                                  # Multi-page Streamlit application
├── requirements.txt
└── README.md
```

Each model is saved as a **single joblib artifact** containing the entire `ColumnTransformer` + classifier pipeline — the app calls `.predict_proba()` directly on a raw feature row with no separate encoder/scaler bookkeeping needed at inference time. The `lookups.pkl` files store the point-in-time statistics (team win %, head-to-head records, venue chase rates, etc.) needed to construct that feature row from whatever the user selects in the UI.

---

## ✨ Features

### Pre-Match Predictor
- Inputs: Team 1, Team 2, Venue, City, Toss Winner, Toss Decision
- Engineered features: team win %, recent form (last 5 matches), head-to-head win %, venue/city-specific win %, toss impact
- Output: predicted winner, win probability for both teams, model confidence, plain-language explanation of the key drivers

### Live-Match Predictor
- Inputs: Batting Team, Bowling Team, Venue, City, Current Score, Wickets Lost, Overs Completed, Target Score
- Auto-calculated: runs remaining, balls remaining, wickets remaining, current/required run rate, run-rate differential
- Context features: venue average first-innings score, venue chase success rate, team chase/defend success %
- Output: predicted winner, live win probability, dynamic progress bar, automated match-situation insights

### Analytics Dashboard
- All-time team win %, toss-decision trends, toss-impact breakdown, venue chase-success rankings, season-over-season match volume

---

## 🔬 Model Development Approach

Rather than mechanically training every algorithm under the sun, the notebook trains **5–6 complementary, well-established models** per predictor (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and — for the live model — LightGBM), compares them on Accuracy / Precision / Recall / F1 / ROC-AUC, and tunes the best one with `RandomizedSearchCV`. CatBoost and Extra Trees were intentionally left out: on this dataset size, they converge to performance very close to the models already included, and adding them would mostly increase training time and dependency surface without adding new insight. This is documented directly in the notebook as a deliberate scoping decision.

**Leakage-prevention measures actually implemented:**
- Pre-match features (team win %, recent form, H2H, venue/city win %) are computed **point-in-time** — only using matches that occurred *before* the match being featurized, via a chronological pass with running counters.
- The live-match model is split into train/test **by `match_id`**, not by row — otherwise the model could see other balls from the same match during training and trivially "predict" the held-out balls of that same match.

---

## 🖥️ Installation & Setup

```bash
# 1. Clone / unzip the project, then create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## 📓 Running the Notebook

```bash
cd notebooks
jupyter notebook IPL_Winning_Team_Prediction.ipynb
```

The notebook runs top-to-bottom without modification (data paths are relative to `notebooks/`, e.g. `../data/matches.csv`). It will regenerate all files under `models/` if re-run.

> **Runtime note:** the full notebook (including hyperparameter search across both predictors, permutation importance, and SHAP) takes roughly 5–7 minutes to execute on a typical machine.

## 🚀 Running the Streamlit App

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. Use the sidebar to switch between the Pre-Match Predictor, Live Match Predictor, and Analytics Dashboard.

---

## 📊 Model Performance Summary

**Pre-Match Predictor** (5-model comparison, test set):

| Model | Accuracy | F1 | ROC-AUC | Deployed? |
|---|---|---|---|---|
| Decision Tree | 0.514 | 0.495 | 0.530 (leaderboard top) | ❌ — coarse probability output, see Limitations below |
| **Logistic Regression** | 0.514 | 0.527 | 0.527 untuned → **0.549 tuned** | ✅ Deployed |
| XGBoost | 0.477 | 0.509 | 0.513 | ❌ |
| Random Forest | 0.482 | 0.519 | 0.481 | ❌ |
| Gradient Boosting | 0.450 | 0.478 | 0.446 | ❌ |

**Live-Match Predictor** (6-model comparison, test set, match-level split):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Gradient Boosting | 0.828 | 0.846 | **0.919** |
| LightGBM | 0.818 | 0.837 | 0.914 |
| Random Forest | 0.800 | 0.822 | 0.901 |
| Logistic Regression | 0.780 | 0.805 | 0.884 |
| XGBoost | 0.788 | 0.812 | 0.871 |
| Decision Tree | 0.776 | 0.795 | 0.804 |

(Exact numbers will vary slightly on re-run due to the random train/test split, but the *gap* between pre-match and live-match performance is the stable, expected finding.)

**Top features (live-match model, via feature importance + permutation importance + SHAP):** required run rate, runs remaining, balls remaining, and current run rate dominate — exactly matching cricketing intuition. Venue and team chase/defend statistics provide secondary signal. Toss has no role in the live model, since it's already "priced in" by the time the chase has started.

---

## 🔮 Future Improvements

- Incorporate player-level features (current batters' historical strike rate, bowler economy) for a sharper live-match model.
- Use a chronological (not random) train/test split for the pre-match model to better simulate real deployment — predicting only future, unseen seasons.
- Add `GroupKFold` cross-validation (grouped by `match_id`) for the live model to get a more robust CV estimate without leakage, beyond the single match-level holdout used here.
- Extend the live model to weight by powerplay/middle-overs/death-overs phase, since run-rate dynamics differ meaningfully across these phases.
- Add CatBoost as an additional model comparison point if categorical-native boosting becomes a priority.

---

## ⚠️ Limitations & Honest Disclosures

- **A real bug was caught and fixed during manual app testing**: the pre-match model was initially deployed as a `DecisionTreeClassifier` (the leaderboard's top-AUC model). In practice, its shallow tree (`max_depth=5`) produced only 25 leaves and 12 unique probability outputs across the *entire* input space, so many genuinely different team/venue/toss combinations returned identical predicted probabilities — visible in the Streamlit app as confidence scores that appeared "stuck" across different inputs. This was not a caching or app-logic bug; the app was correctly feeding fresh inputs to a model that was itself too coarse. The fix was to deploy Logistic Regression instead (statistically tied on AUC, ~0.527 vs 0.530), which produces continuously-varying, input-sensitive probabilities. See notebook Section 6.3.1 for the full reasoning, and verify the fix yourself by running the same matchup across several different venues in the app — the win probability should now visibly shift each time.
- The pre-match model's ROC-AUC (~0.55) is only modestly better than random guessing (0.50). This is reported transparently rather than hidden — it reflects genuine cricket variance, not a modeling failure.
- The dataset has ~1,090 matches after cleaning — small enough that high-cardinality categorical features (venue, city) carry real overfitting risk, which is part of why the pre-match ceiling is what it is.
- Models were trained on a single random split; production deployment should use time-based validation (train on past seasons, test on the most recent season) for a more realistic estimate of forward-looking performance.
