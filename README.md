# FinShield — AML Mule Account Detection System
## Team TRUST BUILDERS | RBIH × IIT Delhi National Fraud Prevention Challenge 2026

---

## Live Dashboard
**[FinShield Analytics Dashboard](https://trust-guard-ai-seven.vercel.app/)**
> Interactive dashboard showing model performance, feature importance, and mule detection insights.

## External Links
| Resource | Link | Notes |
|----------|------|-------|
| Live Dashboard | https://trust-guard-ai-seven.vercel.app/ | Final analytics dashboard |
| GitHub (older experiments) | https://github.com/dhanusharer/rbih | Earlier iterations — NOT final code |
| Architecture Video | https://youtu.be/rQGXH7i91zk?si=dl8quyYQ2VJrjorv | Pipeline walkthrough — some parts are older iterations |

---

## Final Results

| Metric | Public Score | Private Score |
|--------|-------------|---------------|
| AUC-ROC | 0.9819 | 0.9813 |
| F1 Score | 0.8247 | 0.8151 |
| Temporal IoU | 0.2281 | 0.2392 |
| RH Avoidance (avg) | — | 0.9343 |
| RH_Avoidance_6 | — | **0.9968** |

---

## Quickstart — Run Everything In One Command

```bash
python pipeline.py
```

This single command runs the complete pipeline:
1. Base feature engineering
2. Advanced behavioral features
3. Geo / IP / balance features
4. OOF-safe counterparty graph features
5. Alert reason mule type features
6. Model training (LGB + XGB + CatBoost ensemble)
7. Submission generation
8. XAI explainability report

Output: `output/submission.csv`

---

## Step-By-Step (if running manually)

```bash
# Step 1: Feature engineering
python build_features.py
python add_advanced_features.py
python add_geo_features.py
python add_oof_cp_features.py       # KEY INNOVATION: OOF-safe graph features
python add_alert_features.py

# Step 2: Train models
python train_model.py

# Step 3: Generate submission
python make_submission.py

# Step 4: Explainability
python explain_xai.py
```

---

## Environment Setup

### Python Version
Python 3.11 recommended

### Install Dependencies
```bash
pip install -r Requirements.txt
```

### Main Dependencies
```
pandas
numpy
pyarrow
lightgbm
xgboost
catboost
scikit-learn
shap
joblib
matplotlib
```

> Important: Do NOT include `.venv` in the ZIP submission.

---

## Project Structure

```
RBIH_FINAL/
├── pipeline.py                  # Single command to run everything
├── config.py                    # Central configuration
├── utils.py                     # Logging and helpers
├── build_features.py            # Base feature engineering
├── add_advanced_features.py     # Burst, velocity, CP diversity
├── add_geo_features.py          # Geo, IP, balance features
├── add_oof_cp_features.py       # OOF-safe CP mule rate (KEY INNOVATION)
├── add_alert_features.py        # Alert reason type features
├── train_model.py               # LGB + XGB + CatBoost ensemble training
├── make_submission.py           # Inference and submission generation
├── explain_xai.py               # SHAP explainability report
├── generate_architecture_diagram.py
├── feature_builders/            # Modular feature builder components
├── models/                      # Trained model artifacts
│   ├── lgb_models.pkl
│   ├── xgb_models.pkl
│   ├── cb_models.pkl
│   ├── feature_cols.pkl
│   ├── best_threshold.pkl
│   ├── meta_model.pkl
│   └── meta_scaler.pkl
├── experiments/                 # Experiment history
├── archive/legacy_source/       # Older code iterations (for reference)
├── output/                      # Generated outputs
│   ├── metrics.json
│   ├── feature_importance.csv
│   ├── oof_predictions.csv
│   └── xai/xai_report.txt
├── Requirements.txt
└── README.md
```

> Note: `data/`, `features/`, `.venv/`, and `output/submission.csv` are NOT included in the ZIP.

---

## Approach Summary

FinShield is a production-grade AML mule detection system built on:

### 1. Feature Engineering (130+ features)
- **Account profile** — KYC staleness, freeze history, account age
- **Transaction behavior** — structuring, pass-through, burst index, fan-in/fan-out
- **Graph network** — counterparty connections, shared CPs, OOF-safe mule CP rate
- **Geographic** — geo spread, IP diversity, location anomalies
- **MCC anomaly** — transaction amounts vs merchant category norms
- **Balance patterns** — near-zero balance ratio, volatility

### 2. Key Innovation: OOF-Safe Graph Features
Standard counterparty mule rate features leak training labels into validation folds — OOF AUC inflates to 0.999 but test AUC collapses to 0.849. Our **fold-aware computation** uses only out-of-fold mules to score validation accounts, preventing leakage while preserving signal. This improved F1 from 0.796 → 0.824.

### 3. Ensemble Model (15 models)
- LightGBM × 5 folds
- XGBoost × 5 folds
- CatBoost × 5 folds
- Weighted ensemble: LGB×0.50 + XGB×0.25 + CB×0.25
- Pseudo-labeling on high-confidence test predictions

### 4. Red Herring Avoidance
| Red Herring | Our Solution |
|-------------|-------------|
| Future flag dates (754 mules) | Downweight to 0.15 |
| Branch code leakage | Excluded entirely |
| OOF label leakage in graph | Fold-aware OOF computation |
| Aggressive label removal | Sample weights instead |
| Flag date window anchoring | Pure transaction density |

### 5. Temporal Window Detection
Sliding 30-day window scoring: `0.6 × (count/total) + 0.4 × (volume/total)`
Detected 831/960 true mule windows (86.6% coverage).

---

## Manual Explanation for Reviewers

FinShield addresses the fundamental challenge of AML mule detection at national banking scale. Our approach is distinguished by three contributions:

**1. OOF-Safe Graph Propagation:** We discovered that naive counterparty mule rate features leak validation labels, creating artificially perfect OOF scores that collapse on test data. Our fold-aware computation is the first systematic solution to this leakage pattern, improving real-world generalization.

**2. Leakage Detection Heuristic:** We established that OOF AUC > 0.97 reliably indicates label leakage in this dataset. This heuristic guided all feature engineering decisions and prevented multiple iterations from producing misleading results.

**3. Systematic Red Herring Identification:** We identified and mitigated 5 distinct red herring patterns including future-dated mule flags, branch code leakage, and graph label contamination — achieving average RH avoidance of 0.9343 with near-perfect 0.9968 on RH category 6.

The near-zero public/private AUC gap (0.9819 vs 0.9813) confirms genuine generalization rather than overfitting.

---

## Notes for Reviewers

- `models/` contains final trained weights used for the submitted prediction
- `experiments/` and `archive/legacy_source/` contain older iterations for reference
- GitHub repo (https://github.com/dhanusharer/rbih) contains earlier experimental code — **NOT the final solution**
- Architecture video (https://youtu.be/rQGXH7i91zk) shows pipeline design — some components are from earlier iterations
- OOF AUC of ~0.952 is intentional — we used OOF AUC > 0.97 as a leakage warning signal
- Dashboard: https://trust-guard-ai-seven.vercel.app/

---

## Build Submission ZIP

```bash
python build_code_submission.py
```

Produces a clean ZIP in `submission_package/` excluding data, features, venv, and cache.

---

*Team TRUST BUILDERS — FinShield AML Detection — RBIH × IIT Delhi 2026*
