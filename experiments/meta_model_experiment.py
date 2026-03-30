"""
Standalone meta-model experiment on top of the stable v4 baseline.

This script does not modify the main pipeline or final saved models.
It trains an OOF-safe meta model on top of the existing base-model OOF scores
and selected v4 features, then searches for a conservative mid-band blend.
"""

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURES_CACHE_ACTIVE, MODELS_DIR, OUTPUT_DIR, RANDOM_STATE, TEST_ACCOUNTS_PATH
from train_model import best_f1_threshold_grid


EXPERIMENT_DIR = Path(OUTPUT_DIR) / "meta_experiment"
EXPERIMENT_MODELS = EXPERIMENT_DIR / "models"

META_FEATURES = [
    "f_total_cp_connections",
    "f_unique_counterparties",
    "f_shared_counterparties",
    "f_graph_in_degree",
    "f_graph_out_degree",
    "f_ever_frozen",
    "f_account_frozen",
    "f_freeze_unfreeze",
    "f_log_branch_turnover",
    "f_days_since_kyc",
    "f_upi_fraction",
    "f_month_end_txn_ratio",
    "f_morning_ratio",
    "f_night_txn_ratio",
    "f_weekend_txn_ratio",
    "f_avg_mcc_anomaly",
    "f_near_zero_bal_ratio",
    "f_txn_acceleration",
    "f_passthrough_ratio",
    "f_structuring_ratio",
]


def ensure_dirs():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_MODELS.mkdir(parents=True, exist_ok=True)


def robust_scale(train_df, test_df):
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return train_scaled, test_scaled, scaler


def build_meta_frame(base_scores, feat_df):
    out = base_scores.copy()
    out["score_gap_lx"] = out["oof_lgb"] - out["oof_xgb"]
    out["score_gap_lc"] = out["oof_lgb"] - out["oof_cb"]
    out["score_gap_xc"] = out["oof_xgb"] - out["oof_cb"]
    out["score_max"] = out[["oof_lgb", "oof_xgb", "oof_cb"]].max(axis=1)
    out["score_min"] = out[["oof_lgb", "oof_xgb", "oof_cb"]].min(axis=1)
    out["score_range"] = out["score_max"] - out["score_min"]
    out["score_mean"] = out[["oof_lgb", "oof_xgb", "oof_cb"]].mean(axis=1)
    out["score_std"] = out[["oof_lgb", "oof_xgb", "oof_cb"]].std(axis=1).fillna(0)

    cols = ["account_id"] + [c for c in META_FEATURES if c in feat_df.columns]
    out = out.merge(feat_df[cols], on="account_id", how="left")
    return out.fillna(0)


def load_train_data():
    oof = pd.read_csv(Path(OUTPUT_DIR) / "oof_predictions.csv")
    feats = pd.read_parquet(FEATURES_CACHE_ACTIVE)
    df = build_meta_frame(oof, feats)
    y = df["is_mule"].astype(int).to_numpy()
    return df, y


def load_test_data():
    test_ids = pd.read_parquet(TEST_ACCOUNTS_PATH)[["account_id"]]
    feats = pd.read_parquet(FEATURES_CACHE_ACTIVE)

    lgb_models = joblib.load(Path(MODELS_DIR) / "lgb_models.pkl")
    xgb_models = joblib.load(Path(MODELS_DIR) / "xgb_models.pkl")
    try:
        cb_models = joblib.load(Path(MODELS_DIR) / "cb_models.pkl")
        has_cb = True
    except Exception:
        cb_models = []
        has_cb = False

    fcols = joblib.load(Path(MODELS_DIR) / "feature_cols.pkl")
    test_df = test_ids.merge(feats, on="account_id", how="left").fillna(0)
    for c in fcols:
        if c not in test_df.columns:
            test_df[c] = 0.0
    xtest = test_df[fcols].fillna(0).values.astype(np.float32)

    oof_like = pd.DataFrame({"account_id": test_df["account_id"].values})
    oof_like["oof_lgb"] = np.mean([m.predict_proba(xtest)[:, 1] for m in lgb_models], axis=0)
    oof_like["oof_xgb"] = np.mean([m.predict_proba(xtest)[:, 1] for m in xgb_models], axis=0)
    if has_cb:
        oof_like["oof_cb"] = np.mean([m.predict_proba(xtest)[:, 1] for m in cb_models], axis=0)
    else:
        oof_like["oof_cb"] = 0.0

    wa, wb, wc = joblib.load(Path(MODELS_DIR) / "ensemble_weights.pkl")
    raw = oof_like["oof_lgb"] * wa + oof_like["oof_xgb"] * wb + oof_like["oof_cb"] * wc
    oof_like["oof_score_raw"] = raw

    score_transform = joblib.load(Path(MODELS_DIR) / "score_transform.pkl")
    method = score_transform.get("method", "none")
    param = score_transform.get("param")
    model = score_transform.get("model")
    if method == "platt":
        raw = model.predict_proba(raw.to_numpy().reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        raw = model.predict(raw.to_numpy())
    elif method == "power":
        raw = np.clip(raw.to_numpy() ** float(param), 0.0, 1.0)
    else:
        raw = raw.to_numpy()

    oof_like["oof_score"] = raw
    test_meta = build_meta_frame(oof_like, feats)
    return test_meta


def get_meta_columns(df):
    skip = {"account_id", "is_mule"}
    return [c for c in df.columns if c not in skip]


def fit_oof_meta(train_df, y, meta_cols):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    meta_oof = np.zeros(len(y), dtype=np.float32)
    scalers = []
    models = []

    X = train_df[meta_cols].fillna(0)
    for tr_idx, va_idx in skf.split(X, y):
        xtr = X.iloc[tr_idx]
        xva = X.iloc[va_idx]
        ytr = y[tr_idx]

        xtr_scaled, xva_scaled, scaler = robust_scale(xtr, xva)
        model = LogisticRegression(
            C=0.2,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(xtr_scaled, ytr)
        meta_oof[va_idx] = model.predict_proba(xva_scaled)[:, 1]
        scalers.append(scaler)
        models.append(model)

    full_scaler = RobustScaler(quantile_range=(10.0, 90.0))
    full_x = full_scaler.fit_transform(X)
    final_model = LogisticRegression(
        C=0.2,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    final_model.fit(full_x, y)
    return meta_oof, final_model, full_scaler


def blend_midband(base_scores, meta_scores, low, high, weight):
    out = base_scores.copy()
    band = (base_scores >= low) & (base_scores <= high)
    out[band] = np.clip((1.0 - weight) * base_scores[band] + weight * meta_scores[band], 0.0, 1.0)
    return out


def search_best_blend(y, base_scores, meta_scores, baseline_auc, baseline_f1):
    best = {
        "auc": baseline_auc,
        "f1": baseline_f1,
        "threshold": None,
        "low": None,
        "high": None,
        "weight": None,
        "scores": base_scores.copy(),
    }
    for low in [0.35, 0.40, 0.45, 0.50]:
        for high in [0.75, 0.80, 0.85]:
            for weight in [0.05, 0.10, 0.15, 0.20, 0.25]:
                blended = blend_midband(base_scores, meta_scores, low, high, weight)
                auc = float(roc_auc_score(y, blended))
                thr, f1, prec, rec = best_f1_threshold_grid(y, blended)
                if (f1 > best["f1"]) or (f1 == best["f1"] and auc > best["auc"]):
                    best.update(
                        {
                            "auc": auc,
                            "f1": float(f1),
                            "threshold": float(thr),
                            "precision": float(prec),
                            "recall": float(rec),
                            "low": low,
                            "high": high,
                            "weight": weight,
                            "scores": blended,
                        }
                    )
    return best


def main():
    ensure_dirs()

    metrics = json.loads((Path(OUTPUT_DIR) / "metrics.json").read_text(encoding="utf-8"))
    baseline_auc = float(metrics["auc_roc"])
    baseline_f1 = float(metrics["best_f1"])

    train_df, y = load_train_data()
    meta_cols = get_meta_columns(train_df)
    meta_oof, final_model, full_scaler = fit_oof_meta(train_df, y, meta_cols)

    base_scores = train_df["oof_score"].to_numpy(dtype=float)
    meta_auc = float(roc_auc_score(y, meta_oof))
    meta_thr, meta_f1, meta_prec, meta_rec = best_f1_threshold_grid(y, meta_oof)

    best = search_best_blend(y, base_scores, meta_oof, baseline_auc, baseline_f1)

    test_df = load_test_data()
    test_x = full_scaler.transform(test_df[meta_cols].fillna(0))
    test_meta_scores = final_model.predict_proba(test_x)[:, 1]
    test_base_scores = test_df["oof_score"].to_numpy(dtype=float)

    if best["low"] is not None:
        final_test_scores = blend_midband(test_base_scores, test_meta_scores, best["low"], best["high"], best["weight"])
    else:
        final_test_scores = test_base_scores

    summary = {
        "baseline_auc": baseline_auc,
        "baseline_f1": baseline_f1,
        "meta_only_auc": meta_auc,
        "meta_only_f1": float(meta_f1),
        "meta_only_threshold": float(meta_thr),
        "meta_only_precision": float(meta_prec),
        "meta_only_recall": float(meta_rec),
        "best_blend_auc": best["auc"],
        "best_blend_f1": best["f1"],
        "best_blend_threshold": best["threshold"],
        "best_blend_precision": best.get("precision"),
        "best_blend_recall": best.get("recall"),
        "blend_low": best["low"],
        "blend_high": best["high"],
        "blend_weight": best["weight"],
        "meta_feature_count": len(meta_cols),
    }

    with open(EXPERIMENT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    oof_out = train_df[["account_id", "is_mule"]].copy()
    oof_out["base_score"] = base_scores
    oof_out["meta_score"] = meta_oof
    oof_out["blended_score"] = best["scores"]
    oof_out.to_csv(EXPERIMENT_DIR / "oof_meta_scores.csv", index=False)

    sub = pd.DataFrame(
        {
            "account_id": test_df["account_id"],
            "is_mule": final_test_scores,
        }
    )
    base_sub = pd.read_csv(Path(OUTPUT_DIR) / "submission.csv")
    if "suspicious_start" in base_sub.columns:
        sub["suspicious_start"] = base_sub["suspicious_start"].fillna("")
        sub["suspicious_end"] = base_sub["suspicious_end"].fillna("")
    else:
        sub["suspicious_start"] = ""
        sub["suspicious_end"] = ""
    sub.to_csv(EXPERIMENT_DIR / "submission_meta.csv", index=False)

    joblib.dump(final_model, EXPERIMENT_MODELS / "meta_model.pkl")
    joblib.dump(full_scaler, EXPERIMENT_MODELS / "meta_scaler.pkl")
    joblib.dump(meta_cols, EXPERIMENT_MODELS / "meta_feature_cols.pkl")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
