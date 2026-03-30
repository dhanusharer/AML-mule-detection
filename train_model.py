"""
AML Mule Detection - Model Training v7
======================================
Key upgrades over v6:
1. Optimises F1 using the competition-style fixed score grid
2. Searches score transforms that improve grid-F1 without giving up AUC
3. Saves transform metadata so inference uses the same score mapping
4. Records feature source for cleaner experiment tracking
"""
import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import *
from utils import log, safe_read


F1_GRID = np.round(np.arange(0.00, 1.01, 0.01), 2)


def get_feature_cols(df):
    leaky_prefixes = [
        "f_branch_mule", "f_branch_is_hotspot",
        "f_account_at_flagged", "f_account_at_top5",
        "f_n_mule_counterparty", "f_mule_flow",
        "f_mule_neighbor", "f_mule_txn",
        "f_has_mule", "f_log_mule",
        "f_cp2_", "f_cp_weighted", "f_cp_mean",
        "f_cp_max", "f_cp_n_hot", "f_cp_hot_cp",
        "f_cp_mule", "f_cp_n_mule",
        "f_gp_", "f_iter_", "f_br_alert_",
    ]
    all_fcols = [c for c in df.columns if c.startswith("f_")]
    safe_cols = [c for c in all_fcols if not any(c.startswith(p) for p in leaky_prefixes)]
    log.info(f"  Features: {len(all_fcols)} total -> {len(safe_cols)} safe "
             f"(removed {len(all_fcols) - len(safe_cols)} leaky)")

    # Re-allow safe branch features from v4.
    safe_exceptions = ["f_branch_mule_rate_v2", "f_branch_is_hotspot_v2"]
    safe_cols.extend([c for c in all_fcols if c in safe_exceptions])
    safe_cols = list(dict.fromkeys(safe_cols))
    return safe_cols


def get_sample_weights(labels_df, oof_scores=None):
    """
    Downweight known noisy label patterns without aggressively cleaning the set.
    """
    weights = np.ones(len(labels_df))

    if "mule_flag_date" in labels_df.columns:
        flag_dates = pd.to_datetime(labels_df["mule_flag_date"], errors="coerce")
        future_mask = (labels_df["is_mule"] == 1) & (flag_dates > pd.Timestamp("2025-06-30"))
        weights[future_mask.values] = 0.15
        log.info(f"  Stage-1: {future_mask.sum()} future-flagged mules downweighted (w=0.15)")

    if oof_scores is not None:
        y = labels_df["is_mule"].values
        suspicious_mule = (y == 1) & (oof_scores < 0.10)
        suspicious_legit = (y == 0) & (oof_scores > 0.90)

        weights[suspicious_mule] = np.minimum(weights[suspicious_mule], 0.25)
        weights[suspicious_legit] = np.minimum(weights[suspicious_legit], 0.30)
        log.info(f"  Stage-2: {suspicious_mule.sum()} suspicious mules, "
                 f"{suspicious_legit.sum()} suspicious legits downweighted")

    return weights


def apply_focal_weights(base_weights, y, oof_scores, gamma=1.5):
    """Upweight harder examples to improve recall on borderline mules."""
    if oof_scores is None:
        return base_weights
    p = np.clip(oof_scores, 1e-6, 1 - 1e-6)
    focal_w = np.where(y == 1, (1 - p) ** gamma, p ** gamma)
    focal_w = focal_w / (focal_w.mean() + 1e-9)
    combined = np.clip(base_weights * focal_w, 0.05, 5.0)
    log.info(f"  Focal weights (gamma={gamma}): mean={combined.mean():.3f}  max={combined.max():.3f}")
    return combined


def best_f1_threshold_exact(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def best_f1_threshold_grid(y_true, y_score, thresholds=F1_GRID):
    best_t = 0.0
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_t = float(thr)
            best_f1 = float(f1)
            best_prec = float(prec)
            best_rec = float(rec)

    return best_t, best_f1, best_prec, best_rec


def optimise_ensemble_weights(oof_lgb, oof_xgb, oof_cb, y):
    """
    Grid search over ensemble weights using the competition F1 sweep.
    """
    log.info("  Optimising ensemble weights for grid-F1 ...")
    best_w = (0.5, 0.25, 0.25)
    best_f1 = -1.0
    best_auc = -1.0

    has_cb = oof_cb is not None and oof_cb.sum() > 0

    for a in np.arange(0.25, 0.80, 0.05):
        for b in np.arange(0.10, 0.55, 0.05):
            if has_cb:
                c = round(1.0 - a - b, 3)
                if c < 0.05 or c > 0.50:
                    continue
                blend = oof_lgb * a + oof_xgb * b + oof_cb * c
                w = (round(a, 2), round(b, 2), round(c, 2))
            else:
                if abs(a + b - 1.0) > 0.01:
                    continue
                blend = oof_lgb * a + oof_xgb * b
                w = (round(a, 2), round(b, 2), 0.0)

            _, f1_grid, _, _ = best_f1_threshold_grid(y, blend)
            auc = roc_auc_score(y, blend)
            if (f1_grid > best_f1) or (f1_grid == best_f1 and auc > best_auc):
                best_f1 = f1_grid
                best_auc = auc
                best_w = w

    log.info(f"  Best weights -> LGB={best_w[0]}  XGB={best_w[1]}  CB={best_w[2]}  "
             f"GridF1={best_f1:.5f} | AUC={best_auc:.5f}")
    return best_w, best_f1


def fit_score_transform(train_scores, train_labels, method):
    if method == "none":
        return None
    if method == "platt":
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(train_scores.reshape(-1, 1), train_labels)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(train_scores, train_labels)
        return model
    raise ValueError(f"Unknown transform method: {method}")


def apply_score_transform(scores, method, model=None, power_alpha=1.0):
    scores = np.clip(scores, 0.0, 1.0)
    if method == "none":
        return scores
    if method == "platt":
        return model.predict_proba(scores.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return model.predict(scores)
    if method == "power":
        return np.clip(scores ** power_alpha, 0.0, 1.0)
    raise ValueError(f"Unknown transform method: {method}")


def select_score_transform(raw_oof_scores, y):
    """
    Choose a transform that lifts fixed-grid F1 while preserving ranking.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = [("none", None), ("platt", None), ("isotonic", None)]
    candidates.extend([("power", alpha) for alpha in [0.85, 0.9, 1.1, 1.2, 1.35, 1.5]])

    baseline_auc = roc_auc_score(y, raw_oof_scores)
    results = []

    for method, param in candidates:
        transformed = np.zeros_like(raw_oof_scores)
        if method in {"platt", "isotonic"}:
            for tr_idx, va_idx in skf.split(raw_oof_scores.reshape(-1, 1), y):
                model = fit_score_transform(raw_oof_scores[tr_idx], y[tr_idx], method)
                transformed[va_idx] = apply_score_transform(raw_oof_scores[va_idx], method, model=model)
        elif method == "power":
            transformed = apply_score_transform(raw_oof_scores, method, power_alpha=param)
        else:
            transformed = raw_oof_scores.copy()

        auc = roc_auc_score(y, transformed)
        thr, f1_grid, prec, rec = best_f1_threshold_grid(y, transformed)
        thr_exact, f1_exact = best_f1_threshold_exact(y, transformed)
        results.append({
            "method": method,
            "param": param,
            "scores": transformed,
            "auc": float(auc),
            "threshold": float(thr),
            "f1_grid": float(f1_grid),
            "f1_exact": float(f1_exact),
            "threshold_exact": float(thr_exact),
            "precision": float(prec),
            "recall": float(rec),
        })

    auc_floor = baseline_auc - 0.0005
    safe_results = [r for r in results if r["auc"] >= auc_floor]
    pool = safe_results if safe_results else results
    best = max(pool, key=lambda r: (r["f1_grid"], r["auc"], r["f1_exact"]))

    log.info("  Score transform search:")
    for r in sorted(results, key=lambda x: (-x["f1_grid"], -x["auc"])):
        label = r["method"] if r["param"] is None else f"{r['method']}({r['param']})"
        log.info(f"    {label:<14} AUC={r['auc']:.5f} | GridF1={r['f1_grid']:.5f} | "
                 f"ExactF1={r['f1_exact']:.5f} | t_grid={r['threshold']:.2f}")

    chosen = best["method"] if best["param"] is None else f"{best['method']}({best['param']})"
    log.info(f"  Chosen transform -> {chosen} | AUC={best['auc']:.5f} | GridF1={best['f1_grid']:.5f}")
    return best


def build_raw_blend(oof_lgb, oof_xgb, oof_cb, has_cb, weights):
    wa, wb, wc = weights
    if has_cb:
        return oof_lgb * wa + oof_xgb * wb + oof_cb * wc
    return oof_lgb * wa + oof_xgb * wb


def train():
    log.info(f"Loading features from {FEATURES_CACHE_ACTIVE}")
    feats = pd.read_parquet(FEATURES_CACHE_ACTIVE)
    log.info(f"  Shape: {feats.shape}")

    labels_full = safe_read(TRAIN_LABELS_PATH)
    fcols = get_feature_cols(feats)

    merged = feats.merge(labels_full, on="account_id", how="inner")
    X = merged[fcols].fillna(0).values.astype(np.float32)
    y = merged["is_mule"].values.astype(int)
    ids = merged["account_id"].values

    log.info(f"Training: {X.shape[0]:,} x {X.shape[1]} | mules: {y.sum():,} ({y.mean()*100:.2f}%)")

    log.info("\n--- Pass 1: base training (for label cleaning OOF scores) ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_lgb_p1 = np.zeros(len(y))
    oof_xgb_p1 = np.zeros(len(y))
    oof_cb_p1 = np.zeros(len(y))

    w_base = get_sample_weights(merged)
    lgb_models_p1, xgb_models_p1, cb_models_p1 = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        wtr = w_base[tr_idx]

        lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgb_model.fit(
            Xtr, ytr,
            sample_weight=wtr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
        )
        oof_lgb_p1[va_idx] = lgb_model.predict_proba(Xva)[:, 1]
        lgb_models_p1.append(lgb_model)

        scale_pos = max(1, (ytr == 0).sum() / max((ytr == 1).sum(), 1))
        xgb_model = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": scale_pos, "early_stopping_rounds": 100})
        xgb_model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
        oof_xgb_p1[va_idx] = xgb_model.predict_proba(Xva)[:, 1]
        xgb_models_p1.append(xgb_model)

        if HAS_CB:
            cb_model = CatBoostClassifier(**CB_PARAMS)
            cb_model.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xva, yva), early_stopping_rounds=100, verbose=False)
            oof_cb_p1[va_idx] = cb_model.predict_proba(Xva)[:, 1]
            cb_models_p1.append(cb_model)

        log.info(f"  Fold {fold} LGB AUC: {roc_auc_score(yva, oof_lgb_p1[va_idx]):.5f}")

    if HAS_CB and len(cb_models_p1):
        oof_p1 = oof_lgb_p1 * 0.5 + oof_xgb_p1 * 0.25 + oof_cb_p1 * 0.25
    else:
        oof_p1 = oof_lgb_p1 * 0.6 + oof_xgb_p1 * 0.4

    log.info("\n--- Pass 2: retraining with label cleaning + focal weights ---")
    w2 = get_sample_weights(merged, oof_scores=oof_p1)
    w2 = apply_focal_weights(w2, y, oof_p1, gamma=1.5)

    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cb = np.zeros(len(y))
    lgb_models, xgb_models, cb_models = [], [], []
    fi_list = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        log.info(f"\n{'=' * 50}\nFold {fold}/{N_FOLDS}\n{'=' * 50}")
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        wtr = w2[tr_idx]

        lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgb_model.fit(
            Xtr, ytr,
            sample_weight=wtr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
        )
        oof_lgb[va_idx] = lgb_model.predict_proba(Xva)[:, 1]
        lgb_models.append(lgb_model)
        fi_list.append(lgb_model.feature_importances_)
        log.info(f"  LGB AUC: {roc_auc_score(yva, oof_lgb[va_idx]):.5f}")

        scale_pos = max(1, (ytr == 0).sum() / max((ytr == 1).sum(), 1))
        xgb_model = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": scale_pos, "early_stopping_rounds": 100})
        xgb_model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
        oof_xgb[va_idx] = xgb_model.predict_proba(Xva)[:, 1]
        xgb_models.append(xgb_model)
        log.info(f"  XGB AUC: {roc_auc_score(yva, oof_xgb[va_idx]):.5f}")

        if HAS_CB:
            cb_model = CatBoostClassifier(**CB_PARAMS)
            cb_model.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xva, yva), early_stopping_rounds=100, verbose=False)
            oof_cb[va_idx] = cb_model.predict_proba(Xva)[:, 1]
            cb_models.append(cb_model)
            log.info(f"  CB  AUC: {roc_auc_score(yva, oof_cb[va_idx]):.5f}")

    has_cb = HAS_CB and len(cb_models) > 0
    best_w, _ = optimise_ensemble_weights(oof_lgb, oof_xgb, oof_cb if has_cb else None, y)
    raw_oof_final = build_raw_blend(oof_lgb, oof_xgb, oof_cb, has_cb, best_w)
    wa, wb, wc = best_w

    log.info("\nPseudo-labeling pass ...")
    try:
        test_df = safe_read(TEST_ACCOUNTS_PATH, columns=["account_id"])
        test_feats = pd.read_parquet(FEATURES_CACHE_ACTIVE)
        test_feats = test_df.merge(test_feats, on="account_id", how="left")
        for c in fcols:
            if c not in test_feats.columns:
                test_feats[c] = 0.0
        Xtest = test_feats[fcols].fillna(0).values.astype(np.float32)

        test_lgb = np.mean([m.predict_proba(Xtest)[:, 1] for m in lgb_models], axis=0)
        test_xgb = np.mean([m.predict_proba(Xtest)[:, 1] for m in xgb_models], axis=0)
        if has_cb:
            test_cb = np.mean([m.predict_proba(Xtest)[:, 1] for m in cb_models], axis=0)
            test_raw = build_raw_blend(test_lgb, test_xgb, test_cb, True, best_w)
        else:
            test_raw = build_raw_blend(test_lgb, test_xgb, np.zeros(len(test_lgb)), False, best_w)

        pseudo_mule = test_raw >= 0.75
        pseudo_legit = test_raw <= 0.04
        n_pm = pseudo_mule.sum()
        n_pl = pseudo_legit.sum()
        log.info(f"  Pseudo: {n_pm} mules + {n_pl} legit")

        if n_pm > 10:
            Xp = np.vstack([X, Xtest[pseudo_mule], Xtest[pseudo_legit]])
            yp = np.concatenate([y, np.ones(n_pm), np.zeros(n_pl)])
            wp = np.concatenate([w2, np.full(n_pm, 0.5), np.full(n_pl, 0.3)])

            log.info("  Retraining LGB with pseudo-labels ...")
            skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + 1)
            lgb_pseudo = []
            for tr2, va2 in skf2.split(Xp, yp):
                model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": 1000})
                model.fit(
                    Xp[tr2], yp[tr2],
                    sample_weight=wp[tr2],
                    eval_set=[(Xp[va2], yp[va2])],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
                )
                lgb_pseudo.append(model)
            lgb_models = lgb_pseudo
            log.info(f"  Pseudo done -> {len(lgb_models)} LGB models")
    except Exception as exc:
        log.warning(f"  Pseudo-labeling skipped: {exc}")

    transform_result = select_score_transform(raw_oof_final, y)
    transform_method = transform_result["method"]
    transform_param = transform_result["param"]
    oof_final = transform_result["scores"]

    if transform_method in {"platt", "isotonic"}:
        score_transform = fit_score_transform(raw_oof_final, y, transform_method)
    else:
        score_transform = None

    auc_final = roc_auc_score(y, oof_final)
    best_t_exact, best_f1_exact = best_f1_threshold_exact(y, oof_final)
    best_t_grid, best_f1_grid, prec, rec = best_f1_threshold_grid(y, oof_final)

    log.info("\n" + "=" * 50)
    log.info("OOF FINAL METRICS:")
    log.info(f"  AUC-ROC        : {auc_final:.5f}")
    log.info(f"  Best F1 (grid) : {best_f1_grid:.5f}  @ threshold {best_t_grid:.2f}")
    log.info(f"  Best F1 (exact): {best_f1_exact:.5f}  @ threshold {best_t_exact:.4f}")
    log.info(f"  Precision      : {prec:.5f}")
    log.info(f"  Recall         : {rec:.5f}")
    log.info(f"  Ensemble wts   : LGB={wa}  XGB={wb}  CB={wc}")
    if transform_param is None:
        log.info(f"  Score xform    : {transform_method}")
    else:
        log.info(f"  Score xform    : {transform_method} ({transform_param})")
    log.info("=" * 50)

    fi_df = pd.DataFrame({"feature": fcols, "importance": np.mean(fi_list, axis=0)}).sort_values("importance", ascending=False)
    fi_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
    log.info(f"\nTop 20 features:\n{fi_df.head(20).to_string()}")

    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(lgb_models[0])
            shap_values = explainer.shap_values(X[:2000])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_df = pd.DataFrame({
                "feature": fcols,
                "shap_mean": np.abs(shap_values).mean(axis=0),
            }).sort_values("shap_mean", ascending=False)
            shap_df.to_csv(f"{OUTPUT_DIR}/shap_importance.csv", index=False)
            log.info(f"Top 15 SHAP:\n{shap_df.head(15).to_string()}")
        except Exception as exc:
            log.warning(f"SHAP failed: {exc}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lgb_models, f"{MODELS_DIR}/lgb_models.pkl")
    joblib.dump(xgb_models, f"{MODELS_DIR}/xgb_models.pkl")
    joblib.dump(fcols, f"{MODELS_DIR}/feature_cols.pkl")
    joblib.dump(best_t_grid, f"{MODELS_DIR}/best_threshold.pkl")
    joblib.dump(best_w, f"{MODELS_DIR}/ensemble_weights.pkl")
    joblib.dump({
        "method": transform_method,
        "param": transform_param,
        "model": score_transform,
    }, f"{MODELS_DIR}/score_transform.pkl")
    if has_cb:
        joblib.dump(cb_models, f"{MODELS_DIR}/cb_models.pkl")

    oof_df = pd.DataFrame({
        "account_id": ids,
        "oof_lgb": oof_lgb,
        "oof_xgb": oof_xgb,
        "oof_cb": oof_cb,
        "oof_score_raw": raw_oof_final,
        "oof_score": oof_final,
        "is_mule": y,
    })
    oof_df.to_csv(f"{OUTPUT_DIR}/oof_predictions.csv", index=False)

    metrics = {
        "auc_roc": round(float(auc_final), 5),
        "best_f1": round(float(best_f1_grid), 5),
        "best_f1_exact": round(float(best_f1_exact), 5),
        "best_threshold": round(float(best_t_grid), 2),
        "ensemble_lgb": float(wa),
        "ensemble_xgb": float(wb),
        "ensemble_cb": float(wc),
        "score_transform": transform_method if transform_param is None else f"{transform_method}({transform_param})",
        "feature_source": FEATURES_CACHE_ACTIVE,
        "n_train": int(len(y)),
        "n_mule": int(y.sum()),
        "n_features": len(fcols),
    }
    with open(f"{OUTPUT_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info("All models saved!")
    log.info("Now run: python make_submission.py")


if __name__ == "__main__":
    train()
