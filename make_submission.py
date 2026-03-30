"""
make_submission.py — FIXED (auto feature-col matching)
"""
import warnings, os, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from glob import glob
from collections import defaultdict
from datetime import timedelta

from config import *
from utils import log, safe_read

log.info("=" * 60)
log.info("  AML -- Submission Generator  v2 IMPROVED")
log.info("=" * 60)

# -- Step 1: Load features (always use active v4)
feats   = pd.read_parquet(FEATURES_CACHE_ACTIVE)
test_df = safe_read(TEST_ACCOUNTS_PATH, columns=["account_id"])

# -- Step 2: Load models FIRST to get exact feature count
log.info("Loading saved models ...")
lgb_models = joblib.load(os.path.join(MODELS_DIR, "lgb_models.pkl"))
xgb_models = joblib.load(os.path.join(MODELS_DIR, "xgb_models.pkl"))
try:
    cb_models = joblib.load(os.path.join(MODELS_DIR, "cb_models.pkl"))
    HAS_CB = True
except Exception:
    cb_models = []
    HAS_CB = False

# -- Step 3: Build feature cols to match model exactly
n_expected = lgb_models[0].n_features_

# Try saved fcols first
try:
    fcols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    fcols = [c for c in fcols if c in feats.columns]
except:
    fcols = []

# If stale, rebuild from parquet
if len(fcols) != n_expected:
    log.info(f"  feature_cols.pkl has {len(fcols)} cols but model expects {n_expected} — rebuilding ...")
    LEAKY_PFX = [
        "f_branch_mule_count", "f_account_at_flagged", "f_account_at_top5",
        "f_n_mule_", "f_mule_flow", "f_mule_neighbor", "f_mule_txn",
        "f_has_mule", "f_log_mule", "f_cp2_", "f_cp_weighted", "f_cp_mean",
        "f_cp_max", "f_cp_n_hot", "f_cp_hot_cp", "f_cp_mule", "f_cp_n_mule",
        "f_br_alert_", "f_gp_", "f_iter_",
    ]
    ZERO_IMP = {
        "f_pin_mismatch", "f_ip_per_txn", "f_clt_cash_fraction",
        "f_loan_txn_fraction", "f_lon_range", "f_sudden_activation",
        "f_fast_drain_ratio", "f_geo_lon_std", "f_geo_unique_locs",
        "f_bal_range", "f_ip_prefix_nuniq_v2", "f_bal_near_zero_v3",
    }
    all_f = [c for c in feats.columns if c.startswith("f_")]
    fcols = [c for c in all_f
             if c not in ZERO_IMP
             and not any(c.startswith(p) for p in LEAKY_PFX)]

log.info(f"Features: {feats.shape} | Using {len(fcols)} cols | Test: {len(test_df):,}")

test_merged = test_df.merge(feats, on="account_id", how="left").fillna(0)
Xtest = test_merged[fcols].fillna(0).values.astype(np.float32)

log.info(f"  LGB x{len(lgb_models)} + XGB x{len(xgb_models)}" +
         (f" + CB x{len(cb_models)}" if HAS_CB else ""))

# Load optimised ensemble weights
try:
    wa, wb, wc = joblib.load(os.path.join(MODELS_DIR, "ensemble_weights.pkl"))
    log.info(f"  Loaded optimised weights: LGB={wa}  XGB={wb}  CB={wc}")
except:
    wa, wb, wc = 0.6, 0.35, 0.05

try:
    score_transform = joblib.load(os.path.join(MODELS_DIR, "score_transform.pkl"))
    log.info(f"  Loaded score transform: {score_transform.get('method')}")
except:
    score_transform = {"method": "none", "param": None, "model": None}

# -- Step 4: Score
lgb_s = np.mean([m.predict_proba(Xtest)[:, 1] for m in lgb_models], axis=0)
xgb_s = np.mean([m.predict_proba(Xtest)[:, 1] for m in xgb_models], axis=0)
if HAS_CB:
    cb_s  = np.mean([m.predict_proba(Xtest)[:, 1] for m in cb_models], axis=0)
    raw   = lgb_s * wa + xgb_s * wb + cb_s * wc
else:
    raw   = lgb_s * wa + xgb_s * (1 - wa)

method = score_transform.get("method", "none")
param  = score_transform.get("param")
model  = score_transform.get("model")
if method == "platt":
    raw = model.predict_proba(raw.reshape(-1, 1))[:, 1]
elif method == "isotonic":
    raw = model.predict(raw)
elif method == "power":
    raw = np.clip(raw ** float(param), 0.0, 1.0)

# Load best threshold
try:
    best_t = float(joblib.load(os.path.join(MODELS_DIR, "best_threshold.pkl")))
except:
    best_t = 0.59

log.info(f"Scores: min={raw.min():.4f}  p50={np.median(raw):.4f}  "
         f"p97={np.percentile(raw,97):.4f}  max={raw.max():.4f}")

# Auto N_MULES: find natural gap in score distribution
sorted_scores = np.sort(raw)[::-1]
gaps = np.diff(sorted_scores)
gap_idx = np.argmin(gaps[:3000])   # look for biggest gap in top 3000
auto_n  = gap_idx + 1
N_MULES = max(1200, min(auto_n, 2000))
log.info(f"Auto N_MULES={auto_n} → using N_MULES={N_MULES}  threshold={sorted_scores[N_MULES-1]:.4f}")

threshold   = float(sorted_scores[N_MULES - 1])
account_ids = test_merged["account_id"].values
mule_ids    = set(account_ids[raw >= threshold].tolist())
log.info(f"Mules selected: {len(mule_ids):,}")

# -- Step 5: Load transactions for mules
log.info(f"Scanning transactions for {len(mule_ids):,} mules ...")
txn_files = sorted(glob(TRANSACTIONS_GLOB))
acct_txns = defaultdict(list)

for fi, fpath in enumerate(txn_files):
    try:
        df = pd.read_parquet(fpath, columns=["account_id","transaction_timestamp","amount"])
        df = df[df["account_id"].isin(mule_ids)]
        if df.empty:
            del df; continue
        df["ts"]  = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
        df["vol"] = df["amount"].abs()
        df = df.dropna(subset=["ts"])
        for acct, grp in df.groupby("account_id"):
            acct_txns[acct].extend(zip(grp["ts"].tolist(), grp["vol"].tolist()))
        del df; gc.collect()
    except Exception:
        pass
    if (fi+1) % 100 == 0:
        log.info(f"  {fi+1}/{len(txn_files)} files | {len(acct_txns)} accounts loaded")

log.info(f"Transaction data ready for {len(acct_txns):,} mules")


# -- Step 6: Peak window detection
def find_peak_window(txn_list):
    if not txn_list:
        return None, None
    txn_list = sorted(txn_list, key=lambda x: x[0])
    times = [t for t, v in txn_list]
    vols  = [v for t, v in txn_list]
    n     = len(times)
    if n == 1:
        return times[0], times[0]
    if n == 2:
        return times[0], times[1]

    best_s = best_e = times[0]
    best_score = -1.0

    # Search a couple of compact activity windows and prefer dense, high-volume bursts.
    for window_days in [21, 30]:
        window = timedelta(days=window_days)
        for i in range(n):
            cutoff = times[i] + window
            j, wv = i, 0.0
            while j < n and times[j] <= cutoff:
                wv += vols[j]
                j += 1
            cnt = j - i
            if cnt < 2:
                continue
            actual_days = max((times[j - 1] - times[i]).days, 1)
            score = (cnt ** 1.5) * np.log1p(wv) / actual_days
            if score > best_score:
                best_score = score
                best_s = times[i]
                best_e = times[j - 1]

    if (best_e - best_s).days > 30:
        best_e = best_s + timedelta(days=30)
    return best_s, best_e


# -- Step 7: Build submission
log.info("Building submission rows ...")
rows = []
windows_found = 0
window_days_list = []

for i, acct in enumerate(account_ids):
    score     = float(raw[i])
    start_str = ""
    end_str   = ""
    if acct in mule_ids and acct in acct_txns:
        ws, we = find_peak_window(acct_txns[acct])
        if ws is not None:
            start_str = ws.strftime("%Y-%m-%dT%H:00:00")
            end_str   = we.strftime("%Y-%m-%dT%H:00:00")
            windows_found += 1
            window_days_list.append((we - ws).days)
    rows.append({
        "account_id":       acct,
        "is_mule":          round(score, 6),
        "suspicious_start": start_str,
        "suspicious_end":   end_str,
    })

sub = pd.DataFrame(rows)
sub["suspicious_start"] = sub["suspicious_start"].fillna("")
sub["suspicious_end"]   = sub["suspicious_end"].fillna("")

log.info(f"\nMules predicted : {len(mule_ids)}")
log.info(f"Windows found   : {windows_found}")
if window_days_list:
    log.info(f"Avg window days : {np.mean(window_days_list):.1f}")
    log.info(f"Min/Max days    : {min(window_days_list)} / {max(window_days_list)}")
log.info(f"NaN count       : {sub.isnull().sum().sum()}")
log.info(f"Total rows      : {len(sub):,}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
log.info(f"\nSaved → {out_path}")
log.info("\nSample mule rows:")
log.info(sub[sub["suspicious_start"] != ""].head(5).to_string())
log.info("\nDONE -- upload output/submission.csv to portal") 
