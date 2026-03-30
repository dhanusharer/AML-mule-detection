"""
build_network_features.py v4 — OOF-based CP Bipartite Features
===============================================================
REQUIRES: output/oof_predictions.csv (from train_model.py first pass)

Run order:
  1. python train_model.py
  2. python -m feature_builders.build_network_features
  3. python train_model.py
  4. python make_submission.py

Why OOF and not hard labels:
  Hard labels (0/1) cause leakage in cross-validation.
  OOF scores are out-of-sample probabilities — same distribution as test.
  CP mule scores built from OOF generalize properly to test accounts.
"""
import warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict

from config import *
from utils import log, safe_read


def main():
    log.info("=" * 65)
    log.info("  CP Bipartite Network Features v4 — OOF-based")
    log.info("=" * 65)

    # Load OOF scores from first train pass
    oof_path = f"{OUTPUT_DIR}/oof_predictions.csv"
    log.info(f"Loading OOF: {oof_path}")
    oof_df = pd.read_csv(oof_path, usecols=["account_id", "oof_score"])
    oof_df["oof_score"] = oof_df["oof_score"].clip(0, 1)
    acct_mule_prob = dict(zip(oof_df["account_id"], oof_df["oof_score"]))
    log.info(f"  OOF accounts: {len(acct_mule_prob):,} | "
             f"mean={oof_df['oof_score'].mean():.4f} | "
             f"p95={oof_df['oof_score'].quantile(0.95):.4f}")

    test_df    = safe_read(TEST_ACCOUNTS_PATH, columns=["account_id"])
    test_accts = set(test_df["account_id"].tolist())
    all_accts  = set(acct_mule_prob.keys()) | test_accts
    log.info(f"All accounts: {len(all_accts):,}")

    txn_files = sorted(glob(TRANSACTIONS_GLOB))

    # Build CP bipartite graph
    log.info("Pass 1: Building CP <-> Account graph (OOF-weighted)...")
    cp_mule_vol      = defaultdict(float)
    cp_total_vol     = defaultdict(float)
    cp_mule_prob_sum = defaultdict(float)
    cp_acct_count    = defaultdict(int)
    cp_accounts      = defaultdict(set)
    acct_to_cps      = defaultdict(set)
    acct_cp_vol      = dict()

    for fi, fpath in enumerate(txn_files):
        try:
            df = pd.read_parquet(fpath, columns=["account_id","counterparty_id","amount"])
            df = df.dropna(subset=["counterparty_id"])
            df["vol"] = df["amount"].abs()
            for row in df.itertuples(index=False):
                acct = row.account_id
                cp   = row.counterparty_id
                vol  = row.vol
                prob = acct_mule_prob.get(acct, 0.0)
                cp_mule_vol[cp]      += vol * prob
                cp_total_vol[cp]     += vol
                cp_mule_prob_sum[cp] += prob
                cp_acct_count[cp]    += 1
                cp_accounts[cp].add(acct)
                acct_to_cps[acct].add(cp)
                key = (acct, cp)
                if key in acct_cp_vol:
                    acct_cp_vol[key] += vol
                else:
                    acct_cp_vol[key] = vol
            del df; gc.collect()
        except Exception:
            pass
        if (fi+1) % 100 == 0:
            log.info(f"  {fi+1}/{len(txn_files)} | CPs: {len(cp_total_vol):,}")

    log.info(f"Total unique CPs: {len(cp_total_vol):,}")

    # CP mule scores
    log.info("Computing CP mule scores...")
    cp_mule_score = {
        cp: cp_mule_vol[cp] / (cp_total_vol[cp] + 1e-9)
        for cp in cp_total_vol
    }
    cp_avg_prob = {
        cp: cp_mule_prob_sum[cp] / max(len(cp_accounts[cp]), 1)
        for cp in cp_total_vol
    }

    # Hot CP threshold = 90th percentile of mule scores
    all_scores    = list(cp_mule_score.values())
    hot_threshold = np.percentile(all_scores, 90)
    hot_cps       = {cp for cp, s in cp_mule_score.items() if s >= hot_threshold}
    log.info(f"Hot CPs (top 10%, threshold={hot_threshold:.4f}): {len(hot_cps):,}")

    # Per-account features
    log.info("Pass 2: Building per-account features...")
    rows = []
    zero_row = {
        "f_cp2_n_hot_cps":           0,
        "f_cp2_hot_cp_ratio":        0.0,
        "f_cp2_max_mule_score":      0.0,
        "f_cp2_mean_mule_score":     0.0,
        "f_cp2_top3_mule_score":     0.0,
        "f_cp2_mule_vol":            0.0,
        "f_cp2_mule_vol_ratio":      0.0,
        "f_cp2_log_mule_vol":        0.0,
        "f_cp2_weighted_mule_score": 0.0,
        "f_cp2_has_hot_cp":          0,
        "f_cp2_n_cps":               0,
        "f_cp2_avg_cp_mule_score":   0.0,
    }

    for acct in all_accts:
        my_cps = acct_to_cps.get(acct, set())
        if not my_cps:
            rows.append({"account_id": acct, **zero_row})
            continue

        my_hot = my_cps & hot_cps
        n_hot  = len(my_hot)
        n_cps  = len(my_cps)

        sc = sorted([cp_mule_score.get(cp, 0) for cp in my_cps], reverse=True)
        max_score  = sc[0]
        mean_score = float(np.mean(sc))
        top3_score = float(np.mean(sc[:3]))

        mule_vol  = sum(acct_cp_vol.get((acct, cp), 0) for cp in my_hot)
        total_vol = sum(acct_cp_vol.get((acct, cp), 0) for cp in my_cps) + 1e-9
        weighted  = sum(
            acct_cp_vol.get((acct, cp), 0) * cp_mule_score.get(cp, 0)
            for cp in my_cps
        ) / total_vol
        avg_cp = float(np.mean([cp_avg_prob.get(cp, 0) for cp in my_cps]))

        rows.append({
            "account_id":                acct,
            "f_cp2_n_hot_cps":           n_hot,
            "f_cp2_hot_cp_ratio":        n_hot / (n_cps + 1e-9),
            "f_cp2_max_mule_score":      max_score,
            "f_cp2_mean_mule_score":     mean_score,
            "f_cp2_top3_mule_score":     top3_score,
            "f_cp2_mule_vol":            mule_vol,
            "f_cp2_mule_vol_ratio":      mule_vol / total_vol,
            "f_cp2_log_mule_vol":        float(np.log1p(mule_vol)),
            "f_cp2_weighted_mule_score": weighted,
            "f_cp2_has_hot_cp":          int(n_hot > 0),
            "f_cp2_n_cps":               n_cps,
            "f_cp2_avg_cp_mule_score":   avg_cp,
        })

    net_df = pd.DataFrame(rows)
    log.info(f"Network features: {net_df.shape}")

    # Signal check
    labels  = safe_read(TRAIN_LABELS_PATH, columns=["account_id","is_mule"])
    check   = net_df.merge(labels, on="account_id", how="inner")
    mules_  = check[check["is_mule"]==1]
    legits_ = check[check["is_mule"]==0]
    sep = mules_["f_cp2_weighted_mule_score"].mean() / (legits_["f_cp2_weighted_mule_score"].mean() + 1e-9)
    log.info(f"\n=== SIGNAL CHECK ===")
    log.info(f"Mules  weighted_mule_score: {mules_['f_cp2_weighted_mule_score'].mean():.4f}")
    log.info(f"Legits weighted_mule_score: {legits_['f_cp2_weighted_mule_score'].mean():.4f}")
    log.info(f"Separation: {sep:.1f}x  ← should be >3x to be useful")

    # Merge with v3 base
    base = pd.read_parquet("features/all_features_v3.parquet")
    old  = [c for c in base.columns if c.startswith("f_cp") or c.startswith("f_net_")
            or c in ["f_branch_mule_rate","f_branch_mule_count"]]
    if old:
        base = base.drop(columns=old)
        log.info(f"Dropped old: {old}")

    merged = base.merge(net_df, on="account_id", how="left")
    new_cols = [c for c in net_df.columns if c != "account_id"]
    merged[new_cols] = merged[new_cols].fillna(0)

    out_path = "features/all_features_v4.parquet"
    merged.to_parquet(out_path, index=False)
    log.info(f"\nSaved → {out_path} | Shape: {merged.shape}")
    log.info("Now run: python train_model.py  (2nd pass with v4 features)")


if __name__ == "__main__":
    main()
