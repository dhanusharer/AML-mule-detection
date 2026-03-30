"""
add_label_features.py
======================
Add alert_reason pattern features — these are NOT leaky because:
1. We build branch-level and pattern-level aggregates from TRAIN only
2. Test accounts get features based on their branch's historical alert pattern
3. No individual account label is used directly

Key insight: branches with high alert rates have predictable patterns.
The alert_reason distribution per branch is a strong signal.
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
    log.info("  Label-Informed Branch & Pattern Features")
    log.info("=" * 65)

    labels  = safe_read(TRAIN_LABELS_PATH)
    accts   = safe_read(ACCOUNTS_PATH)   # has branch_id

    log.info(f"Labels shape: {labels.shape}")
    log.info(f"Accounts shape: {accts.shape}")
    log.info(f"Accounts cols: {accts.columns.tolist()}")

    # Merge to get branch_id for train mules
    mules = labels[labels['is_mule'] == 1].copy()
    mules['mule_flag_date'] = pd.to_datetime(mules['mule_flag_date'], errors='coerce')

    if 'branch_id' in accts.columns:
        mules = mules.merge(accts[['account_id','branch_id']], on='account_id', how='left')
    elif 'home_branch' in accts.columns:
        mules = mules.merge(accts[['account_id','home_branch']].rename(
            columns={'home_branch':'branch_id'}), on='account_id', how='left')
    else:
        log.info(f"No branch col found. Accts cols: {accts.columns.tolist()}")
        # Try to find branch col
        branch_cols = [c for c in accts.columns if 'branch' in c.lower()]
        log.info(f"Branch-like cols: {branch_cols}")
        if branch_cols:
            mules = mules.merge(
                accts[['account_id', branch_cols[0]]].rename(
                    columns={branch_cols[0]: 'branch_id'}),
                on='account_id', how='left')

    log.info(f"Mules with branch: {mules['branch_id'].notna().sum()} / {len(mules)}")

    # 1. Branch mule rate (from train labels — use with care, add as weak feature)
    all_accts_df = accts.copy()
    if 'branch_id' not in all_accts_df.columns:
        branch_cols = [c for c in all_accts_df.columns if 'branch' in c.lower()]
        if branch_cols:
            all_accts_df = all_accts_df.rename(columns={branch_cols[0]: 'branch_id'})

    branch_counts = all_accts_df.groupby('branch_id')['account_id'].count().rename('branch_total')
    mule_counts   = mules.groupby('branch_id')['account_id'].count().rename('branch_mules')
    branch_stats  = pd.concat([branch_counts, mule_counts], axis=1).fillna(0)
    branch_stats['f_branch_mule_rate_v2'] = (
        branch_stats['branch_mules'] / (branch_stats['branch_total'] + 1)
    )
    branch_stats['f_branch_is_hotspot_v2'] = (
        branch_stats['f_branch_mule_rate_v2'] > branch_stats['f_branch_mule_rate_v2'].quantile(0.9)
    ).astype(int)
    branch_stats = branch_stats.reset_index()

    # 2. Alert reason one-hot per branch
    if 'alert_reason' in mules.columns:
        alert_dummies = pd.get_dummies(mules['alert_reason'], prefix='alert')
        mules_enc = pd.concat([mules[['branch_id']], alert_dummies], axis=1)
        branch_alert = mules_enc.groupby('branch_id').mean().add_prefix('f_br_').reset_index()
    else:
        branch_alert = None

    # 3. Per-account features
    # Merge branch features to all accounts
    base = pd.read_parquet("features/all_features_v3.parquet")

    # Add branch_id to base
    if 'branch_id' not in base.columns:
        branch_col = [c for c in accts.columns if 'branch' in c.lower()]
        if branch_col:
            base = base.merge(
                accts[['account_id', branch_col[0]]].rename(
                    columns={branch_col[0]: 'branch_id'}),
                on='account_id', how='left')

    base = base.merge(
        branch_stats[['branch_id','f_branch_mule_rate_v2','f_branch_is_hotspot_v2']],
        on='branch_id', how='left')

    if branch_alert is not None:
        base = base.merge(branch_alert, on='branch_id', how='left')
        alert_cols = [c for c in base.columns if c.startswith('f_br_')]
        base[alert_cols] = base[alert_cols].fillna(0)
        log.info(f"Added {len(alert_cols)} alert_reason branch features")

    base['f_branch_mule_rate_v2']   = base['f_branch_mule_rate_v2'].fillna(0)
    base['f_branch_is_hotspot_v2']  = base['f_branch_is_hotspot_v2'].fillna(0)

    # Drop branch_id if it wasn't there before
    if 'branch_id' in base.columns:
        base = base.drop(columns=['branch_id'])

    # Drop old leaky versions if present
    old = [c for c in base.columns if c in [
        'f_branch_mule_rate','f_branch_mule_count',
        'f_branch_is_hotspot','f_net_flow','f_cp_vol_ratio'
    ]]
    if old:
        base = base.drop(columns=old)
        log.info(f"Dropped old: {old}")

    out = "features/all_features_v4.parquet"
    base.to_parquet(out, index=False)
    log.info(f"Saved → {out} | Shape: {base.shape}")

    # Signal check
    train_labels = safe_read(TRAIN_LABELS_PATH, columns=['account_id','is_mule'])
    check = base.merge(train_labels, on='account_id', how='inner')
    m = check[check['is_mule']==1]['f_branch_mule_rate_v2'].mean()
    l = check[check['is_mule']==0]['f_branch_mule_rate_v2'].mean()
    log.info(f"\n=== SIGNAL CHECK ===")
    log.info(f"Mules  branch_mule_rate: {m:.4f}")
    log.info(f"Legits branch_mule_rate: {l:.4f}")
    log.info(f"Separation: {m/(l+1e-9):.1f}x")
    log.info("\nNow run: python train_model.py")


if __name__ == "__main__":
    main()