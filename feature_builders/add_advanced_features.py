"""
add_advanced_features.py
========================
Adds burst, velocity, graph centrality features to base features.
Input:  features/all_features.parquet  (v1 from build_features.py)
Output: features/all_features_v2.parquet  (v2 = v1 + advanced)

Run AFTER: python -m feature_builders.build_features
"""
import warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from datetime import timedelta

from config import *
from utils import log, safe_read, reduce_mem


def build_burst_velocity_features():
    """
    Per-account burst and velocity features from transactions.
    - Txn burst: ratio of recent txns to long-term average
    - Inter-txn time stats: mean/std/min gap between transactions
    - Daily velocity: avg txns per active day
    - Multi-period activity breakdown (last 7, 30, 90 days)
    """
    log.info("=== Building burst/velocity features ===")
    txn_files = sorted(glob(TRANSACTIONS_GLOB))
    WINDOW_90D  = pd.Timestamp("2025-04-01")
    WINDOW_30D  = pd.Timestamp("2025-06-01")
    WINDOW_7D   = pd.Timestamp("2025-06-23")

    acct_ts = defaultdict(list)

    for fi, fpath in enumerate(txn_files):
        try:
            df = pd.read_parquet(fpath, columns=["account_id","transaction_timestamp","amount"])
            df["ts"]  = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
            df["vol"] = df["amount"].abs()
            df = df.dropna(subset=["ts"])
            for acct, grp in df.groupby("account_id"):
                acct_ts[acct].extend(grp["ts"].tolist())
            del df; gc.collect()
        except Exception:
            pass
        if (fi+1) % 100 == 0:
            log.info(f"  {fi+1}/{len(txn_files)} files | accounts so far: {len(acct_ts):,}")

    log.info(f"Computing burst/velocity for {len(acct_ts):,} accounts...")
    rows = []
    for acct, ts_list in acct_ts.items():
        ts_list = sorted(ts_list)
        n_total = len(ts_list)
        n_90d   = sum(1 for t in ts_list if t >= WINDOW_90D)
        n_30d   = sum(1 for t in ts_list if t >= WINDOW_30D)
        n_7d    = sum(1 for t in ts_list if t >= WINDOW_7D)

        # Inter-txn time gaps
        gaps = []
        if len(ts_list) > 1:
            for i in range(1, len(ts_list)):
                g = (ts_list[i] - ts_list[i-1]).total_seconds() / 3600  # hours
                gaps.append(g)

        mean_gap = float(np.mean(gaps)) if gaps else 0
        std_gap  = float(np.std(gaps))  if gaps else 0
        min_gap  = float(np.min(gaps))  if gaps else 0
        max_gap  = float(np.max(gaps))  if gaps else 0

        # Burst score: recent rate vs long-term
        span_days = max((ts_list[-1] - ts_list[0]).days, 1)
        long_rate = n_total / (span_days / 30.0 + 1)
        burst_30d = n_30d / (long_rate + 1e-9)

        # Unique active days
        days_active = len(set(t.date() for t in ts_list))

        rows.append({
            "account_id":         acct,
            "f_n_txn_7d":         n_7d,
            "f_n_txn_30d":        n_30d,
            "f_n_txn_90d":        n_90d,
            "f_burst_30d_score":  burst_30d,
            "f_mean_gap_hours":   mean_gap,
            "f_std_gap_hours":    std_gap,
            "f_min_gap_hours":    min_gap,
            "f_max_gap_hours":    max_gap,
            "f_days_active":      days_active,
            "f_daily_velocity":   n_total / (days_active + 1),
            "f_recent_accel":     n_7d / (n_30d / 4 + 1e-9),  # recent week vs avg week
        })

    return pd.DataFrame(rows)


def build_counterparty_features():
    """
    Counterparty diversity and fan-in/fan-out features.
    - # unique CPs per credit/debit
    - CP concentration: top-1 CP fraction of total volume
    - CP reuse ratio: how often same CPs used
    """
    log.info("=== Building counterparty features ===")
    txn_files = sorted(glob(TRANSACTIONS_GLOB))

    acct_cp_cr  = defaultdict(set)  # CPs sending money in
    acct_cp_db  = defaultdict(set)  # CPs receiving money out
    acct_cp_vol = defaultdict(lambda: defaultdict(float))

    for fi, fpath in enumerate(txn_files):
        try:
            df = pd.read_parquet(fpath, columns=["account_id","counterparty_id","amount","txn_type"])
            df = df.dropna(subset=["counterparty_id"])
            df["vol"] = df["amount"].abs()
            for row in df.itertuples(index=False):
                acct = row.account_id
                cp   = row.counterparty_id
                vol  = row.vol
                if row.txn_type == "C":
                    acct_cp_cr[acct].add(cp)
                else:
                    acct_cp_db[acct].add(cp)
                acct_cp_vol[acct][cp] += vol
            del df; gc.collect()
        except Exception:
            pass
        if (fi+1) % 100 == 0:
            log.info(f"  {fi+1}/{len(txn_files)} files")

    log.info(f"Computing CP diversity for {len(acct_cp_vol):,} accounts...")
    rows = []
    for acct, cp_vols in acct_cp_vol.items():
        vols = list(cp_vols.values())
        total_vol = sum(vols) + 1e-9
        max_cp_vol = max(vols)
        n_cps = len(vols)
        n_cr_cps = len(acct_cp_cr.get(acct, set()))
        n_db_cps = len(acct_cp_db.get(acct, set()))

        # Top-1 CP concentration
        top1_frac = max_cp_vol / total_vol

        # Herfindahl Index — measures concentration (high = few dominant CPs)
        hhi = sum((v/total_vol)**2 for v in vols)

        rows.append({
            "account_id":          acct,
            "f_n_credit_cps":      n_cr_cps,
            "f_n_debit_cps":       n_db_cps,
            "f_cp_top1_vol_frac":  top1_frac,
            "f_cp_herfindahl":     hhi,
            "f_cp_diversity":      1.0 - hhi,
            "f_cr_db_cp_overlap":  len(acct_cp_cr.get(acct, set()) &
                                       acct_cp_db.get(acct, set())) / (n_cps + 1),
        })

    return pd.DataFrame(rows)


def main():
    log.info("=" * 55)
    log.info("  Add Advanced Features → all_features_v2.parquet")
    log.info("=" * 55)

    base = pd.read_parquet(FEATURES_CACHE)
    log.info(f"Base features: {base.shape}")

    burst_df = build_burst_velocity_features()
    cp_df    = build_counterparty_features()

    log.info("Merging...")
    merged = (base
              .merge(burst_df, on="account_id", how="left")
              .merge(cp_df,    on="account_id", how="left")
              .fillna(0))

    out_path = f"{FEATURES_DIR}/all_features_v2.parquet"
    merged.to_parquet(out_path, index=False)
    log.info(f"Saved → {out_path} | Shape: {merged.shape}")
    log.info(f"New features added: {merged.shape[1] - base.shape[1]}")
    log.info("Next: python -m feature_builders.add_geo_features")


if __name__ == "__main__":
    main()
