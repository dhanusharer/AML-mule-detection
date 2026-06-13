"""
add_geo_features.py — Geographic + IP + Balance velocity features
=================================================================
Uses transactions_additional for:
1. Geographic spread (lat/lon variance) — anomaly = mule signal
2. IP diversity — many different IPs = suspicious
3. Balance velocity — large balance swings
4. Transaction amount / balance ratio — income mismatch proxy

These features come from transactions_additional which has
lat, lon, ip_address, balance_after_transaction columns.

Runtime: ~40 min
"""
import warnings, gc, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
import math

from config import *
from utils import log

FEATURES_CACHE_V3 = "features/all_features_v3.parquet"

def haversine_spread(lats, lons):
    """Compute geographic spread as max distance between any two points (km)."""
    if len(lats) < 2:
        return 0.0
    # Simplified: use std of lat/lon as proxy (faster than pairwise)
    lat_std = np.std(lats)
    lon_std = np.std(lons)
    # Convert degrees to km approx (1 deg lat ~ 111km)
    return float(np.sqrt((lat_std * 111)**2 + (lon_std * 111)**2))

def main():
    log.info("=" * 55)
    log.info("  Geo + IP + Balance Feature Builder")
    log.info("=" * 55)

    # Load v2 features as base
    base_path = "features/all_features_v2.parquet"
    if not os.path.exists(base_path):
        base_path = FEATURES_CACHE
    log.info(f"Loading base: {base_path}")
    base = pd.read_parquet(base_path)
    log.info(f"  Base shape: {base.shape}")

    txn_add_files = sorted(glob(TRANSACTIONS_ADDITIONAL_GLOB))
    txn_cor_files = sorted(glob(TRANSACTIONS_GLOB))
    log.info(f"  Additional txn files: {len(txn_add_files)}")

    # Per-account accumulators
    acct_lats     = defaultdict(list)
    acct_lons     = defaultdict(list)
    acct_ips      = defaultdict(set)
    acct_ip_pre   = defaultdict(set)   # /24 subnet prefixes
    acct_bals     = defaultdict(list)
    acct_bal_amts = defaultdict(list)  # (balance, amount) pairs

    for fi, fpath in enumerate(txn_add_files):
        try:
            add_df = pd.read_parquet(fpath, columns=[
                "transaction_id", "latitude", "longitude",
                "ip_address", "balance_after_transaction"
            ])
            # Get account_id from core file
            core_path = fpath.replace("transactions_additional", "transactions")
            if not os.path.exists(core_path):
                # try matching by index
                core_path = txn_cor_files[fi] if fi < len(txn_cor_files) else None
            if core_path is None or not os.path.exists(core_path):
                del add_df; continue

            core_df = pd.read_parquet(core_path, columns=[
                "transaction_id", "account_id", "amount"
            ])
            df = add_df.merge(core_df, on="transaction_id", how="inner")
            del add_df, core_df

            # Geographic
            geo = df.dropna(subset=["latitude","longitude"])
            for acct, grp in geo.groupby("account_id"):
                lats = grp["latitude"].values
                lons = grp["longitude"].values
                acct_lats[acct].extend(lats.tolist())
                acct_lons[acct].extend(lons.tolist())

            # IP diversity
            ip_df = df.dropna(subset=["ip_address"])
            for acct, grp in ip_df.groupby("account_id"):
                for ip in grp["ip_address"]:
                    ip_str = str(ip)
                    acct_ips[acct].add(ip_str)
                    # /24 prefix
                    parts = ip_str.split(".")
                    if len(parts) >= 3:
                        acct_ip_pre[acct].add(".".join(parts[:3]))

            # Balance velocity
            bal_df = df.dropna(subset=["balance_after_transaction"])
            for acct, grp in bal_df.groupby("account_id"):
                bals = grp["balance_after_transaction"].values
                amts = grp["amount"].abs().values
                acct_bals[acct].extend(bals.tolist())
                acct_bal_amts[acct].extend(zip(bals.tolist(), amts.tolist()))

            del df; gc.collect()
        except Exception as e:
            pass
        if (fi+1) % 50 == 0:
            log.info(f"  Additional: {fi+1}/{len(txn_add_files)} | geo:{len(acct_lats)} ip:{len(acct_ips)} bal:{len(acct_bals)}")

    log.info("Computing per-account geo/IP/balance features...")
    all_accts = set(acct_lats) | set(acct_ips) | set(acct_bals)
    rows = []

    for acct in all_accts:
        # Geographic spread
        lats = acct_lats.get(acct, [])
        lons = acct_lons.get(acct, [])
        geo_spread    = haversine_spread(lats, lons) if len(lats) >= 2 else 0.0
        geo_lat_std   = float(np.std(lats)) if len(lats) >= 2 else 0.0
        geo_lon_std   = float(np.std(lons)) if len(lons) >= 2 else 0.0
        n_unique_locs = len(set(zip(
            [round(x,1) for x in lats],
            [round(x,1) for x in lons]
        ))) if lats else 0

        # IP diversity
        n_ips     = len(acct_ips.get(acct, set()))
        n_ip_pre  = len(acct_ip_pre.get(acct, set()))

        # Balance features
        bals = acct_bals.get(acct, [])
        if len(bals) >= 2:
            bal_arr   = np.array(bals)
            bal_min   = float(bal_arr.min())
            bal_max   = float(bal_arr.max())
            bal_range = bal_max - bal_min
            bal_std   = float(np.std(bal_arr))
            bal_mean  = float(np.mean(np.abs(bal_arr))) + 1e-9
            bal_cv    = bal_std / bal_mean
            # Near-zero balance ratio
            near_zero = float((np.abs(bal_arr) < 100).mean())
            # Negative balance (overdraft usage)
            neg_bal   = float((bal_arr < 0).mean())
        else:
            bal_range = bal_std = bal_cv = near_zero = neg_bal = 0.0

        # Amount vs balance ratio (income mismatch)
        bal_amts = acct_bal_amts.get(acct, [])
        if bal_amts:
            ratios = []
            for bal, amt in bal_amts:
                if abs(bal) > 100 and amt > 0:
                    ratios.append(amt / (abs(bal) + 1e-9))
            txn_bal_ratio = float(np.mean(ratios)) if ratios else 0.0
            high_ratio    = float(sum(1 for r in ratios if r > 0.5) / (len(ratios)+1))
        else:
            txn_bal_ratio = high_ratio = 0.0

        rows.append({
            "account_id":          acct,
            "f_geo_spread_km":     geo_spread,
            "f_geo_lat_std":       geo_lat_std,
            "f_geo_lon_std":       geo_lon_std,
            "f_geo_unique_locs":   n_unique_locs,
            "f_ip_nuniq_v2":       n_ips,
            "f_ip_prefix_nuniq_v2":n_ip_pre,
            "f_bal_range":         np.log1p(bal_range),
            "f_bal_std":           np.log1p(bal_std),
            "f_bal_cv_v2":         bal_cv,
            "f_bal_near_zero_v3":  near_zero,
            "f_bal_negative_ratio":neg_bal,
            "f_txn_bal_ratio":     txn_bal_ratio,
            "f_high_bal_ratio":    high_ratio,
        })

    geo_df = pd.DataFrame(rows)
    log.info(f"  Geo/IP/Balance features: {geo_df.shape}")

    # Merge into base
    base = base.merge(geo_df, on="account_id", how="left")
    new_cols = [c for c in geo_df.columns if c != "account_id"]
    base[new_cols] = base[new_cols].fillna(0)

    log.info(f"Final shape: {base.shape}")
    base.to_parquet(FEATURES_CACHE_V3, index=False)
    log.info(f"Saved → {FEATURES_CACHE_V3}")
    log.info(f"New features: {new_cols}")
    log.info("\nNow update config.py: set FEATURES_CACHE_V2 = 'features/all_features_v3.parquet'")
    log.info("Then run: python train_model.py && python make_submission.py")

if __name__ == "__main__":
    main()