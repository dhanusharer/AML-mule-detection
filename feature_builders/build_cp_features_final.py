"""
build_cp_features_final.py
===========================
Build CP suspiciousness scores WITHOUT using labels at all.
Instead: score each CP by the BEHAVIORAL features of transactions through it.

Suspicious CP characteristics (label-free):
1. High round-amount ratio (structuring)
2. High night/weekend transaction ratio  
3. Many accounts using it (hub behavior)
4. High passthrough: receives and sends quickly
5. High velocity: many txns in short time
6. Accounts using it have many counterparties (fan-out)

This gives a genuine signal on BOTH train and test accounts.
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
    log.info("  CP Behavioral Features — Label Free")
    log.info("=" * 65)

    test_df   = safe_read(TEST_ACCOUNTS_PATH, columns=["account_id"])
    labels    = safe_read(TRAIN_LABELS_PATH, columns=["account_id","is_mule"])
    all_accts = set(test_df["account_id"]) | set(labels["account_id"])
    log.info(f"All accounts: {len(all_accts):,}")

    txn_files = sorted(glob(TRANSACTIONS_GLOB))

    # Per CP accumulators
    cp_n_txns        = defaultdict(int)
    cp_n_accounts    = defaultdict(set)
    cp_vol_total     = defaultdict(float)
    cp_n_round       = defaultdict(int)     # round amounts
    cp_n_night       = defaultdict(int)     # night txns (22-6)
    cp_n_weekend     = defaultdict(int)     # weekend txns
    cp_n_struct      = defaultdict(int)     # structuring amounts
    cp_credit_vol    = defaultdict(float)   # vol received
    cp_debit_vol     = defaultdict(float)   # vol sent

    # Per account accumulators
    acct_to_cps      = defaultdict(set)
    acct_cp_vol      = defaultdict(float)

    ROUND_AMTS = {1000,2000,5000,10000,20000,25000,50000,100000}

    log.info("Scanning transactions...")
    for fi, fpath in enumerate(txn_files):
        try:
            df = pd.read_parquet(fpath, columns=[
                "account_id","counterparty_id","amount",
                "txn_type","transaction_timestamp"
            ])
            df = df.dropna(subset=["counterparty_id"])
            df["vol"]  = df["amount"].abs()
            df["ts"]   = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
            df["hour"] = df["ts"].dt.hour
            df["dow"]  = df["ts"].dt.dayofweek
            df["is_night"]   = ((df["hour"]>=22)|(df["hour"]<=6)).astype(int)
            df["is_weekend"] = (df["dow"]>=5).astype(int)
            df["is_round"]   = df["vol"].isin(ROUND_AMTS).astype(int)
            df["is_struct"]  = df["vol"].between(45000,50000).astype(int)

            for row in df.itertuples(index=False):
                cp  = row.counterparty_id
                vol = row.vol
                acct = row.account_id

                cp_n_txns[cp]     += 1
                cp_vol_total[cp]  += vol
                cp_n_accounts[cp].add(acct)
                cp_n_round[cp]    += row.is_round
                cp_n_night[cp]    += row.is_night
                cp_n_weekend[cp]  += row.is_weekend
                cp_n_struct[cp]   += row.is_struct

                if row.txn_type == "C":
                    cp_credit_vol[cp] += vol
                else:
                    cp_debit_vol[cp]  += vol

                acct_to_cps[acct].add(cp)
                key = (acct, cp)
                acct_cp_vol[key] = acct_cp_vol.get(key, 0.0) + vol

            del df; gc.collect()
        except Exception as e:
            pass
        if (fi+1) % 100 == 0:
            log.info(f"  {fi+1}/{len(txn_files)} | CPs: {len(cp_n_txns):,}")

    log.info(f"Total CPs: {len(cp_n_txns):,}")

    # Compute CP suspiciousness score (label-free)
    log.info("Computing CP suspiciousness scores...")
    cp_susp = {}
    for cp in cp_n_txns:
        n    = cp_n_txns[cp] + 1e-9
        vol  = cp_vol_total[cp] + 1e-9
        n_ac = len(cp_n_accounts[cp])

        round_ratio   = cp_n_round[cp] / n
        night_ratio   = cp_n_night[cp] / n
        weekend_ratio = cp_n_weekend[cp] / n
        struct_ratio  = cp_n_struct[cp] / n
        hub_score     = min(n_ac / 50, 1.0)   # normalized hub-ness

        # Passthrough: how balanced are credit vs debit
        cred = cp_credit_vol[cp]
        deb  = cp_debit_vol[cp]
        passthrough = 1 - abs(cred - deb) / (cred + deb + 1e-9)

        # Combined suspiciousness
        cp_susp[cp] = (
            round_ratio   * 0.25 +
            night_ratio   * 0.20 +
            struct_ratio  * 0.25 +
            hub_score     * 0.15 +
            passthrough   * 0.15
        )

    all_susp = list(cp_susp.values())
    hot_threshold = np.percentile(all_susp, 85)
    hot_cps = {cp for cp, s in cp_susp.items() if s >= hot_threshold}
    log.info(f"Hot CPs (top 15%): {len(hot_cps):,} | threshold={hot_threshold:.4f}")

    # Per-account features
    log.info("Building per-account features...")
    rows = []
    for acct in all_accts:
        my_cps = acct_to_cps.get(acct, set())
        if not my_cps:
            rows.append({
                "account_id":             acct,
                "f_cpb_n_hot_cps":        0,
                "f_cpb_hot_ratio":        0.0,
                "f_cpb_max_susp":         0.0,
                "f_cpb_mean_susp":        0.0,
                "f_cpb_weighted_susp":    0.0,
                "f_cpb_hot_vol_ratio":    0.0,
                "f_cpb_log_hot_vol":      0.0,
                "f_cpb_n_cps":            0,
            })
            continue

        my_hot     = my_cps & hot_cps
        n_hot      = len(my_hot)
        n_cps      = len(my_cps)
        susp_list  = [cp_susp.get(cp, 0) for cp in my_cps]
        max_susp   = max(susp_list)
        mean_susp  = float(np.mean(susp_list))

        total_vol  = sum(acct_cp_vol.get((acct,cp),0) for cp in my_cps) + 1e-9
        hot_vol    = sum(acct_cp_vol.get((acct,cp),0) for cp in my_hot)
        weighted   = sum(
            acct_cp_vol.get((acct,cp),0) * cp_susp.get(cp,0)
            for cp in my_cps
        ) / total_vol

        rows.append({
            "account_id":             acct,
            "f_cpb_n_hot_cps":        n_hot,
            "f_cpb_hot_ratio":        n_hot / (n_cps + 1e-9),
            "f_cpb_max_susp":         max_susp,
            "f_cpb_mean_susp":        mean_susp,
            "f_cpb_weighted_susp":    weighted,
            "f_cpb_hot_vol_ratio":    hot_vol / total_vol,
            "f_cpb_log_hot_vol":      float(np.log1p(hot_vol)),
            "f_cpb_n_cps":            n_cps,
        })

    feat_df = pd.DataFrame(rows)

    # Signal check
    check   = feat_df.merge(labels, on="account_id", how="inner")
    mules_  = check[check["is_mule"]==1]
    legits_ = check[check["is_mule"]==0]
    sep = mules_["f_cpb_weighted_susp"].mean() / (legits_["f_cpb_weighted_susp"].mean()+1e-9)
    log.info(f"\n=== SIGNAL CHECK (label-free) ===")
    log.info(f"Mules  weighted_susp: {mules_['f_cpb_weighted_susp'].mean():.4f}")
    log.info(f"Legits weighted_susp: {legits_['f_cpb_weighted_susp'].mean():.4f}")
    log.info(f"Separation: {sep:.2f}x")

    # Merge into v3
    base = pd.read_parquet("features/all_features_v3.parquet")
    old  = [c for c in base.columns if c.startswith("f_cpb") or
            c.startswith("f_cp2") or c.startswith("f_cp_") or
            c.startswith("f_net_") or
            c in ["f_branch_mule_rate","f_branch_mule_count"]]
    if old:
        base = base.drop(columns=old)
        log.info(f"Dropped old: {old}")

    merged = base.merge(feat_df, on="account_id", how="left")
    new_cols = [c for c in feat_df.columns if c != "account_id"]
    merged[new_cols] = merged[new_cols].fillna(0)

    out = "features/all_features_v4.parquet"
    merged.to_parquet(out, index=False)
    log.info(f"\nSaved → {out} | Shape: {merged.shape}")
    log.info("Now run: python train_model.py && python make_submission.py")


if __name__ == "__main__":
    main()