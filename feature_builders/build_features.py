"""
AML Mule Detection - Feature Engineering
==========================================
ULTRA-FAST VERSION v4 — Fixed memory growth issue
- NO Python sets (were causing RAM to grow unboundedly)
- NO per-row loops
- Pure pandas groupby only
- Constant memory usage throughout all 396 files
- Expected time: 15-25 minutes

Usage: python -m feature_builders.build_features
"""
import gc, os, sys, traceback, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from glob import glob

from config import *
from utils import log, safe_read, reduce_mem, safe_ratio


# ══════════════════════════════════════════════════════════════
#  STATIC FEATURES
# ══════════════════════════════════════════════════════════════

def build_static_features():
    log.info("=== [1/3] Static features ===")
    accounts        = safe_read(ACCOUNTS_PATH)
    linkage         = safe_read(LINKAGE_PATH)
    customers       = safe_read(CUSTOMERS_PATH)
    demographics    = safe_read(DEMOGRAPHICS_PATH)
    product_details = safe_read(PRODUCT_DETAILS_PATH)
    acc_add         = safe_read(ACCOUNTS_ADDITIONAL_PATH)
    branch          = safe_read(BRANCH_PATH)

    df = (accounts
          .merge(linkage,         on="account_id",  how="left")
          .merge(customers,       on="customer_id", how="left")
          .merge(demographics,    on="customer_id", how="left")
          .merge(product_details, on="customer_id", how="left")
          .merge(acc_add,         on="account_id",  how="left")
          .merge(branch,          on="branch_code", how="left"))

    today = pd.Timestamp("2025-06-30")
    for col in ["account_opening_date","last_mobile_update_date","last_kyc_date",
                "freeze_date","unfreeze_date","relationship_start_date","date_of_birth",
                "address_last_update_date","passbook_last_update_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    def c0(n):       return df[n].fillna(0) if n in df.columns else pd.Series(0, index=df.index)
    def cn(n):       return df[n]           if n in df.columns else pd.Series(pd.NaT, index=df.index)
    def fl(n, v="Y"): return (df[n]==v).astype(int) if n in df.columns else pd.Series(0, index=df.index)

    df["f_account_age_days"]           = (today - cn("account_opening_date")).dt.days.clip(0)
    df["f_customer_age"]               = ((today - cn("date_of_birth")).dt.days / 365.25).clip(0)
    df["f_relationship_days"]          = (today - cn("relationship_start_date")).dt.days.clip(0)
    df["f_is_new_account"]             = (df["f_account_age_days"] < NEW_ACCOUNT_DAYS).astype(int)
    df["f_days_since_mobile_update"]   = (today - cn("last_mobile_update_date")).dt.days.clip(0)
    df["f_recent_mobile_change"]       = (df["f_days_since_mobile_update"] < MOBILE_CHANGE_LOOKBACK_DAYS).astype(int)
    df["f_days_since_kyc"]             = (today - cn("last_kyc_date")).dt.days.clip(0)
    df["f_account_frozen"]             = (df.get("account_status", pd.Series("", index=df.index)) == "frozen").astype(int)
    df["f_kyc_non_compliant"]          = fl("kyc_compliant", "N")
    df["f_log_avg_balance"]            = np.log1p(c0("avg_balance").clip(0))
    df["f_log_monthly_avg_balance"]    = np.log1p(c0("monthly_avg_balance").clip(0))
    df["f_avg_balance_raw"]            = c0("avg_balance")
    df["f_near_zero_balance"]          = (c0("avg_balance").abs() < 500).astype(int)
    df["f_product_family_enc"]         = df.get("product_family", pd.Series("", index=df.index)).map({"S":0,"K":1,"O":2}).fillna(-1)
    df["f_rural_branch"]               = fl("rural_branch")
    df["f_no_nomination"]              = fl("nomination_flag", "N")
    df["f_num_chequebooks"]            = c0("num_chequebooks")
    df["f_mobile_banking"]             = fl("mobile_banking_flag")
    df["f_internet_banking"]           = fl("internet_banking_flag")
    df["f_has_atm_card"]               = fl("atm_card_flag")
    df["f_digital_footprint"]          = (df["f_mobile_banking"] + df["f_internet_banking"] +
                                          df["f_has_atm_card"] + fl("demat_flag") +
                                          fl("credit_card_flag") + fl("fastag_flag"))
    df["f_kyc_doc_count"]              = fl("pan_available") + fl("aadhaar_available") + fl("passport_available")
    df["f_nri"]                        = fl("nri_flag")
    df["f_joint_account"]              = fl("joint_account_flag")
    df["f_days_since_address_update"]  = (today - cn("address_last_update_date")).dt.days.clip(0)
    df["f_days_since_passbook_update"] = (today - cn("passbook_last_update_date")).dt.days.clip(0)
    df["f_pin_mismatch"]               = (df.get("customer_pin", pd.Series("", index=df.index)).astype(str) !=
                                          df.get("branch_pin",   pd.Series("", index=df.index)).astype(str)).astype(int)
    df["f_high_risk_scheme"]           = (df["scheme_code"].map({"PMJDY":1}).fillna(0)
                                          if "scheme_code" in df.columns else 0)
    for c in ["loan_count","cc_count","od_count","ka_count","sa_count"]:
        df[c] = c0(c)
    df["f_product_count"]    = df["loan_count"]+df["cc_count"]+df["od_count"]+df["ka_count"]+df["sa_count"]
    df["f_loan_to_savings"]  = safe_ratio(c0("loan_sum").abs(), c0("sa_sum").abs()+1)
    df["f_log_branch_turnover"] = np.log1p(c0("branch_turnover").clip(0))
    df["f_branch_type_enc"]  = df.get("branch_type", pd.Series("", index=df.index)).map(
                                    {"urban":0,"semi-urban":1,"rural":2}).fillna(1)
    df["f_acct_near_rel_start"] = ((cn("account_opening_date") - cn("relationship_start_date")).dt.days.abs() < 30).astype(int)

    # ── NEW STATIC FEATURES ──────────────────────────────────
    # Was account ever frozen?
    df["f_ever_frozen"]     = (cn("freeze_date").notna()).astype(int)
    # Multiple freeze cycles (freeze + unfreeze = suspicious)
    df["f_freeze_unfreeze"] = (cn("freeze_date").notna() & cn("unfreeze_date").notna()).astype(int)
    # Account opened very recently (< 30 days = extreme new account risk)
    df["f_very_new_account"]= (df["f_account_age_days"] < 30).astype(int)
    # No cheque book but high digital = proxy for cash-only mule
    df["f_no_cheque_digital"]= ((fl("cheque_availed","N")) & (df["f_digital_footprint"] > 2)).astype(int)
    # Overdraft account (higher layering risk)
    df["f_is_overdraft"]    = (df.get("product_family", pd.Series("",index=df.index)) == "O").astype(int)
    # Customer has multiple accounts (multi-account mule pattern)
    acct_per_cust = df.groupby("customer_id")["account_id"].transform("count") if "customer_id" in df.columns else 1
    df["f_multi_account_customer"] = (acct_per_cust > 1).astype(int)
    df["f_num_accounts_customer"]  = acct_per_cust
    # Quarterly balance vs monthly balance ratio (volatility signal)
    df["f_bal_quarterly_monthly_ratio"] = safe_ratio(
        c0("quarterly_avg_balance").abs(), c0("monthly_avg_balance").abs() + 1)
    # Branch employee count (low employees = less oversight)
    df["f_low_branch_employees"] = (c0("branch_employee_count") < 10).astype(int)
    df["f_branch_employee_count"]= np.log1p(c0("branch_employee_count").clip(0))

    df["f_days_frozen"] = 0.0
    if "freeze_date" in df.columns and df["freeze_date"].notna().any():
        mask = df["freeze_date"].notna()
        unfreeze = cn("unfreeze_date").fillna(today)
        df.loc[mask, "f_days_frozen"] = (unfreeze[mask] - df.loc[mask, "freeze_date"]).dt.days.clip(0)

    fcols = ["account_id"] + [c for c in df.columns if c.startswith("f_")]
    out = df[fcols].copy()
    log.info(f"  {out.shape[1]-1} static features, {len(out):,} accounts")
    return out


# ══════════════════════════════════════════════════════════════
#  FAST TRANSACTION FEATURES — fixed memory, no sets
# ══════════════════════════════════════════════════════════════

def process_one_file(fpath, add_lookup, has_add, mcc_median_map, WINDOW_START):
    """
    Process a single parquet file pair (core + additional).
    Returns a DataFrame with per-account aggregates for this file only.
    All operations are vectorized pandas groupby — no Python loops.
    """
    COLS_CORE = ["transaction_id","account_id","transaction_timestamp",
                 "amount","txn_type","channel","counterparty_id","mcc_code"]
    COLS_ADD  = ["transaction_id","latitude","longitude",
                 "ip_address","balance_after_transaction","transaction_sub_type"]

    df = pd.read_parquet(fpath, columns=COLS_CORE)
    df["ts"]     = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    df["abs"]    = df["amount"].abs().astype(np.float32)
    df["is_cr"]  = (df["txn_type"] == "C").astype(np.int8)
    df["is_db"]  = (df["txn_type"] == "D").astype(np.int8)
    df["is_rev"] = (df["amount"] < 0).astype(np.int8)

    # Merge additional
    if has_add:
        def file_key(path):
            p = path.replace("\\","/").split("/")
            return (p[-2].replace("batch-",""), p[-1].replace("part_","").replace(".parquet",""))
        key      = file_key(fpath)
        add_path = add_lookup.get(key)
        if add_path:
            add = pd.read_parquet(add_path, columns=COLS_ADD)
            df  = df.merge(add, on="transaction_id", how="left")
            del add

    # Derived columns — all vectorized
    df["cr_amt"]  = df["abs"] * df["is_cr"]
    df["db_amt"]  = df["abs"] * df["is_db"]
    df["abs_sq"]  = df["abs"] ** 2
    df["is_struct"] = df["abs"].between(STRUCTURING_THRESHOLD_LOW, STRUCTURING_THRESHOLD_HIGH).astype(np.int8)
    df["is_round"]  = df["abs"].isin(ROUND_AMOUNTS).astype(np.int8)
    df["is_upi"]    = df["channel"].isin(["UPC","UPD"]).astype(np.int8)
    df["is_atm"]    = (df["channel"] == "ATW").astype(np.int8)
    df["is_cash"]   = df["channel"].isin(["CSD","OCD"]).astype(np.int8)
    df["is_inter"]  = df["channel"].isin(["IAD","IFD","IFC"]).astype(np.int8)
    df["is_night"]  = ((df["ts"].dt.hour >= 22) | (df["ts"].dt.hour < 6)).astype(np.int8)
    df["is_wkend"]  = (df["ts"].dt.dayofweek >= 5).astype(np.int8)
    df["is_mend"]   = df["ts"].dt.day.isin([28,29,30,31]).astype(np.int8)
    df["is_early_big"] = ((df["is_cr"]==1) & df["ts"].dt.day.isin(range(1,6)) & (df["abs"] > 10_000)).astype(np.int8)
    df["is_recent"] = (df["ts"] >= WINDOW_START).astype(np.int8)
    df["ts_int"]    = df["ts"].astype(np.int64).where(df["ts"].notna(), other=0)

    # MCC anomaly
    df["mcc_med"]   = df["mcc_code"].map(mcc_median_map).fillna(1.0).astype(np.float32)
    df["mcc_anom"]  = df["abs"] / (df["mcc_med"] + 1.0)

    # ── NEW HIGH-VALUE FEATURES ──────────────────────────────
    # Large transactions (> 50k = above reporting threshold)
    df["is_large"]    = (df["abs"] > 50_000).astype(np.int8)
    # Very large (> 1 lakh)
    df["is_xlarge"]   = (df["abs"] > 1_00_000).astype(np.int8)
    # Large credits specifically (fan-in signal)
    df["is_large_cr"] = ((df["is_cr"]==1) & (df["abs"] > 50_000)).astype(np.int8)
    # Large debits specifically (fan-out signal)
    df["is_large_db"] = ((df["is_db"]==1) & (df["abs"] > 50_000)).astype(np.int8)
    # Same-day credit then debit (rapid passthrough within same day)
    df["date_str"]    = df["ts"].dt.date.astype(str)
    daily_cr = df[df["is_cr"]==1].groupby(["account_id","date_str"])["abs"].sum().reset_index()
    daily_cr.columns = ["account_id","date_str","day_cr"]
    daily_db = df[df["is_db"]==1].groupby(["account_id","date_str"])["abs"].sum().reset_index()
    daily_db.columns = ["account_id","date_str","day_db"]
    daily = daily_cr.merge(daily_db, on=["account_id","date_str"], how="inner")
    daily["is_passday"] = (
        (daily["day_cr"] > 10_000) &
        (daily["day_db"] / (daily["day_cr"] + 1) > 0.8)
    ).astype(int)
    passthrough_days = daily.groupby("account_id")["is_passday"].sum().reset_index()
    passthrough_days.columns = ["account_id","n_passthrough_days"]
    df = df.merge(passthrough_days, on="account_id", how="left")
    df["n_passthrough_days"] = df["n_passthrough_days"].fillna(0).astype(np.float32)
    # IMPS / NEFT / fund transfer channels (inter-bank layering)
    df["is_imps_neft"] = df["channel"].isin(["IPM","NTD","FTD","FTC","P2A"]).astype(np.int8)
    # High-value UPI (above 50k via UPI = suspicious)
    df["is_upi_large"] = (df["channel"].isin(["UPC","UPD"]) & (df["abs"] > 50_000)).astype(np.int8)
    # Quarter-end transactions
    df["is_qend"] = df["ts"].dt.month.isin([3,6,9,12]).astype(np.int8)

    # All groupby aggregations in ONE pass
    agg_dict = {
        "transaction_id": "count",
        "cr_amt":  "sum", "db_amt":  "sum",
        "is_cr":   "sum", "is_db":   "sum", "is_rev": "sum",
        "abs":     "sum", "abs_sq":  "sum", "abs_max": pd.NamedAgg("abs","max"),
        "is_struct":"sum","is_round":"sum",
        "is_upi":  "sum","is_atm":  "sum","is_cash": "sum","is_inter":"sum",
        "is_night":"sum","is_wkend":"sum","is_mend": "sum",
        "is_early_big":"sum","is_recent":"sum",
        "mcc_anom_sum": pd.NamedAgg("mcc_anom","sum"),
        "mcc_anom_max": pd.NamedAgg("mcc_anom","max"),
        "ts_min": pd.NamedAgg("ts_int","min"),
        "ts_max": pd.NamedAgg("ts_int","max"),
        "cp_nunique":  pd.NamedAgg("counterparty_id","nunique"),
        "cp_cr_nuniq": pd.NamedAgg("counterparty_id", lambda x: x[df.loc[x.index,"is_cr"]==1].nunique()),
        "cp_db_nuniq": pd.NamedAgg("counterparty_id", lambda x: x[df.loc[x.index,"is_db"]==1].nunique()),
        "date_nunique": pd.NamedAgg("ts", lambda x: x.dt.date.nunique()),
    }

    # Simpler approach — split into fast and slow aggs
    g = df.groupby("account_id")

    fast = g.agg(
        n_total    =("transaction_id","count"),
        sum_cr     =("cr_amt","sum"),
        sum_db     =("db_amt","sum"),
        n_cr       =("is_cr","sum"),
        n_db       =("is_db","sum"),
        n_rev      =("is_rev","sum"),
        sum_abs    =("abs","sum"),
        sum_sq     =("abs_sq","sum"),
        max_abs    =("abs","max"),
        n_struct   =("is_struct","sum"),
        n_round    =("is_round","sum"),
        n_upi      =("is_upi","sum"),
        n_atm      =("is_atm","sum"),
        n_cash     =("is_cash","sum"),
        n_inter    =("is_inter","sum"),
        n_night    =("is_night","sum"),
        n_wkend    =("is_wkend","sum"),
        n_mend     =("is_mend","sum"),
        n_early_big=("is_early_big","sum"),
        n_recent   =("is_recent","sum"),
        n_rev2     =("is_rev","sum"),
        mcc_asum   =("mcc_anom","sum"),
        mcc_amax   =("mcc_anom","max"),
        ts_min     =("ts_int","min"),
        ts_max     =("ts_int","max"),
        cp_nuniq   =("counterparty_id","nunique"),
        date_nuniq =("ts", lambda x: x.dt.date.nunique()),
        # NEW features
        n_large       =("is_large","sum"),
        n_xlarge      =("is_xlarge","sum"),
        n_large_cr    =("is_large_cr","sum"),
        n_large_db    =("is_large_db","sum"),
        n_passdays    =("n_passthrough_days","max"),
        n_imps_neft   =("is_imps_neft","sum"),
        n_upi_large   =("is_upi_large","sum"),
        n_qend        =("is_qend","sum"),
        sum_large_cr  =("abs", lambda x: x[(df.loc[x.index,"is_large_cr"]==1)].sum()),
        sum_large_db  =("abs", lambda x: x[(df.loc[x.index,"is_large_db"]==1)].sum()),
    )

    # Optional additional columns
    if "balance_after_transaction" in df.columns:
        bal = g["balance_after_transaction"].agg(["sum","min"])
        bal.columns = ["bal_sum","bal_min"]
        bal["bal_sq"] = g["balance_after_transaction"].apply(lambda x: (x.dropna()**2).sum())
        bal["bal_n"]  = g["balance_after_transaction"].apply(lambda x: x.dropna().count())
        fast = fast.join(bal)

    if "latitude" in df.columns and df["latitude"].notna().any():
        df["lat_valid"] = df["latitude"].notna() & df["longitude"].notna()
        # Geographic spread — high spread = suspicious (multiple cities)
        geo = pd.DataFrame({
            "lat_sum":  g["latitude"].sum(),
            "lat_sq":   g["latitude"].apply(lambda x: (x.dropna()**2).sum()),
            "lat_n":    g["latitude"].apply(lambda x: x.dropna().count()),
            "lon_sum":  g["longitude"].sum(),
            "lon_sq":   g["longitude"].apply(lambda x: (x.dropna()**2).sum()),
            "lat_min":  g["latitude"].min(),
            "lat_max":  g["latitude"].max(),
            "lon_min":  g["longitude"].min(),
            "lon_max":  g["longitude"].max(),
        })
        fast = fast.join(geo)
        # Geo range = bounding box diagonal (geographic spread)
        if "lat_min" in fast.columns:
            fast["geo_range"] = np.sqrt(
                (fast["lat_max"] - fast["lat_min"])**2 +
                (fast["lon_max"] - fast["lon_min"])**2)

    if "ip_address" in df.columns:
        fast["ip_nuniq"] = g["ip_address"].nunique()
        # IP prefix diversity (different /24 subnets = different locations)
        df["ip_prefix"] = df["ip_address"].str.extract(r"^(\d+\.\d+\.\d+)")[0].fillna("0.0.0")
        fast["ip_prefix_nuniq"] = g["ip_prefix"].nunique()

    if "transaction_sub_type" in df.columns:
        df["is_clt"]  = (df["transaction_sub_type"] == "CLT_CASH").astype(np.int8)
        df["is_loan"] = (df["transaction_sub_type"] == "LOAN").astype(np.int8)
        fast["n_clt"]  = g["is_clt"].sum()
        fast["n_loan"] = g["is_loan"].sum()

    if "balance_after_transaction" in df.columns:
        # Near-zero balance after txn = rapid passthrough signal
        df["bal"] = df["balance_after_transaction"].fillna(0)
        df["is_near_zero_bal"] = (df["bal"].abs() < 1000).astype(np.int8)
        df["is_negative_bal"]  = (df["bal"] < 0).astype(np.int8)
        fast["n_near_zero_bal"] = g["is_near_zero_bal"].sum()
        fast["n_negative_bal"]  = g["is_negative_bal"].sum()
        fast["bal_min2"]        = g["bal"].min()
        fast["bal_max2"]        = g["bal"].max()
        fast["bal_range"]       = fast["bal_max2"] - fast["bal_min2"]

    del df; gc.collect()
    # Reset index so account_id becomes a column (not index)
    return fast.reset_index()


def build_transaction_features():
    log.info("=== [2/3] Transaction features ===")

    core_files = sorted(glob(TRANSACTIONS_GLOB))
    add_files  = sorted(glob(TRANSACTIONS_ADDITIONAL_GLOB))

    def file_key(path):
        p = path.replace("\\","/").split("/")
        return (p[-2].replace("batch-",""), p[-1].replace("part_","").replace(".parquet",""))
    add_lookup = {file_key(f): f for f in add_files}
    has_add    = len(add_files) > 0
    log.info(f"  Core: {len(core_files)} files | Additional: {len(add_files)} files")

    # ── Pass 1: MCC medians ──────────────────────────────────
    log.info("  Pass 1/2: MCC medians (fast) ...")
    mcc_sums   = {}
    mcc_counts = {}
    for i, fpath in enumerate(core_files):
        tmp = pd.read_parquet(fpath, columns=["mcc_code","amount"])
        tmp["amount"] = tmp["amount"].abs()
        grp = tmp.groupby("mcc_code")["amount"].agg(["sum","count"])
        for mcc, row in grp.iterrows():
            mcc_sums[mcc]   = mcc_sums.get(mcc,0)   + row["sum"]
            mcc_counts[mcc] = mcc_counts.get(mcc,0) + row["count"]
        del tmp; gc.collect()
        if (i+1) % 100 == 0:
            log.info(f"    MCC pass {i+1}/{len(core_files)}")
    mcc_median_map = {k: mcc_sums[k]/mcc_counts[k] for k in mcc_sums}
    del mcc_sums, mcc_counts; gc.collect()
    log.info(f"  MCC medians for {len(mcc_median_map):,} codes")

    WINDOW_START = pd.Timestamp("2025-06-01")

    # ── Pass 2: Stream files, accumulate with pd.concat + groupby ──
    log.info(f"  Pass 2/2: Streaming {len(core_files)} files ...")

    # Accumulate in CHUNKS of 50 files then reduce — keeps memory flat
    CHUNK = 50
    chunk_results = []

    for i, fpath in enumerate(core_files):
        partial = process_one_file(fpath, add_lookup, has_add, mcc_median_map, WINDOW_START)
        chunk_results.append(partial)

        # Every CHUNK files (or last file): reduce by summing/maxing
        if (i+1) % CHUNK == 0 or (i+1) == len(core_files):
            log.info(f"  Reducing chunk at file {i+1}/{len(core_files)} ...")
            combined = pd.concat(chunk_results, ignore_index=False)

            # Columns that need max aggregation
            MAX_COLS = {"max_abs", "mcc_amax", "ts_max"}
            # Columns that need min aggregation
            MIN_COLS = {"ts_min", "bal_min"}
            # Everything else gets summed
            all_data_cols = [c for c in combined.columns if c != "account_id"]
            sum_cols = [c for c in all_data_cols if c not in MAX_COLS and c not in MIN_COLS]
            max_cols = [c for c in all_data_cols if c in MAX_COLS]
            min_cols = [c for c in all_data_cols if c in MIN_COLS]

            agg_dict = {}
            for c in sum_cols: agg_dict[c] = "sum"
            for c in max_cols: agg_dict[c] = "max"
            for c in min_cols: agg_dict[c] = "min"

            # IMPORTANT: use groupby on account_id COLUMN not index
            # so all accounts across all chunks are properly merged
            if "account_id" not in combined.columns:
                combined = combined.reset_index()
            reduced = combined.groupby("account_id", as_index=False).agg(agg_dict)
            del combined, chunk_results; gc.collect()
            chunk_results = [reduced]
            log.info(f"  Chunk reduced. Accounts: {len(reduced):,}")

    # Final result
    final = chunk_results[0]
    log.info(f"  Final aggregated accounts: {len(final):,}")

    # ── Build feature columns ────────────────────────────────
    log.info("  Building feature columns ...")
    # account_id is already a column (reset_index was done in reduce step)
    if "account_id" not in final.columns and final.index.name == "account_id":
        final = final.reset_index()
    f = final.copy()

    n  = f["n_total"].values.astype(float)
    n1 = n + 1  # avoid division by zero

    avg_amt = f["sum_abs"].values / n1
    var_amt = np.maximum(0, f["sum_sq"].values / n1 - avg_amt**2)
    std_amt = np.sqrt(var_amt)

    # Timestamps back to datetime for span
    ts_min = pd.to_datetime(f["ts_min"].replace(0, np.nan), unit="ns", errors="coerce")
    ts_max = pd.to_datetime(f["ts_max"].replace(0, np.nan), unit="ns", errors="coerce")
    spans  = (ts_max - ts_min).dt.days.fillna(0).clip(0).values
    udays  = f["date_nuniq"].values.astype(float)
    lifetime = spans / 30.0 + 1
    burst    = f["n_recent"].values / (n / lifetime + 1e-9)

    feat_df = pd.DataFrame({
        "account_id":              f["account_id"],
        "f_txn_count":             n,
        "f_total_credit":          f["sum_cr"].values,
        "f_total_debit":           f["sum_db"].values,
        "f_avg_txn_amount":        avg_amt,
        "f_std_txn_amount":        std_amt,
        "f_max_txn_amount":        f["max_abs"].values,
        "f_credit_debit_ratio":    f["sum_cr"].values / (f["sum_db"].values + 1),
        "f_net_flow":              f["sum_cr"].values - f["sum_db"].values,
        "f_passthrough_ratio":     np.minimum(f["sum_cr"].values, f["sum_db"].values) /
                                   (np.maximum(f["sum_cr"].values, f["sum_db"].values) + 1),
        "f_structuring_count":     f["n_struct"].values,
        "f_structuring_ratio":     f["n_struct"].values / n1,
        "f_round_amount_ratio":    f["n_round"].values  / n1,
        "f_unique_counterparties": f["cp_nuniq"].values,
        "f_fan_in_ratio":          f["cp_nuniq"].values / (f["n_cr"].values + 1),
        "f_fan_out_ratio":         f["cp_nuniq"].values / (f["n_db"].values + 1),
        "f_upi_fraction":          f["n_upi"].values   / n1,
        "f_atm_fraction":          f["n_atm"].values   / n1,
        "f_cash_deposit_fraction": f["n_cash"].values  / n1,
        "f_interaccount_fraction": f["n_inter"].values / n1,
        "f_night_txn_ratio":       f["n_night"].values / n1,
        "f_weekend_txn_ratio":     f["n_wkend"].values / n1,
        "f_month_end_txn_ratio":   f["n_mend"].values  / n1,
        "f_active_span_days":      spans,
        "f_unique_active_days":    udays,
        "f_txn_density":           n / (udays + 1),
        "f_recent_30d_count":      f["n_recent"].values,
        "f_burst_index":           burst,
        "f_salary_cycle_credits":  f["n_early_big"].values,
        "f_reversal_count":        f["n_rev"].values,
        "f_reversal_ratio":        f["n_rev"].values / n1,
        "f_avg_mcc_anomaly":       f["mcc_asum"].values / n1,
        "f_max_mcc_anomaly":       f["mcc_amax"].values,
    })

    # Optional features from transactions_additional
    if "bal_n" in f.columns:
        bn  = f["bal_n"].values + 1
        bm  = f["bal_sum"].values / bn
        bv  = np.maximum(0, f["bal_sq"].values / bn - bm**2)
        feat_df["f_balance_volatility"] = np.sqrt(bv)
        feat_df["f_min_balance"]        = f["bal_min"].values

    if "lat_n" in f.columns:
        ln   = f["lat_n"].values + 1
        lvar = np.maximum(0, f["lat_sq"].values/ln - (f["lat_sum"].values/ln)**2)
        lovar= np.maximum(0, f["lon_sq"].values/ln - (f["lon_sum"].values/ln)**2)
        feat_df["f_geo_spread"] = np.sqrt(lvar + lovar)

    if "ip_nuniq" in f.columns:
        feat_df["f_unique_ips"]        = f["ip_nuniq"].values
        feat_df["f_ip_per_txn"]        = f["ip_nuniq"].values / n1
    if "ip_prefix_nuniq" in f.columns:
        feat_df["f_ip_prefix_nuniq"]   = f["ip_prefix_nuniq"].values
        feat_df["f_ip_prefix_ratio"]   = f["ip_prefix_nuniq"].values / (f["ip_nuniq"].values + 1)

    if "n_clt" in f.columns:
        feat_df["f_clt_cash_fraction"] = f["n_clt"].values / n1
    if "n_loan" in f.columns:
        feat_df["f_loan_txn_fraction"] = f["n_loan"].values / n1

    # Balance volatility features
    if "n_near_zero_bal" in f.columns:
        feat_df["f_near_zero_bal_ratio"]= f["n_near_zero_bal"].values / n1
        feat_df["f_negative_bal_ratio"] = f["n_negative_bal"].values / n1
    if "bal_range" in f.columns:
        feat_df["f_balance_range"]      = np.log1p(f["bal_range"].values.clip(0))
        feat_df["f_min_balance_raw"]    = f["bal_min2"].values

    # Extended geo features
    if "geo_range" in f.columns:
        feat_df["f_geo_range"]          = f["geo_range"].values.clip(0)
        feat_df["f_geo_spread_lat"]     = np.sqrt(np.maximum(0,
            f["lat_sq"].values/(f["lat_n"].values+1) -
            (f["lat_sum"].values/(f["lat_n"].values+1))**2))
    if "lat_min" in f.columns:
        feat_df["f_lat_range"]          = (f["lat_max"] - f["lat_min"]).values.clip(0)
        feat_df["f_lon_range"]          = (f["lon_max"] - f["lon_min"]).values.clip(0)

    del final, f; gc.collect()
    log.info(f"  {feat_df.shape[1]-1} transaction features, {len(feat_df):,} accounts")
    return reduce_mem(feat_df)


# ══════════════════════════════════════════════════════════════
#  GRAPH FEATURES
# ══════════════════════════════════════════════════════════════

def build_graph_features():
    log.info("=== [3/3] Graph features ===")
    files = sorted(glob(TRANSACTIONS_GLOB))

    # Accumulate edge list as DataFrame chunks, reduce periodically
    cp_acct_chunks = []
    CHUNK = 100

    for i, fpath in enumerate(files):
        df = pd.read_parquet(fpath, columns=["account_id","counterparty_id"])
        # Keep unique pairs only per file (saves memory)
        pairs = df.drop_duplicates()
        cp_acct_chunks.append(pairs)
        del df, pairs; gc.collect()

        if (i+1) % CHUNK == 0 or (i+1) == len(files):
            combined = pd.concat(cp_acct_chunks).drop_duplicates()
            del cp_acct_chunks; gc.collect()
            cp_acct_chunks = [combined]
            log.info(f"  Graph: {i+1}/{len(files)} | unique pairs: {len(combined):,}")

    edge_df = cp_acct_chunks[0]

    # Shared counterparties: CPs connected to > 5 accounts
    cp_acct_count = edge_df.groupby("counterparty_id")["account_id"].nunique()
    shared_cps    = set(cp_acct_count[cp_acct_count > 5].index.tolist())

    edge_df["is_shared"] = edge_df["counterparty_id"].isin(shared_cps).astype(int)
    result = edge_df.groupby("account_id").agg(
        f_shared_counterparties = ("is_shared",       "sum"),
        f_total_cp_connections  = ("counterparty_id", "nunique"),
    ).reset_index()

    del edge_df, shared_cps; gc.collect()
    log.info(f"  {result.shape[1]-1} graph features, {len(result):,} accounts")
    return reduce_mem(result)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("  AML Feature Engineering — ULTRA FAST v4")
    log.info("=" * 55)

    for p in [ACCOUNTS_PATH, LINKAGE_PATH, CUSTOMERS_PATH, DEMOGRAPHICS_PATH,
              PRODUCT_DETAILS_PATH, ACCOUNTS_ADDITIONAL_PATH, BRANCH_PATH]:
        if not os.path.exists(p):
            log.error(f"MISSING: {p}"); sys.exit(1)

    txn_files = sorted(glob(TRANSACTIONS_GLOB))
    if not txn_files:
        log.error(f"No transaction files at: {TRANSACTIONS_GLOB}"); sys.exit(1)
    log.info(f"Transaction files: {len(txn_files)}")

    static_feats = build_static_features()
    txn_feats    = build_transaction_features()
    graph_feats  = build_graph_features()

    log.info("Merging all features ...")
    all_feats = (static_feats
                 .merge(txn_feats,   on="account_id", how="left")
                 .merge(graph_feats, on="account_id", how="left")
                 .fillna(0))

    fcols = [c for c in all_feats.columns if c.startswith("f_")]
    log.info(f"Total features: {len(fcols)}")
    log.info(f"Total accounts: {len(all_feats):,}")

    os.makedirs(FEATURES_DIR, exist_ok=True)
    all_feats.to_parquet(FEATURES_CACHE, index=False)
    log.info(f"Saved -> {FEATURES_CACHE}")
    log.info("DONE!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
