"""
Project configuration for AML mule detection.
"""

import json
import os


DATA_DIR = "data"
OUTPUT_DIR = "output"
FEATURES_DIR = "features"
MODELS_DIR = "models"
TUNING_DIR = "tuning"
TUNED_PARAMS_PATH = os.path.join(TUNING_DIR, "best_params.json")

for directory in [OUTPUT_DIR, FEATURES_DIR, MODELS_DIR, TUNING_DIR]:
    os.makedirs(directory, exist_ok=True)


CUSTOMERS_PATH = f"{DATA_DIR}/customers.parquet"
ACCOUNTS_PATH = f"{DATA_DIR}/accounts.parquet"
DEMOGRAPHICS_PATH = f"{DATA_DIR}/demographics.parquet"
ACCOUNTS_ADDITIONAL_PATH = f"{DATA_DIR}/accounts-additional.parquet"
BRANCH_PATH = f"{DATA_DIR}/branch.parquet"
LINKAGE_PATH = f"{DATA_DIR}/customer_account_linkage.parquet"
PRODUCT_DETAILS_PATH = f"{DATA_DIR}/product_details.parquet"
TRAIN_LABELS_PATH = f"{DATA_DIR}/train_labels.parquet"
TEST_ACCOUNTS_PATH = f"{DATA_DIR}/test_accounts.parquet"

TRANSACTIONS_GLOB = f"{DATA_DIR}/transactions/batch-*/part_*.parquet"
TRANSACTIONS_ADDITIONAL_GLOB = f"{DATA_DIR}/transactions_additional/batch-*/part_*.parquet"

FEATURES_CACHE_ACTIVE = f"{FEATURES_DIR}/all_features_v4.parquet"

STRUCTURING_THRESHOLD_LOW = 45_000
STRUCTURING_THRESHOLD_HIGH = 50_000
ROUND_AMOUNTS = {1_000, 2_000, 5_000, 10_000, 20_000, 25_000, 50_000, 100_000}
DORMANCY_DAYS = 90
NEW_ACCOUNT_DAYS = 180
MOBILE_CHANGE_LOOKBACK_DAYS = 30
PASSTHROUGH_WINDOW_HOURS = 24
DATA_END = "2025-06-30"
DATA_START = "2020-07-01"

RANDOM_STATE = 42
N_FOLDS = 5

LGBM_PARAMS = dict(
    objective="binary",
    metric=["auc", "binary_logloss"],
    learning_rate=0.02,
    num_leaves=255,
    max_depth=-1,
    min_child_samples=15,
    feature_fraction=0.6,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.05,
    reg_lambda=0.5,
    min_split_gain=0.01,
    n_estimators=5000,
    n_jobs=-1,
    verbose=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    is_unbalance=False,
)

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    learning_rate=0.02,
    max_depth=7,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.05,
    reg_lambda=0.5,
    n_estimators=5000,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=0,
    use_label_encoder=False,
    early_stopping_rounds=100,
)

CB_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=1,
    iterations=5000,
    random_seed=RANDOM_STATE,
    verbose=0,
    auto_class_weights="Balanced",
    early_stopping_rounds=100,
)


def _apply_param_overrides():
    """Load tuned parameter overrides if a tuning run has saved them."""
    if not os.path.exists(TUNED_PARAMS_PATH):
        return

    with open(TUNED_PARAMS_PATH, "r", encoding="utf-8") as f:
        overrides = json.load(f)

    if overrides.get("LGBM_PARAMS"):
        LGBM_PARAMS.update(overrides["LGBM_PARAMS"])
    if overrides.get("XGB_PARAMS"):
        XGB_PARAMS.update(overrides["XGB_PARAMS"])
    if overrides.get("CB_PARAMS"):
        CB_PARAMS.update(overrides["CB_PARAMS"])


_apply_param_overrides()
