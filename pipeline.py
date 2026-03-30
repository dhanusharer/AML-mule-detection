"""
pipeline.py - AML Mule Detection Full Pipeline
==============================================
Single command to run the full workflow end to end.

Usage:
    python pipeline.py
    python pipeline.py --from 3
    python pipeline.py --skip-features

Steps:
    1. feature_builders.build_features          - base transaction + static features
    2. feature_builders.add_advanced_features   - burst, velocity, graph centrality
    3. feature_builders.add_geo_features        - geo, IP, balance velocity
    4. feature_builders.build_cp_features_final - label-free counterparty features
    5. feature_builders.add_label_features      - branch label features -> all_features_v4
    6. train_model.py             - pass 1, generate OOF scores
    7. feature_builders.build_network_features  - OOF-based network features
    8. train_model.py             - pass 2, train final models
    9. make_submission.py         - generate submission.csv
    10. explain_xai.py            - generate XAI outputs
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def log(msg, colour=RESET):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{colour}{BOLD}[{ts}]{RESET} {colour}{msg}{RESET}", flush=True)


def run_step(step_num, name, target, should_skip_if_output_exists=False, required_output=None):
    """Run one pipeline step with timing, error handling, and output-aware skipping."""
    print(f"\n{'=' * 60}")

    if should_skip_if_output_exists and required_output and os.path.exists(required_output):
        log(f"Step {step_num}: {name}", BLUE)
        log(f"  Output already exists: {required_output}", YELLOW)
        log("  Skipping (delete the output to re-run this step)", YELLOW)
        return True

    log(f"Step {step_num}/{TOTAL_STEPS}: {name}", BLUE)
    command = [sys.executable, "-m", target] if is_module_target(target) else [sys.executable, target]
    display = f"python -m {target}" if is_module_target(target) else f"python {target}"
    log(f"  Running: {display}", BLUE)

    start = time.time()
    result = subprocess.run(
        command,
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        log(f"  FAILED after {elapsed:.0f}s - check output above", RED)
        log(f"  To resume from this step: python pipeline.py --from {step_num}", YELLOW)
        return False

    log(f"  Done in {elapsed / 60:.1f} min", GREEN)
    return True


def is_module_target(target):
    return target.startswith("feature_builders.")


def resolve_step_target(target):
    """Resolve a configured execution target."""
    if is_module_target(target):
        module_path = target.replace(".", os.sep) + ".py"
        if os.path.exists(module_path):
            return target, None
        return None, None

    if os.path.exists(target):
        return target, None

    v2_target = target.replace(".py", "_v2.py")
    if os.path.exists(v2_target):
        return v2_target, f"Using {v2_target} (fallback _v2 variant)"

    return None, None


def check_prerequisites():
    """Verify the core data files exist before feature building starts."""
    required = [
        "data/accounts.parquet",
        "data/customers.parquet",
        "data/train_labels.parquet",
        "data/test_accounts.parquet",
        "data/transactions",
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        log("Missing required data files:", RED)
        for path in missing:
            log(f"  {path}", RED)
        return False
    return True


STEPS = [
    (1, "Base features", "feature_builders.build_features", "features/all_features.parquet"),
    (2, "Advanced features (burst/vel)", "feature_builders.add_advanced_features", None),
    (3, "Geo + IP + balance features", "feature_builders.add_geo_features", None),
    (4, "CP behavioural features", "feature_builders.build_cp_features_final", None),
    (5, "Branch label features -> v4", "feature_builders.add_label_features", "features/all_features_v4.parquet"),
    (6, "Train pass-1 (OOF generation)", "train_model.py", "output/oof_predictions.csv"),
    (7, "OOF-based network features", "feature_builders.build_network_features", None),
    (8, "Train pass-2 (final models)", "train_model.py", None),
    (9, "Generate submission", "make_submission.py", "output/submission.csv"),
    (10, "XAI explainability outputs", "explain_xai.py", "output/xai/xai_report.txt"),
]

TOTAL_STEPS = len(STEPS)

# These steps should always run even if their usual outputs already exist.
ALWAYS_RERUN = {6, 8, 9, 10}

# These steps depend on transactions_additional and should be skipped if it is absent.
NEEDS_TXN_ADDITIONAL = {3}


def main():
    parser = argparse.ArgumentParser(description="AML Mule Detection Pipeline")
    parser.add_argument(
        "--from",
        dest="from_step",
        type=int,
        default=1,
        help="Resume from step number N (default: 1)",
    )
    parser.add_argument(
        "--to",
        dest="to_step",
        type=int,
        default=TOTAL_STEPS,
        help=f"Stop after step number N (default: {TOTAL_STEPS})",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip steps 1-5 and use existing feature files",
    )
    parser.add_argument(
        "--only-submission",
        action="store_true",
        help="Run only the final training and submission steps",
    )
    parser.add_argument(
        "--only-xai",
        action="store_true",
        help="Run only the XAI generation step",
    )
    args = parser.parse_args()

    if args.skip_features:
        args.from_step = 6
    if args.only_submission:
        args.from_step = 8
    if args.only_xai:
        args.from_step = 10
        args.to_step = 10

    print(f"\n{BOLD}{'=' * 60}")
    print("  AML Mule Detection - Full Pipeline")
    print(f"  Steps {args.from_step} -> {args.to_step}")
    print(f"{'=' * 60}{RESET}")

    if args.from_step <= 5 and not check_prerequisites():
        sys.exit(1)

    has_txn_additional = os.path.exists("data/transactions_additional")
    if not has_txn_additional:
        log("data/transactions_additional not found - step 3 will be skipped", YELLOW)

    pipeline_start = time.time()
    failed_step = None

    for step_num, name, target, required_output in STEPS:
        if step_num < args.from_step or step_num > args.to_step:
            continue

        if step_num in NEEDS_TXN_ADDITIONAL and not has_txn_additional:
            log(f"Step {step_num}: {name} - SKIPPED (no transactions_additional)", YELLOW)
            continue

        resolved_target, note = resolve_step_target(target)
        if resolved_target is None:
            log(f"Step {step_num}: {target} not found - skipping", YELLOW)
            continue
        if note:
            log(f"Step {step_num}: {note}", YELLOW)

        should_skip_if_output_exists = step_num not in ALWAYS_RERUN
        success = run_step(
            step_num,
            name,
            resolved_target,
            should_skip_if_output_exists=should_skip_if_output_exists,
            required_output=required_output,
        )
        if not success:
            failed_step = step_num
            break

    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 60}")

    if failed_step is not None:
        log(f"Pipeline FAILED at step {failed_step}", RED)
        log("Fix the error above then run:", YELLOW)
        log(f"  python pipeline.py --from {failed_step}", YELLOW)
        sys.exit(1)

    log(f"Pipeline COMPLETE in {total_time / 60:.1f} minutes", GREEN)
    log("Outputs:", GREEN)

    outputs = {
        "Submission": "output/submission.csv",
        "Metrics": "output/metrics.json",
        "OOF preds": "output/oof_predictions.csv",
        "Feature imp": "output/feature_importance.csv",
        "SHAP": "output/shap_importance.csv",
        "XAI report": "output/xai/xai_report.txt",
    }
    for label, path in outputs.items():
        exists = os.path.exists(path)
        marker = "OK" if exists else "missing"
        colour = GREEN if exists else RED
        log(f"  {label:<12} {path}  {marker}", colour)

    print(f"\n{BOLD}Upload output/submission.csv to the portal.{RESET}\n")


if __name__ == "__main__":
    main()
