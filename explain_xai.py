"""
explain_xai.py — Fixed (auto feature-col matching)
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODELS_DIR = "models"
OUTPUT_DIR = "output"
XAI_DIR    = os.path.join(OUTPUT_DIR, "xai")
TRAIN_LABELS = "data/train_labels.parquet"
os.makedirs(XAI_DIR, exist_ok=True)

print("=" * 55)
print("  AML XAI v2 — Explainability Engine (Fixed)")
print("=" * 55)

# Load config to get FEATURES_CACHE_ACTIVE
from config import FEATURES_CACHE_ACTIVE, OUTPUT_DIR as CFG_OUT

print("\n[1/7] Loading features and models ...")
feats      = pd.read_parquet(FEATURES_CACHE_ACTIVE)
lgb_models = joblib.load(os.path.join(MODELS_DIR, "lgb_models.pkl"))
xgb_models = joblib.load(os.path.join(MODELS_DIR, "xgb_models.pkl"))
try:
    wa, wb, wc = joblib.load(os.path.join(MODELS_DIR, "ensemble_weights.pkl"))
except:
    wa, wb, wc = 0.6, 0.35, 0.05
print(f"  Ensemble wts : LGB={wa}  XGB={wb}  CB={wc}")

# Auto-detect feature cols to match model
n_expected = lgb_models[0].n_features_
try:
    fcols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    fcols = [c for c in fcols if c in feats.columns]
except:
    fcols = []

if len(fcols) != n_expected:
    LEAKY = ["f_branch_mule_count","f_account_at_flagged","f_account_at_top5",
             "f_n_mule_","f_mule_flow","f_mule_neighbor","f_mule_txn",
             "f_has_mule","f_log_mule","f_cp2_","f_cp_weighted","f_cp_mean",
             "f_cp_max","f_cp_n_hot","f_cp_hot_cp","f_cp_mule","f_cp_n_mule",
             "f_br_alert_","f_gp_","f_iter_"]
    ZERO = {"f_pin_mismatch","f_ip_per_txn","f_clt_cash_fraction",
            "f_loan_txn_fraction","f_lon_range","f_sudden_activation",
            "f_fast_drain_ratio","f_geo_lon_std","f_geo_unique_locs",
            "f_bal_range","f_ip_prefix_nuniq_v2","f_bal_near_zero_v3"}
    fcols = [c for c in feats.columns
             if c.startswith("f_") and c not in ZERO
             and not any(c.startswith(p) for p in LEAKY)]

labels = pd.read_parquet(TRAIN_LABELS)
merged = feats.merge(labels[["account_id","is_mule"]], on="account_id", how="inner")
for c in fcols:
    if c not in merged.columns: merged[c] = 0.0
merged = merged.fillna(0)
X_train = merged[fcols].values.astype(np.float32)
y_train = merged["is_mule"].values
submission = pd.read_csv(os.path.join(OUTPUT_DIR, "submission.csv"))
print(f"  Train samples: {len(X_train):,} | Known mules: {y_train.sum():,}")
print(f"  Feature cols : {len(fcols)}")

# [2] Feature importance
print("\n[2/7] Computing feature importances ...")
fi_gain  = np.zeros(len(fcols))
fi_split = np.zeros(len(fcols))
for m in lgb_models:
    imp_g = m.booster_.feature_importance(importance_type="gain")
    imp_s = m.booster_.feature_importance(importance_type="split")
    if len(imp_g) == len(fcols):
        fi_gain  += imp_g
        fi_split += imp_s
fi_gain  /= max(len(lgb_models), 1)
fi_split /= max(len(lgb_models), 1)

fi_df = pd.DataFrame({
    "feature":    fcols,
    "gain":       fi_gain,
    "split":      fi_split,
    "importance": fi_gain * 0.7 + fi_split * 0.3,
}).sort_values("importance", ascending=False).reset_index(drop=True)

PATTERN_MAP = {
    "f_burst_index":"Dormant Activation","f_structuring_ratio":"Structuring",
    "f_fan_in_ratio":"Fan-In / Fan-Out","f_fan_out_ratio":"Fan-In / Fan-Out",
    "f_night_txn_ratio":"Layered/Subtle","f_round_amount_ratio":"Round Amount Patterns",
    "f_avg_mcc_anomaly":"MCC-Amount Anomaly","f_days_since_kyc":"KYC Non-Compliance",
    "f_account_age_days":"New Account High Value","f_shared_counterparties":"Branch-Level Collusion",
    "f_log_branch_turnover":"Branch-Level Collusion","f_ever_frozen":"Dormant Activation",
    "f_days_frozen":"Dormant Activation","f_atm_fraction":"Layered/Subtle",
    "f_txn_acceleration":"Dormant Activation","f_near_zero_bal_ratio":"Structuring",
    "f_total_cp_connections":"Fan-In / Fan-Out","f_graph_in_degree":"Fan-In / Fan-Out",
    "f_graph_fan_out":"Fan-In / Fan-Out","f_smurf_ratio":"Structuring",
}
COLORS = {
    "Dormant Activation":"#e74c3c","Structuring":"#e67e22",
    "Fan-In / Fan-Out":"#2ecc71","Layered/Subtle":"#3498db",
    "Round Amount Patterns":"#9b59b6","MCC-Amount Anomaly":"#1abc9c",
    "KYC Non-Compliance":"#e91e63","New Account High Value":"#ff5722",
    "Branch-Level Collusion":"#455a64","Other":"#bdc3c7",
}
fi_df["aml_pattern"] = fi_df["feature"].map(PATTERN_MAP).fillna("Other")
fi_df.to_csv(os.path.join(XAI_DIR, "feature_importance.csv"), index=False)

# [3] Feature importance plot
print("\n[3/7] Feature importance plot ...")
top30 = fi_df.head(30)
fig, ax = plt.subplots(figsize=(12, 10))
bar_colors = [COLORS.get(p, "#bdc3c7") for p in top30["aml_pattern"]]
ax.barh(range(len(top30)), top30["importance"].values[::-1],
        color=bar_colors[::-1], edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels([f.replace("f_","").replace("_"," ")
                    for f in top30["feature"].values[::-1]], fontsize=9)
ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_title("Top 30 Feature Importances\nColored by AML Pattern",
             fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)
seen, handles = [], []
for p, c in COLORS.items():
    if p in top30["aml_pattern"].values and p not in seen:
        handles.append(mpatches.Patch(color=c, label=p)); seen.append(p)
ax.legend(handles=handles, loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(XAI_DIR, "feature_importance_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: feature_importance_plot.png")

# [4] SHAP
print("\n[4/7] Computing SHAP values ...")
try:
    import shap
    np.random.seed(42)
    idx = np.random.choice(len(X_train), min(2000, len(X_train)), replace=False)
    explainer = shap.TreeExplainer(lgb_models[0])
    sv = explainer.shap_values(X_train[idx])
    if isinstance(sv, list): sv = sv[1]
    shap_df = pd.DataFrame({
        "feature": fcols, "shap_mean": np.abs(sv).mean(axis=0)
    }).sort_values("shap_mean", ascending=False)
    shap_df.to_csv(os.path.join(XAI_DIR, "shap_importance.csv"), index=False)
    # Simple bar plot instead of beeswarm
    top20 = shap_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top20)), top20["shap_mean"].values[::-1], color="#e74c3c", alpha=0.8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f.replace("f_","").replace("_"," ")
                        for f in top20["feature"].values[::-1]], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Feature Importance\nTop 20 Features", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(XAI_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_summary.png")
except Exception as e:
    print(f"  SHAP skipped: {e}")

# [5] Score distribution + PR curve
print("\n[5/7] Score distribution + PR curve ...")
from sklearn.metrics import precision_recall_curve, roc_curve, auc
oof = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_predictions.csv"))
y_oof = oof["is_mule"].values
s_oof = oof["oof_score"].values
p, r, t = precision_recall_curve(y_oof, s_oof)
f1s = 2*p*r/(p+r+1e-8)
best_t = t[np.argmax(f1s[:-1])]
best_f1 = f1s[:-1].max()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(r, p, color="#e74c3c", linewidth=2, label=f"PR curve (Best F1={best_f1:.3f})")
ax.scatter([r[np.argmax(f1s[:-1])]], [p[np.argmax(f1s[:-1])]],
           color="black", s=100, zorder=5, label=f"Best threshold={best_t:.3f}")
ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curve\n(OOF predictions)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax2 = axes[1]
mule_s  = s_oof[y_oof==1]
legit_s = s_oof[y_oof==0]
ax2.hist(legit_s, bins=50, alpha=0.6, color="#3498db", density=True,
         label=f"Legit (n={len(legit_s):,})")
ax2.hist(mule_s,  bins=50, alpha=0.7, color="#e74c3c", density=True,
         label=f"Mule (n={len(mule_s):,})")
ax2.axvline(best_t, color="black", linestyle="--", label=f"Threshold={best_t:.3f}")
ax2.set_xlabel("OOF Score", fontsize=11); ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Score Distribution\nMules vs Legits", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(XAI_DIR, "score_distribution_pr_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: score_distribution_pr_curve.png")

# [6] Pattern breakdown
print("\n[6/7] AML pattern breakdown ...")
pat_imp = fi_df.groupby("aml_pattern")["importance"].sum().sort_values(ascending=False)
pat_imp = pat_imp[pat_imp.index != "Other"]
fig, ax = plt.subplots(figsize=(10, 5))
colors_p = [COLORS.get(p, "#bdc3c7") for p in pat_imp.index]
bars = ax.bar(range(len(pat_imp)), pat_imp.values, color=colors_p, edgecolor="white")
ax.set_xticks(range(len(pat_imp)))
ax.set_xticklabels(pat_imp.index, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Total Feature Importance", fontsize=11)
ax.set_title("Feature Importance by AML Pattern", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, pat_imp.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.0f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(XAI_DIR, "pattern_breakdown.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: pattern_breakdown.png")

# [7] Per-account explanations
print("\n[7/7] Per-account explanations ...")
legit_means = X_train[y_train==0].mean(axis=0)
feat_stds   = X_train.std(axis=0) + 1e-9
top_mules   = submission[submission["is_mule"] >= 0.5].sort_values(
    "is_mule", ascending=False).head(100)
top_feats   = feats[feats["account_id"].isin(top_mules["account_id"])].copy()
for c in fcols:
    if c not in top_feats.columns: top_feats[c] = 0.0
top_feats = top_feats.fillna(0)

explanations = []
for _, row in top_mules.iterrows():
    acct = row["account_id"]
    af   = top_feats[top_feats["account_id"]==acct]
    if af.empty: continue
    xv   = af[fcols].values[0]
    devs = []
    for fi_rank, feat_name in enumerate(fi_df.head(30)["feature"]):
        if feat_name not in fcols: continue
        fidx = fcols.index(feat_name)
        dev  = (xv[fidx] - legit_means[fidx]) / feat_stds[fidx]
        if dev > 0.5:
            devs.append((feat_name, xv[fidx], legit_means[fidx], dev,
                         PATTERN_MAP.get(feat_name,"Other")))
    devs.sort(key=lambda x: -x[3])
    top3 = devs[:3]
    reasons = [f"{d[0].replace('f_','').replace('_',' ')}={d[1]:.3f} "
               f"(baseline={d[2]:.3f}, +{d[3]:.1f}std) [{d[4]}]"
               for d in top3]
    explanations.append({
        "account_id":  acct, "mule_score": round(float(row["is_mule"]),4),
        "reason_1":    reasons[0] if len(reasons)>0 else "",
        "reason_2":    reasons[1] if len(reasons)>1 else "",
        "reason_3":    reasons[2] if len(reasons)>2 else "",
        "aml_patterns": ", ".join(set(d[4] for d in top3)),
        "suspicious_start": row.get("suspicious_start",""),
        "suspicious_end":   row.get("suspicious_end",""),
    })

exp_df = pd.DataFrame(explanations)
exp_df.to_csv(os.path.join(XAI_DIR, "top_mule_explanations.csv"), index=False)
print(f"  Saved: top_mule_explanations.csv ({len(exp_df)} accounts)")

# Report
top_pattern = pat_imp.index[0] if len(pat_imp) else "N/A"
top_feature  = fi_df.iloc[0]["feature"]
with open(os.path.join(XAI_DIR, "xai_report.txt"), "w") as f:
    f.write(f"AML MULE DETECTION XAI REPORT\n{'='*50}\n\n")
    f.write(f"Features used : {len(fcols)}\n")
    f.write(f"Best OOF F1   : {best_f1:.5f} @ threshold {best_t:.4f}\n")
    f.write(f"Top pattern   : {top_pattern}\n")
    f.write(f"Top feature   : {top_feature}\n\n")
    f.write("TOP 15 FEATURES:\n")
    for _, r in fi_df.head(15).iterrows():
        f.write(f"  {r['feature']:<40} {r['importance']:.1f}  [{r['aml_pattern']}]\n")

print("\n" + "="*55)
print("  XAI v2 COMPLETE — all outputs in output/xai/")
print("="*55)
print(f"\n  Top AML pattern : {top_pattern}")
print(f"  Top feature     : {top_feature}")
print(f"  Best F1         : {best_f1:.5f}  @ threshold {best_t:.4f}")
print(f"  Accounts explained: {len(exp_df)}")