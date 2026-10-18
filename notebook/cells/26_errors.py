# ----------------------------------------------------------------------------------------------
# Figure 10 -- what the model relies on (permutation importance) and what the ablation was worth.
# Plus subgroup recall and a false-negative profile.
# ----------------------------------------------------------------------------------------------
set_global_seed()

# ---- Q1: permutation importance of the best classical model, measured on the test partition -----
pi_idx = train_test_split(np.arange(len(X_test_fe)), train_size=20_000,
                          stratify=y_test, random_state=SEED)[0]
X_pi, y_pi = X_test_fe.iloc[pi_idx], y_test.iloc[pi_idx]

t0 = time.perf_counter()
perm = permutation_importance(
    hgb, X_pi, y_pi, scoring="average_precision",
    n_repeats=5, random_state=SEED, n_jobs=1,
)
pi_seconds = time.perf_counter() - t0

importance = pd.DataFrame({
    "feature": FULL_FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)
importance.to_csv(RESULT_DIR / "feature_importance_permutation.csv", index=False)

print(f"Permutation importance: {len(FULL_FEATURES)} features x 5 repeats on {len(X_pi):,} test rows "
      f"in {pi_seconds:.1f}s")
print(importance.head(15).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

# ---- Q2: subgroup performance ------------------------------------------------------------------
tuned_thr = float(exp_df.loc[exp_df.exp_id == best_overall_id, "tuned_threshold"].iloc[0])
pred_best = (proba_best >= tuned_thr).astype(int)
sub = X_test_fe.copy()
sub["y_true"] = y_test.to_numpy()
sub["y_pred"] = pred_best

AGE_LABELS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
              "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
BMI_LABELS = ["<18.5", "18.5-24.9", "25-29.9", "30-34.9", "35-39.9", ">=40"]


def subgroup_table(frame, column, labels=None, one_indexed=False):
    """Recall/precision within each level of `column`.

    `one_indexed=True` for BRFSS code variables, which start at 1 rather than 0 (Age band 1 is
    "18-24", so the label is labels[value - 1], not labels[value]).
    """
    rows = []
    for value, grp in frame.groupby(column, observed=True):
        tp = int(((grp.y_true == 1) & (grp.y_pred == 1)).sum())
        fn = int(((grp.y_true == 1) & (grp.y_pred == 0)).sum())
        fp = int(((grp.y_true == 0) & (grp.y_pred == 1)).sum())
        tn = int(((grp.y_true == 0) & (grp.y_pred == 0)).sum())

        if labels is None:
            label = str(value)
        else:
            idx = int(value) - 1 if one_indexed else int(value)
            label = labels[idx] if 0 <= idx < len(labels) else str(value)

        rows.append({
            "group": label, "n": len(grp), "prevalence": round(grp.y_true.mean(), 4),
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else np.nan,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else np.nan,
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else np.nan,
        })
    return pd.DataFrame(rows)


print(f"\nSubgroup performance of {best_overall_id} at its validation-tuned threshold {tuned_thr:.3f}")
for column, labels, title, one_idx in [
        ("Age", AGE_LABELS, "Age band", True),
        ("GenHlth", None, "Self-rated general health (1 excellent - 5 poor)", False),
        ("BMI_band", BMI_LABELS, "BMI band (kg/m^2)", False),
        ("Sex", ["female", "male"], "Sex", False),
        ("Income", None, "Income band (1 <$10k - 8 >=$75k)", False)]:
    tbl = subgroup_table(sub, column, labels, one_indexed=one_idx)
    tbl.to_csv(RESULT_DIR / f"subgroup_{column}.csv", index=False)
    print(f"\n  -- {title} --")
    print(tbl.to_string(index=False))

# ---- Q3: false negatives versus true positives -------------------------------------------------
positives = sub[sub.y_true == 1]
fn_profile = positives[positives.y_pred == 0]
tp_profile = positives[positives.y_pred == 1]
profile_cols = ["BMI", "Age", "GenHlth", "PhysHlth", "MentHlth", "CardioRisk", "HighBP", "HighChol"]
profile = pd.DataFrame({
    "missed (FN) mean": fn_profile[profile_cols].mean(),
    "caught (TP) mean": tp_profile[profile_cols].mean(),
})
profile["difference"] = profile["missed (FN) mean"] - profile["caught (TP) mean"]
profile.to_csv(RESULT_DIR / "false_negative_profile.csv", index_label="feature")
print(f"\nMissed cases: {len(fn_profile):,} of {len(positives):,} positives "
      f"({len(fn_profile)/len(positives):.1%}). Their profile versus correctly detected cases:")
print(profile.to_string(float_format=lambda x: f"{x:+.3f}"))

# ---- Figure 10: ablation deltas + permutation importance ---------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(7.0, 9.6))

pairs_plot = [
    ("Logistic\nRegression", "A1 LR | raw features", "A2 LR | raw + engineered"),
    ("Random\nForest", "A3 RF | raw features", "A4 RF | raw + engineered"),
    ("Gradient\nBoosting", "A5 HGB | raw features", "A6 HGB | raw + engineered"),
]
x = np.arange(len(pairs_plot))
raw_vals = [ablation_df.loc[ablation_df.run == a, "pr_auc"].iloc[0] for _, a, _ in pairs_plot]
fe_vals = [ablation_df.loc[ablation_df.run == b, "pr_auc"].iloc[0] for _, _, b in pairs_plot]

axes[0].bar(x - 0.18, raw_vals, width=0.36, label="Raw 21 features", color="#A8C4E0",
            edgecolor="black", linewidth=0.5)
axes[0].bar(x + 0.18, fe_vals, width=0.36, label="+ 9 engineered features", color="#4C72B0",
            edgecolor="black", linewidth=0.5)
for i, (a, b) in enumerate(zip(raw_vals, fe_vals)):
    axes[0].annotate(f"{b-a:+.4f}", xy=(i + 0.18, b), xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold",
                     color="#2E7D32" if b > a else "#B71C1C")
axes[0].set_xticks(x)
axes[0].set_xticklabels([p[0] for p in pairs_plot])
axes[0].set_ylabel("Test PR-AUC")
axes[0].set_ylim(min(raw_vals + fe_vals) - 0.05, max(raw_vals + fe_vals) + 0.05)
axes[0].legend(fontsize=8.5, loc="best")
axes[0].set_title("(a) Feature-engineering ablation, one change at a time")

top_n = importance.head(15).iloc[::-1]
colour_map = ["#C44E52" if f in ENGINEERED else "#4C72B0" for f in top_n["feature"]]
axes[1].barh(top_n["feature"], top_n["importance_mean"], xerr=top_n["importance_std"],
             color=colour_map, edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 0.8})
axes[1].set_xlabel("Mean decrease in PR-AUC when permuted")
axes[1].set_title("(b) Permutation importance, gradient boosting (red = engineered)")
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(facecolor="#4C72B0", edgecolor="black", label="original"),
                        Patch(facecolor="#C44E52", edgecolor="black", label="engineered")],
               fontsize=8.5, loc="lower right")

fig.suptitle("Figure 10 — What the ablation was worth, and what the model actually relies on",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig10_ablation_importance.png")
n_eng_top15 = int(importance.head(15)["feature"].isin(ENGINEERED).sum())
print(f"\nEngineered features in the top 15 by permutation importance: {n_eng_top15} of {len(ENGINEERED)}")
