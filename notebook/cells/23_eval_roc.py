# ----------------------------------------------------------------------------------------------
# Figure 7 -- ROC (left) and precision-recall (right) for the same four models.
# ----------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(6.8, 9.4))
colors = sns.color_palette("husl", len(panel_ids))

for colour, pid in zip(colors, panel_ids):
    rec = TEST_PREDICTIONS[pid]
    y_true, proba = rec["y_true"], rec["proba"]
    label = f"{pid} {rec['name']}"

    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, linewidth=2, color=colour,
                 label=f"{label} (AUC {roc_auc_score(y_true, proba):.4f})")

    prec, rec_, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(rec_, prec, linewidth=2, color=colour,
                 label=f"{label} (AP {average_precision_score(y_true, proba):.4f})")

axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance (AUC 0.500)")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate (recall)")
axes[0].set_title("(a) ROC — insensitive to the 6.2:1 imbalance")
axes[0].legend(fontsize=8, loc="lower right")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1.01)

baseline_pr = y_test.mean()
axes[1].axhline(baseline_pr, color="k", linestyle="--", linewidth=1,
                label=f"Prevalence baseline ({baseline_pr:.3f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("(b) Precision–recall — the honest view under imbalance")
axes[1].legend(fontsize=8, loc="upper right")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1.01)

fig.suptitle("Figure 7 — The same models look strong on ROC and merely moderate on precision–recall",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig7_roc_pr_curves.png")
print("ROC-AUC versus PR-AUC for every experiment:")
gap_df = exp_df[["exp_id", "model", "roc_auc", "pr_auc"]].copy()
gap_df["optimism_gap"] = (gap_df["roc_auc"] - gap_df["pr_auc"]).round(4)
print(gap_df.to_string(index=False))
print(f"\nMean ROC-AUC {gap_df['roc_auc'].mean():.4f} vs mean PR-AUC {gap_df['pr_auc'].mean():.4f}: "
      f"the {gap_df['optimism_gap'].mean():.4f} average gap is the imbalance effect made visible.")
