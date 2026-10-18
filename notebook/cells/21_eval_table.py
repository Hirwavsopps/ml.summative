# ----------------------------------------------------------------------------------------------
# Table II -- consolidate every experiment, then Figure 8 (metric comparison).
# ----------------------------------------------------------------------------------------------
exp_df = pd.DataFrame(EXPERIMENT_LOG)
exp_df = exp_df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
exp_df.to_csv(RESULT_DIR / "table2_experiments.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
show_cols = ["exp_id", "model", "paradigm", "accuracy", "precision", "recall", "f1",
             "roc_auc", "pr_auc", "mcc", "brier", "recall_tuned", "f1_tuned", "fit_time_s"]
print("TABLE II -- all experiments, ranked by PR-AUC (test partition, threshold 0.5 unless *_tuned)\n")
print(exp_df[show_cols].to_string(index=False))

print("\nFull hyperparameters per run:")
for _, r in exp_df.iterrows():
    print(f"  {r['exp_id']:<3} {r['model']:<32} {r['hyperparameters']}")

baseline_acc = max(y_test.mean(), 1 - y_test.mean())
best = exp_df.iloc[0]
print(f"\nMajority-class baseline      : accuracy {baseline_acc:.4f}, PR-AUC {y_test.mean():.4f}")
print(f"Best model by PR-AUC         : {best['exp_id']} {best['model']}  PR-AUC {best['pr_auc']:.4f}, "
      f"ROC-AUC {best['roc_auc']:.4f}")
print(f"Best classical               : "
      f"{exp_df[exp_df.paradigm=='Classical'].iloc[0]['exp_id']} "
      f"({exp_df[exp_df.paradigm=='Classical'].iloc[0]['pr_auc']:.4f} PR-AUC)")
print(f"Best deep learning           : "
      f"{exp_df[exp_df.paradigm=='Deep Learning'].iloc[0]['exp_id']} "
      f"({exp_df[exp_df.paradigm=='Deep Learning'].iloc[0]['pr_auc']:.4f} PR-AUC)")

# ---- Figure 8: metric comparison across experiments -------------------------------------------
metric_cols = ["roc_auc", "pr_auc", "f1", "recall", "precision"]
plot_df = exp_df.sort_values("pr_auc")
ypos = np.arange(len(plot_df))
height = 0.16

fig, ax = plt.subplots(figsize=(7.2, 7.0))
palette = sns.color_palette("viridis", len(metric_cols))
for k, (metric, colour) in enumerate(zip(metric_cols, palette)):
    ax.barh(ypos + (k - 2) * height, plot_df[metric], height=height,
            label=metric.replace("_", "-").upper(), color=colour, edgecolor="black", linewidth=0.3)

ax.set_yticks(ypos)
ax.set_yticklabels([f"{i}  {m}" for i, m in zip(plot_df["exp_id"], plot_df["model"])], fontsize=8.5)
ax.set_xlabel("Score (test partition)")
ax.set_xlim(0, 1.0)
ax.axvline(baseline_acc, color="black", linestyle=":", linewidth=1.2)
ax.text(baseline_acc + 0.005, len(plot_df) - 0.4, f"majority-class accuracy {baseline_acc:.3f}",
        fontsize=8, style="italic")
ax.legend(ncol=5, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
ax.set_title("Figure 8 — Every experiment scores similarly on accuracy, and differently on recall",
             pad=32)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig8_model_comparison.png")
print("\nFigure 8 written to figures/fig8_model_comparison.png")
