# ----------------------------------------------------------------------------------------------
# Threshold behaviour of the best model: the precision/recall trade made explicit.
# ----------------------------------------------------------------------------------------------
best_rec = TEST_PREDICTIONS[best_overall_id]
y_true_best, proba_best = best_rec["y_true"], best_rec["proba"]

grid = np.linspace(0.02, 0.98, 97)
curve = []
for t in grid:
    m = metrics_at(y_true_best, proba_best, t)
    curve.append({"threshold": t, "precision": m["precision"], "recall": m["recall"],
                  "f1": m["f1"], "accuracy": m["accuracy"]})
thr_curve = pd.DataFrame(curve)

opt_idx = thr_curve["f1"].idxmax()
opt_threshold = float(thr_curve.loc[opt_idx, "threshold"])

fig, axes = plt.subplots(2, 1, figsize=(6.8, 8.8))

axes[0].plot(thr_curve["threshold"], thr_curve["precision"], linewidth=2, label="Precision", color="#4C72B0")
axes[0].plot(thr_curve["threshold"], thr_curve["recall"], linewidth=2, label="Recall", color="#C44E52")
axes[0].plot(thr_curve["threshold"], thr_curve["f1"], linewidth=2, label="F1", color="#55A868")
axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1, label="Default 0.5")
axes[0].axvline(opt_threshold, color="green", linestyle=":", linewidth=1.6,
                label=f"F1-optimal {opt_threshold:.2f}")
axes[0].set_xlabel("Decision threshold")
axes[0].set_ylabel("Score")
axes[0].set_title(f"(a) {best_overall_id} — metric dependence on the threshold")
axes[0].legend(fontsize=8.5)
axes[0].set_ylim(0, 1.02)

axes[1].plot(thr_curve["recall"], thr_curve["precision"], linewidth=2, color="#8172B2")
axes[1].scatter([thr_curve.loc[thr_curve.threshold.idxmin(), "recall"]],
                [thr_curve.loc[thr_curve.threshold.idxmin(), "precision"]],
                marker="o", s=40, color="black", zorder=5, label="threshold 0.02")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("(b) The same trade in precision–recall space")
axes[1].legend(fontsize=8.5)
axes[1].set_xlim(0, 1.02)
axes[1].set_ylim(0, 1.02)

fig.suptitle("Figure 9 — Lowering the threshold trades precision for recall; at 6:1 imbalance that is usually the right trade",
             fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig9_threshold_analysis.png")
# ---- operating points worth thinking about ----------------------------------------------------
operating_points = []
for t in [0.10, 0.20, 0.30, 0.40, 0.50, round(opt_threshold, 2), 0.60, 0.70]:
    m = metrics_at(y_true_best, proba_best, t)
    operating_points.append({
        "threshold": t, "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4), "accuracy": round(m["accuracy"], 4),
        "flagged_per_1000": round(1000 * (proba_best >= t).mean(), 1),
        "missed_cases_per_1000_pos": round(1000 * (1 - m["recall"]), 1),
    })
op_df = pd.DataFrame(operating_points).drop_duplicates(subset="threshold").sort_values("threshold")
op_df.to_csv(RESULT_DIR / "threshold_operating_points.csv", index=False)
print(op_df.to_string(index=False))
print(f"\nF1-optimal threshold on test (for display only): {opt_threshold:.3f}")
print(f"Threshold actually adopted, tuned on validation: {TEST_PREDICTIONS[best_overall_id] and exp_df.iloc[0]['tuned_threshold']:.3f}")
