# ----------------------------------------------------------------------------------------------
# Figure 6 -- row-normalised confusion matrices, raw counts annotated.
# ----------------------------------------------------------------------------------------------
best_classical_id = exp_df[exp_df.paradigm == "Classical"].iloc[0]["exp_id"]
best_dl_id = exp_df[exp_df.paradigm == "Deep Learning"].iloc[0]["exp_id"]
best_overall_id = exp_df.iloc[0]["exp_id"]

panel_ids = [pid for pid in ["E1", "E6", best_classical_id, best_overall_id]]
panel_ids = list(dict.fromkeys(panel_ids))          # de-duplicate while preserving order

n_panels = len(panel_ids)
n_rows = 2 if n_panels > 1 else 1
n_cols = int(np.ceil(n_panels / n_rows))
fig, _axes = plt.subplots(n_rows, n_cols, figsize=(3.35 * n_cols, 3.35 * n_rows))
_all_axes = np.ravel(np.atleast_1d(_axes))
for extra in _all_axes[n_panels:]:        # a 2x2 grid with three panels leaves one cell empty
    extra.axis("off")
axes = _all_axes[:n_panels]

cm_summary = []
for ax, pid in zip(axes, panel_ids):
    rec = TEST_PREDICTIONS[pid]
    pred = (rec["proba"] >= 0.5).astype(int)
    cm = confusion_matrix(rec["y_true"], pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=False, cmap="Blues", vmin=0, vmax=1, cbar=False,
                square=True, linewidths=1.4, linecolor="white", ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]:,}\n{cm_norm[i, j]:.1%}",
                    ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color="white" if cm_norm[i, j] > 0.55 else "black")
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
    ax.set_yticklabels(["True 0", "True 1"], fontsize=9, rotation=0)
    ax.set_title(f"{pid}: {rec['name']}", fontsize=9.5)

    fn = int(cm[1, 0])
    fp = int(cm[0, 1])
    cm_summary.append({
        "exp_id": pid, "model": rec["name"],
        "true_pos": int(cm[1, 1]), "false_neg": fn, "false_pos": fp, "true_neg": int(cm[0, 0]),
        "sensitivity": round(cm[1, 1] / cm[1].sum(), 4),
        "specificity": round(cm[0, 0] / cm[0].sum(), 4),
        "missed_cases_per_1000": round(1000 * fn / cm[1].sum(), 1),
    })

fig.suptitle("Figure 6 — Confusion matrices at threshold 0.5 (row-normalised, raw counts annotated)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_confusion_matrices.png")
cm_df = pd.DataFrame(cm_summary)
cm_df.to_csv(RESULT_DIR / "confusion_summary.csv", index=False)
print(cm_df.to_string(index=False))
print("\nReading the last row: 'missed_cases_per_1000' is the number of true diabetics the model would")
print("fail to flag per 1,000 diabetics screened -- the operationally meaningful cost of the threshold.")
