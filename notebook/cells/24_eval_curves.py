# ----------------------------------------------------------------------------------------------
# Figure 5 -- loss and validation PR-AUC trajectories for E6-E9.
# ----------------------------------------------------------------------------------------------
histories = {
    "E6 MLP 64-64": hist_e6,
    "E7 MLP 128-64-32 + BN/Dropout": hist_e7,
    "E8 Embeddings (Functional)": hist_e8,
    "E9 Weighted + LR decay": hist_e9,
}

# Rows = one network per row, columns = loss | PR-AUC.  Column-shaped so the panel is legible at
# the single-column width used in the written report.
fig, axes = plt.subplots(len(histories), 2, figsize=(6.9, 2.85 * len(histories)))
curve_cols = [("loss", "Loss (binary cross-entropy)", "Train", "Validation"),
              ("pr_auc", "PR-AUC", "Train", "Validation")]

for row, (label, hist) in enumerate(histories.items()):
    h = hist.history
    epochs = np.arange(1, len(h["loss"]) + 1)
    for col, (key, ylabel, train_lbl, val_lbl) in enumerate(curve_cols):
        ax = axes[row, col]
        if key == "loss":
            ax.plot(epochs, h["loss"], color="#C44E52", linewidth=1.7, label=train_lbl)
            ax.plot(epochs, h["val_loss"], color="#4C72B0", linewidth=1.7, linestyle="--", label=val_lbl)
            best_epoch = int(np.argmin(h["val_loss"])) + 1
            ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.1)
            ax.set_yscale("log")
        else:
            ax.plot(epochs, h["pr_auc"], color="#C44E52", linewidth=1.7, label=train_lbl)
            ax.plot(epochs, h["val_pr_auc"], color="#4C72B0", linewidth=1.7, linestyle="--", label=val_lbl)
            best_epoch = int(np.argmax(h["val_pr_auc"])) + 1
            ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.1,
                       label=f"best epoch {best_epoch}")
            ax.set_ylim(0.15, 1.0)
        ax.set_title(f"{label}\n({key})", fontsize=8.4)
        ax.set_xlabel("Epoch", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.2)
        ax.tick_params(labelsize=7.6)
        ax.legend(fontsize=6.9, loc="best")

fig.suptitle("Figure 5 — Divergence between training and validation loss marks where overfitting begins",
             fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_keras_training_curves.png")
print("Network training summary:")
for label, hist in histories.items():
    h = hist.history
    print(f"  {label:<34} epochs {len(h['loss']):>2} | final train loss {h['loss'][-1]:.4f} | "
          f"final val loss {h['val_loss'][-1]:.4f} | best val PR-AUC {max(h['val_pr_auc']):.4f} "
          f"@ epoch {int(np.argmax(h['val_pr_auc']))+1}")
print("\nTensorBoard event files were written under logs/tensorboard/ -- one subdirectory per run.")
import os
for run in sorted(os.listdir(LOG_DIR / "tensorboard")):
    p = LOG_DIR / "tensorboard" / run
    n = len(os.listdir(p)) if p.is_dir() else 0
    print(f"    {run:<22} {n} event file(s)")
