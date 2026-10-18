# ----------------------------------------------------------------------------------------------
# Figure 4 -- learning curves for the three classical families.
# ----------------------------------------------------------------------------------------------
set_global_seed()

LC_N = 40_000
lc_idx = train_test_split(
    np.arange(len(Xtr_s)), train_size=LC_N, stratify=y_train, random_state=SEED,
)[0]
X_lc, y_lc = Xtr_s[lc_idx], y_train.iloc[lc_idx].to_numpy()
X_lc_fe, y_lc_fe = X_train_fe.iloc[lc_idx], y_train.iloc[lc_idx].to_numpy()

lc_models = {
    "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
                            X_lc_fe, y_lc_fe),
    "Random Forest": (RandomForestClassifier(n_estimators=60, min_samples_leaf=3, max_features="sqrt",
                                             random_state=SEED, n_jobs=1), X_lc_fe, y_lc_fe),
    "Hist. Gradient Boosting": (HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                                               max_leaf_nodes=40, l2_regularization=1.0,
                                                               min_samples_leaf=40, random_state=SEED),
                                X_lc_fe, y_lc_fe),
}

train_sizes_abs = np.linspace(0.15, 1.0, 4)
lc_records = []
fig, axes = plt.subplots(3, 1, figsize=(6.8, 11.0))

for ax, (label, (est, Xl, yl)) in zip(axes, lc_models.items()):
    t0 = time.perf_counter()
    sizes, tr_scores, va_scores = learning_curve(
        est, Xl, yl, train_sizes=train_sizes_abs, cv=3,
        scoring="average_precision", shuffle=True, random_state=SEED, n_jobs=1,
    )
    elapsed = time.perf_counter() - t0
    tr_m, tr_s = tr_scores.mean(axis=1), tr_scores.std(axis=1)
    va_m, va_s = va_scores.mean(axis=1), va_scores.std(axis=1)

    ax.plot(sizes, tr_m, "o-", color="#C44E52", label="Training PR-AUC", linewidth=1.9, markersize=5)
    ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, color="#C44E52", alpha=0.16)
    ax.plot(sizes, va_m, "s-", color="#4C72B0", label="Validation PR-AUC", linewidth=1.9, markersize=5)
    ax.fill_between(sizes, va_m - va_s, va_m + va_s, color="#4C72B0", alpha=0.16)

    ax.set_xlabel("Training examples")
    ax.set_ylabel("PR-AUC (3-fold CV)")
    ax.set_title(f"{label}\n({elapsed:.0f}s)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_ylim(min(0.20, va_m.min() - 0.05), 1.0)

    gap = tr_m[-1] - va_m[-1]
    verdict = ("underfitting" if va_m[-1] < 0.35 and gap < 0.10
               else "overfitting" if gap > 0.25
               else "mild variance gap")
    lc_records.append({
        "model": label,
        "final_train_pr_auc": round(float(tr_m[-1]), 4),
        "final_val_pr_auc": round(float(va_m[-1]), 4),
        "gap": round(float(gap), 4),
        "val_slope_last": round(float(va_m[-1] - va_m[-2]), 4),
        "diagnosis": verdict,
        "seconds": round(elapsed, 1),
    })
    print(f"{label:<26} train {tr_m[-1]:.4f} | val {va_m[-1]:.4f} | gap {gap:+.4f} | "
          f"val still rising by {va_m[-1]-va_m[-2]:+.4f} at the last point -> {verdict}")

fig.suptitle("Figure 4 — Learning curves separate the bias-limited model from the variance-limited ones",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_learning_curves_sklearn.png")
lc_df = pd.DataFrame(lc_records)
lc_df.to_csv(RESULT_DIR / "table4_learning_curves.csv", index=False)
print()
print(lc_df.to_string(index=False))
