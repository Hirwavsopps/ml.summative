# ----------------------------------------------------------------------------------------------
# Figure 2: Spearman correlation structure.  Figure 3: univariate signal per feature.
# ----------------------------------------------------------------------------------------------
corr_cols = FEATURES + [TARGET]
spearman = raw[corr_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(10.2, 8.4))
mask = np.triu(np.ones_like(spearman, dtype=bool), k=1)
sns.heatmap(
    spearman, mask=mask, cmap="RdBu_r", center=0, vmin=-0.45, vmax=0.45,
    square=True, linewidths=0.4, linecolor="white", ax=ax,
    cbar_kws={"shrink": 0.72, "label": "Spearman rank correlation"},
    annot=False,
)
ax.set_title("Figure 2 — Spearman correlation among predictors and the target")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_correlation_heatmap.png")
# ---- strongest off-diagonal feature pairs -----------------------------------------------------
pairs = (
    spearman.loc[FEATURES, FEATURES]
    .where(~np.eye(len(FEATURES), dtype=bool))
    .stack()
    .rename("rho")
    .reset_index()
    .rename(columns={"level_0": "a", "level_1": "b"})
)
pairs["abs_rho"] = pairs["rho"].abs()
pairs = pairs[pairs["a"] < pairs["b"]].sort_values("abs_rho", ascending=False).head(6)
print("Strongest inter-feature associations (Spearman):")
print(pairs[["a", "b", "rho"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# ---- univariate association with the target ---------------------------------------------------
# Mutual information needs a manageable sample; 40k rows gives a stable estimate and runs in seconds.
mi_sample = raw.sample(n=40_000, random_state=SEED)
mi = mutual_info_classif(
    mi_sample[FEATURES], mi_sample[TARGET],
    discrete_features=True, n_neighbors=3, random_state=SEED,
)
assoc = pd.DataFrame({
    "feature": FEATURES,
    "mutual_info": mi,
    "spearman_rho": [spearman.loc[f, TARGET] for f in FEATURES],
})
assoc["abs_rho"] = assoc["spearman_rho"].abs()
assoc = assoc.sort_values("mutual_info", ascending=False).reset_index(drop=True)
assoc.to_csv(RESULT_DIR / "feature_association.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(6.8, 10.4))
shown = assoc.iloc[::-1]

axes[0].barh(shown["feature"], shown["mutual_info"], color="#4C72B0", edgecolor="black", linewidth=0.4)
axes[0].set_xlabel("Mutual information with target (nats)")
axes[0].set_title("(a) Any dependence — mutual information")
axes[0].tick_params(axis="y", labelsize=8.5)

colors = ["#C44E52" if r > 0 else "#4C72B0" for r in shown["spearman_rho"]]
axes[1].barh(shown["feature"], shown["spearman_rho"], color=colors, edgecolor="black", linewidth=0.4)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Spearman rank correlation with target")
axes[1].set_title("(b) Monotone dependence only — Spearman rho")
axes[1].tick_params(axis="y", labelsize=8.5)

fig.suptitle("Figure 3 — Univariate signal, two complementary screens", fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_feature_association.png")
print("\nFeature ranking by mutual information (top 12):")
print(assoc.head(12)[["feature", "mutual_info", "spearman_rho"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

weaker = assoc.tail(6)["feature"].tolist()
print(f"\nWeakest six features by mutual information: {', '.join(weaker)}")
print("These are candidates for the redundancy analysis in Section 9.6, not for automatic deletion:")
print("a feature can be weak alone and useful in interaction.")
