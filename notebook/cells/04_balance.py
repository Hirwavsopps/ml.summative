# ----------------------------------------------------------------------------------------------
# Table I (variable dictionary) and Figure 1 (class distribution).
# ----------------------------------------------------------------------------------------------
table1 = pd.DataFrame([
    {
        "Variable": var,
        "Type": typ,
        "Description": desc,
        "Values": (f"{sorted(allowed)[0]}-{sorted(allowed)[-1]}"
                   if allowed is not None else f"{raw[var].min()}-{raw[var].max()}"),
        "Mean": float(raw[var].mean()),
        "Role": "Target" if var == TARGET else "Feature",
    }
    for var, (typ, desc, allowed) in VARIABLE_SPEC.items()
])
table1.to_csv(RESULT_DIR / "table1_dataset_summary.csv", index=False)
print("Table I written to results/table1_dataset_summary.csv")
print(table1.to_string(index=False))

# ---- class balance ----------------------------------------------------------------------------
counts = raw[TARGET].value_counts().sort_index()
prev = raw[TARGET].mean()
labels = ["No diabetes (0)", "Diabetes / pre-diabetes (1)"]
colors = ["#4C72B0", "#C44E52"]

fig, ax = plt.subplots(figsize=(6.6, 4.3))

bars = ax.bar(labels, counts.values, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
for bar, c in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, c + 4000, f"{c:,}\n({c/len(raw):.2%})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Respondents")
ax.set_ylim(0, counts.max() * 1.30)
ax.set_title("Figure 1 — Class distribution in the raw cohort")
ax.annotate(
    f"Imbalance ratio {counts[0]/counts[1]:.2f} : 1\n"
    f"Majority-class baseline accuracy {max(prev, 1-prev):.4f}\n"
    f"(a model that always predicts 0 scores this)",
    xy=(0.03, 0.80), xycoords="axes fraction", fontsize=8.5, style="italic",
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.6", "linewidth": 0.7},
)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_class_distribution.png")
print(f"\nPositive prevalence          : {prev:.4f} ({prev:.2%})")
print(f"Imbalance ratio              : {counts[0]/counts[1]:.2f} : 1")
print(f"Majority-class-only accuracy : {max(prev, 1-prev):.4f}   <-- the floor any useful model must clear")
