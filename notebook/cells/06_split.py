# ----------------------------------------------------------------------------------------------
# Stratified 70/15/15 split.  Train -> fit.  Validation -> early stopping, tuning, threshold.
# Test -> touched once, in Section 9.
# ----------------------------------------------------------------------------------------------
X_all = raw[FEATURES].copy()
y_all = raw[TARGET].astype(int).copy()

X_train, X_hold, y_train, y_hold = train_test_split(
    X_all, y_all, test_size=0.30, stratify=y_all, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_hold, y_hold, test_size=0.50, stratify=y_hold, random_state=SEED
)

split_summary = pd.DataFrame({
    "partition": ["train", "validation", "test"],
    "rows": [len(X_train), len(X_val), len(X_test)],
    "share": [len(X_train) / len(X_all), len(X_val) / len(X_all), len(X_test) / len(X_all)],
    "positives": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())],
    "prevalence": [y_train.mean(), y_val.mean(), y_test.mean()],
})
print(split_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---- assertions: the split must be exactly what the report claims ------------------------------
assert len(X_train) + len(X_val) + len(X_test) == len(X_all)
assert len(set(X_train.index) & set(X_val.index)) == 0
assert len(set(X_train.index) & set(X_test.index)) == 0
assert len(set(X_val.index) & set(X_test.index)) == 0
for s in split_summary["prevalence"]:
    assert abs(s - y_all.mean()) < 0.002, "stratification drifted"
print("\nPartitions are disjoint, exhaustive and equally stratified.")
