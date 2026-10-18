# ----------------------------------------------------------------------------------------------
# Controlled ablation: one variable changed at a time.
# ----------------------------------------------------------------------------------------------
set_global_seed()

ABLATION = []


def ablate(label: str, model, X_tr, X_te, note: str = ""):
    """Fit one ablation model and record its test metrics (no Table II entry)."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_train)
    elapsed = time.perf_counter() - t0
    proba = model.predict_proba(X_te)[:, 1]
    m = metrics_at(y_test, proba)
    ABLATION.append({
        "run": label, "features": X_tr.shape[1], "roc_auc": round(m["roc_auc"], 4),
        "pr_auc": round(m["pr_auc"], 4), "f1": round(m["f1"], 4),
        "recall": round(m["recall"], 4), "fit_time_s": round(elapsed, 1), "note": note,
    })
    print(f"  {label:<34} features={X_tr.shape[1]:>2}  ROC-AUC {m['roc_auc']:.4f}  "
          f"PR-AUC {m['pr_auc']:.4f}  F1 {m['f1']:.4f}  ({elapsed:.1f}s)")
    return proba


print("A1-A2  Logistic Regression")
ablate("A1 LR | raw features", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                 random_state=SEED), Xtr_s[:, :len(FEATURES)], Xte_s[:, :len(FEATURES)])
ablate("A2 LR | raw + engineered", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                     random_state=SEED), Xtr_s, Xte_s)

print("A3-A4  Random Forest (100 trees, shared configuration)")
rf_fixed = dict(n_estimators=100, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=1)
ablate("A3 RF | raw features", RandomForestClassifier(**rf_fixed), X_train, X_test)
ablate("A4 RF | raw + engineered", RandomForestClassifier(**rf_fixed), X_train_fe, X_test_fe)

print("A5-A6  Histogram Gradient Boosting (shared configuration)")
hgb_fixed = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=40, l2_regularization=1.0,
                 min_samples_leaf=40, random_state=SEED)
ablate("A5 HGB | raw features", HistGradientBoostingClassifier(**hgb_fixed), X_train, X_test)
ablate("A6 HGB | raw + engineered", HistGradientBoostingClassifier(**hgb_fixed), X_train_fe, X_test_fe)

print("A7     Effect of standardisation on the linear model")
ablate("A7 LR | engineered, unscaled", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                         random_state=SEED), X_train_fe, X_test_fe,
       note="no StandardScaler; compared with A2")

print("A8     Feature selection: drop the 5 lowest-MI features")
drop5 = assoc.tail(5)["feature"].tolist()
keep = [c for c in FULL_FEATURES if c not in drop5]
ablate("A8 HGB | engineered minus 5", HistGradientBoostingClassifier(**hgb_fixed),
       X_train_fe[keep], X_test_fe[keep], note=f"dropped: {', '.join(drop5)}")

print("A9     Sensitivity: exact duplicate response profiles removed (Section 2.4 finding)")
dedup = raw.drop_duplicates(subset=FEATURES, keep="first").reset_index(drop=True)
Xd, yd = dedup[FEATURES], dedup[TARGET].astype(int)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.30, stratify=yd, random_state=SEED)
engineer_dedup = FeatureEngineer().fit(Xd_tr)
t0 = time.perf_counter()
hgb_dedup = HistGradientBoostingClassifier(**hgb_fixed).fit(engineer_dedup.transform(Xd_tr), yd_tr)
proba_dedup = hgb_dedup.predict_proba(engineer_dedup.transform(Xd_te))[:, 1]
elapsed = time.perf_counter() - t0
m_dedup = metrics_at(yd_te, proba_dedup)
ABLATION.append({
    "run": "A9 HGB | duplicates removed", "features": Xd_tr.shape[1] + len(ENGINEERED),
    "roc_auc": round(m_dedup["roc_auc"], 4), "pr_auc": round(m_dedup["pr_auc"], 4),
    "f1": round(m_dedup["f1"], 4), "recall": round(m_dedup["recall"], 4),
    "fit_time_s": round(elapsed, 1),
    "note": f"{len(dedup):,} rows after removing {len(raw)-len(dedup):,} repeats; compared with A6",
})
print(f"  {'A9 HGB | duplicates removed':<34} features={Xd_tr.shape[1]+len(ENGINEERED):>2}  "
      f"ROC-AUC {m_dedup['roc_auc']:.4f}  PR-AUC {m_dedup['pr_auc']:.4f}  F1 {m_dedup['f1']:.4f}  "
      f"({elapsed:.1f}s)")

ablation_df = pd.DataFrame(ABLATION)
ablation_df.to_csv(RESULT_DIR / "table3_ablation.csv", index=False)


def delta(a: str, b: str, metric: str = "pr_auc") -> float:
    va = ablation_df.loc[ablation_df.run == a, metric].iloc[0]
    vb = ablation_df.loc[ablation_df.run == b, metric].iloc[0]
    return float(vb - va)


print("\nPaired effect of feature engineering (PR-AUC, engineered minus raw):")
for raw_run, fe_run in [("A1 LR | raw features", "A2 LR | raw + engineered"),
                        ("A3 RF | raw features", "A4 RF | raw + engineered"),
                        ("A5 HGB | raw features", "A6 HGB | raw + engineered")]:
    print(f"    {raw_run.split('|')[0].strip():<5} {delta(raw_run, fe_run):+.4f}")

print(f"\nA9 duplicate sensitivity (A9 minus A6, PR-AUC): "
      f"{delta('A6 HGB | raw + engineered', 'A9 HGB | duplicates removed'):+.4f}")
print("A shift of a few thousandths means the duplicate profiles were not inflating the headline result;")
print("a large negative shift would mean the original figure was partly memorisation.")
