# ----------------------------------------------------------------------------------------------
# E2 -- Random Forest, out-of-the-box hyperparameters, no class weighting.
# ----------------------------------------------------------------------------------------------
set_global_seed()

t0 = time.perf_counter()
rf_base = RandomForestClassifier(
    n_estimators=100, max_depth=None, max_features="sqrt",
    min_samples_leaf=1, random_state=SEED, n_jobs=1,
)
rf_base.fit(X_train_fe, y_train)
fit_time = time.perf_counter() - t0

val_proba_rf = rf_base.predict_proba(X_val_fe)[:, 1]
test_proba_rf = rf_base.predict_proba(X_test_fe)[:, 1]

record_experiment(
    exp_id="E2",
    name="Random Forest (default)",
    paradigm="Classical",
    y_true=y_test,
    proba=test_proba_rf,
    fit_time=fit_time,
    hyperparams="n_estimators=100, max_depth=None, max_features=sqrt, min_samples_leaf=1, no class weights",
    n_params=int(sum(t.get_n_leaves() for t in rf_base.estimators_)),
    notes="Bagged deep trees; unpruned, so variance reduction comes only from averaging.",
    val_proba=val_proba_rf,
    y_val_=y_val,
)

train_proba_rf = rf_base.predict_proba(X_train_fe)[:, 1]
print(f"\nTrain ROC-AUC {roc_auc_score(y_train, train_proba_rf):.4f} vs "
      f"test ROC-AUC {roc_auc_score(y_test, test_proba_rf):.4f} "
      f"-> generalisation gap {roc_auc_score(y_train, train_proba_rf) - roc_auc_score(y_test, test_proba_rf):.4f}")
