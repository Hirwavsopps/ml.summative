# ----------------------------------------------------------------------------------------------
# E3 -- Random Forest with randomised hyperparameter search (PR-AUC objective).
# ----------------------------------------------------------------------------------------------
from scipy.stats import randint

set_global_seed()

TUNE_N = 30_000
tune_idx = train_test_split(
    np.arange(len(X_train_fe)), train_size=TUNE_N,
    stratify=y_train, random_state=SEED,
)[0]
X_tune, y_tune = X_train_fe.iloc[tune_idx], y_train.iloc[tune_idx]

param_dist = {
    "n_estimators": randint(120, 260),
    "max_depth": [None, 12, 20, 30],
    "min_samples_leaf": randint(1, 12),
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "class_weight": [None, "balanced_subsample"],
}

search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=SEED, n_jobs=1, bootstrap=True),
    param_distributions=param_dist,
    n_iter=6, scoring="average_precision", cv=2,
    random_state=SEED, n_jobs=1, verbose=0, refit=True,
)

t0 = time.perf_counter()
search.fit(X_tune, y_tune)
search_time = time.perf_counter() - t0

best_params = search.best_params_
print(f"Randomised search: {search_time:.1f}s over 6 candidates x 2 folds on {TUNE_N:,} rows")
print(f"Best CV PR-AUC  : {search.best_score_:.4f}")
print("Best configuration:")
for k, v in best_params.items():
    print(f"    {k:<18}= {v}")

# ---- refit the winning configuration on the FULL training partition ----------------------------
t0 = time.perf_counter()
rf_tuned = RandomForestClassifier(random_state=SEED, n_jobs=1, **best_params)
rf_tuned.fit(X_train_fe, y_train)
fit_time = time.perf_counter() - t0

val_proba_rft = rf_tuned.predict_proba(X_val_fe)[:, 1]
test_proba_rft = rf_tuned.predict_proba(X_test_fe)[:, 1]

record_experiment(
    exp_id="E3",
    name="Random Forest (tuned)",
    paradigm="Classical",
    y_true=y_test,
    proba=test_proba_rft,
    fit_time=fit_time + search_time,
    hyperparams=", ".join(f"{k}={v}" for k, v in best_params.items()) + " (RandomizedSearchCV, 6x2, PR-AUC)",
    n_params=int(sum(t.get_n_leaves() for t in rf_tuned.estimators_)),
    notes="Same model class as E2; the only change is systematic hyperparameter search plus full-data refit.",
    val_proba=val_proba_rft,
    y_val_=y_val,
)
