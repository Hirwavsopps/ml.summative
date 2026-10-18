# ----------------------------------------------------------------------------------------------
# E4 -- Histogram Gradient Boosting with randomised search over four regularisation knobs.
# ----------------------------------------------------------------------------------------------
from scipy.stats import loguniform, uniform

set_global_seed()

GB_N = 100_000
gb_idx = train_test_split(
    np.arange(len(X_train_fe)), train_size=GB_N,
    stratify=y_train, random_state=SEED,
)[0]
X_gb, y_gb = X_train_fe.iloc[gb_idx], y_train.iloc[gb_idx]

gb_dist = {
    "learning_rate": loguniform(0.02, 0.30),
    "max_leaf_nodes": randint(15, 90),
    "max_iter": randint(150, 450),
    "l2_regularization": loguniform(1e-3, 10.0),
    "min_samples_leaf": randint(20, 120),
}

gb_search = RandomizedSearchCV(
    estimator=HistGradientBoostingClassifier(random_state=SEED, early_stopping=True,
                                             validation_fraction=0.1, n_iter_no_change=20),
    param_distributions=gb_dist,
    n_iter=10, scoring="average_precision", cv=3,
    random_state=SEED, n_jobs=1, verbose=0,
)

t0 = time.perf_counter()
gb_search.fit(X_gb, y_gb)
gb_search_time = time.perf_counter() - t0

gb_best = gb_search.best_params_
print(f"Randomised search: {gb_search_time:.1f}s over 10 candidates x 3 folds on {GB_N:,} rows")
print(f"Best CV PR-AUC  : {gb_search.best_score_:.4f}")
for k, v in gb_best.items():
    print(f"    {k:<18}= {v if isinstance(v, (int, str)) else round(v, 5)}")

t0 = time.perf_counter()
hgb = HistGradientBoostingClassifier(
    random_state=SEED, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, **gb_best
)
hgb.fit(X_train_fe, y_train)
fit_time = time.perf_counter() - t0

val_proba_gb = hgb.predict_proba(X_val_fe)[:, 1]
test_proba_gb = hgb.predict_proba(X_test_fe)[:, 1]

record_experiment(
    exp_id="E4",
    name="Histogram Gradient Boosting (tuned)",
    paradigm="Classical",
    y_true=y_test,
    proba=test_proba_gb,
    fit_time=fit_time + gb_search_time,
    hyperparams=", ".join(f"{k}={v if isinstance(v, (int, str)) else round(v, 5)}"
                          for k, v in gb_best.items()) + " (RandomizedSearchCV, 10x3, PR-AUC)",
    n_params=int(getattr(hgb, "n_iter_", gb_best.get("max_iter", 0))),
    notes="Sequential residual fitting; boosting rounds used as the capacity measure.",
    val_proba=val_proba_gb,
    y_val_=y_val,
)
print(f"\nBoosting rounds actually used (early stopping): {getattr(hgb, 'n_iter_', 'n/a')}")
