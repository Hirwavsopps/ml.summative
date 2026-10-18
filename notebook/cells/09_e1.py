# ----------------------------------------------------------------------------------------------
# E1 -- Logistic Regression, balanced class weights, all 30 features, standardised.
# ----------------------------------------------------------------------------------------------
from sklearn.metrics import log_loss

set_global_seed()
Xtr_s, Xval_s, Xte_s = scaled_versions("full")

t0 = time.perf_counter()
lr = LogisticRegression(
    penalty="l2", C=1.0, solver="lbfgs", max_iter=3000,
    class_weight="balanced", random_state=SEED,
)
lr.fit(Xtr_s, y_train)
fit_time = time.perf_counter() - t0

val_proba_lr = lr.predict_proba(Xval_s)[:, 1]
test_proba_lr = lr.predict_proba(Xte_s)[:, 1]

record_experiment(
    exp_id="E1",
    name="Logistic Regression",
    paradigm="Classical",
    y_true=y_test,
    proba=test_proba_lr,
    fit_time=fit_time,
    hyperparams=("penalty=l2, C=1.0, solver=lbfgs, max_iter=3000, class_weight=balanced, "
                 "StandardScaler on 30 features"),
    n_params=int(lr.coef_.size + lr.intercept_.size),
    notes="Linear boundary in log-odds space; coefficients are directly interpretable as adjusted log-ORs.",
    val_proba=val_proba_lr,
    y_val_=y_val,
)

# ---- interpretation: the ten largest standardised coefficients ---------------------------------
coefs = pd.Series(lr.coef_[0], index=FULL_FEATURES).sort_values(key=np.abs, ascending=False)
print("\nLargest standardised log-odds coefficients (positive -> increases diabetes odds):")
print(coefs.head(10).to_string(float_format=lambda x: f"{x:+.4f}"))
print("\nOdds-ratio equivalents for the same ten:", )
print(np.exp(coefs.head(10)).round(3).to_string())
