# ----------------------------------------------------------------------------------------------
# E5 -- RBF SVM on a 25k stratified subsample, Platt-calibrated on the validation partition.
# ----------------------------------------------------------------------------------------------
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

set_global_seed()

SVM_N = 25_000
svm_idx = train_test_split(
    np.arange(len(Xtr_s)), train_size=SVM_N,
    stratify=y_train, random_state=SEED,
)[0]
X_svm, y_svm = Xtr_s[svm_idx], y_train.iloc[svm_idx].to_numpy()

# ---- small, explicit exploration of the two knobs that matter for an RBF kernel ---------------
svm_grid = [
    {"kernel": "rbf", "C": 1.0,  "gamma": "scale"},
    {"kernel": "rbf", "C": 5.0,  "gamma": "scale"},
    {"kernel": "rbf", "C": 5.0,  "gamma": "auto"},
    {"kernel": "rbf", "C": 10.0, "gamma": "auto"},
]

svm_results, best_auc, svm_best = [], -np.inf, None
t0 = time.perf_counter()
for cfg in svm_grid:
    set_global_seed()
    candidate = SVC(cache_size=1500, class_weight="balanced", random_state=SEED, **cfg)
    candidate.fit(X_svm, y_svm)
    auc = roc_auc_score(y_val, candidate.decision_function(Xval_s))
    svm_results.append({**cfg, "val_roc_auc": round(auc, 4)})
    if auc > best_auc:
        best_auc, svm_best, svm_model = auc, cfg, candidate
svm_fit_time = time.perf_counter() - t0

print(f"SVM exploration: {len(svm_grid)} configurations on {SVM_N:,} rows in {svm_fit_time:.1f}s")
print(pd.DataFrame(svm_results).to_string(index=False))
print(f"\nSelected: kernel={svm_best['kernel']}, C={svm_best['C']}, gamma={svm_best['gamma']} "
      f"(validation ROC-AUC {best_auc:.4f})")

# ---- Platt calibration: one-dimensional logistic fit on the validation decision scores --------
calibrator = LogisticRegression(C=1e6, solver="lbfgs").fit(
    svm_model.decision_function(Xval_s).reshape(-1, 1), y_val
)
val_proba_svm = calibrator.predict_proba(svm_model.decision_function(Xval_s).reshape(-1, 1))[:, 1]
test_proba_svm = calibrator.predict_proba(svm_model.decision_function(Xte_s).reshape(-1, 1))[:, 1]

record_experiment(
    exp_id="E5",
    name="RBF-kernel SVM",
    paradigm="Classical",
    y_true=y_test,
    proba=test_proba_svm,
    fit_time=svm_fit_time,
    hyperparams=(f"kernel=rbf, C={svm_best['C']}, gamma={svm_best['gamma']}, class_weight=balanced, "
                 f"Platt calibration on val; trained on {SVM_N:,}-row subsample"),
    n_params=int(svm_model.support_vectors_.shape[0]),
    notes="Trained on a 25k subsample for tractability -> not like-for-like with E3/E4; optimistic estimate.",
    val_proba=val_proba_svm,
    y_val_=y_val,
)
print(f"\nSupport vectors used: {svm_model.support_vectors_.shape[0]:,} "
      f"({svm_model.support_vectors_.shape[0]/SVM_N:.1%} of the subsample)")
