# ----------------------------------------------------------------------------------------------
# Save the two required models, the preprocessor and run metadata; then verify the round-trip.
# ----------------------------------------------------------------------------------------------
import joblib

FITTED_MODELS = {
    "E1": lr, "E2": rf_base, "E3": rf_tuned, "E4": hgb, "E5": svm_model,
    "E6": mlp_base, "E7": mlp_reg, "E8": emb_model, "E9": net_e9,
}

best_classical_id = exp_df[exp_df.paradigm == "Classical"].iloc[0]["exp_id"]
best_dl_id = exp_df[exp_df.paradigm == "Deep Learning"].iloc[0]["exp_id"]
best_classical_model = FITTED_MODELS[best_classical_id]
best_dl_model = FITTED_MODELS[best_dl_id]


def predict_with_bundle(bundle: dict, X_raw: pd.DataFrame) -> np.ndarray:
    """Score raw survey rows end to end: engineer -> scale -> estimator -> probability."""
    X_fe = bundle["feature_engineer"].transform(X_raw[bundle["raw_feature_order"]])
    # Only the models that were *trained* on standardised inputs may be scored on standardised
    # inputs. Applying the scaler unconditionally shifts the values by which a histogram-based
    # booster assigns its bins, which silently changes every prediction -- the round-trip check
    # below is what caught exactly that.
    X = bundle["scaler"].transform(X_fe[bundle["feature_order"]]) if bundle["uses_scaling"] else X_fe[bundle["feature_order"]]
    est = bundle["estimator"]
    if bundle.get("calibrator") is not None:          # SVM path: Platt-calibrated decision values
        scores = est.decision_function(X).reshape(-1, 1)
        return bundle["calibrator"].predict_proba(scores)[:, 1]
    return est.predict_proba(X)[:, 1]


# ---- traditional model: bundle estimator + preprocessing so one call accepts raw rows ----------
classical_bundle = {
    "estimator": best_classical_model,
    "feature_engineer": engineer,
    "scaler": SCALERS["full"],
    "feature_order": FULL_FEATURES,
    "raw_feature_order": FEATURES,
    "calibrator": calibrator if best_classical_id == "E5" else None,
    "uses_scaling": best_classical_id in ("E1", "E5"),
    "decision_threshold": float(exp_df.loc[exp_df.exp_id == best_classical_id, "tuned_threshold"].iloc[0]),
    "exp_id": best_classical_id,
    "test_metrics": exp_df.loc[exp_df.exp_id == best_classical_id].drop(columns=["hyperparameters"]).to_dict(),
}
joblib.dump(classical_bundle, MODEL_DIR / "best_traditional.joblib")

# ---- deep model -------------------------------------------------------------------------------
# Two artefacts, deliberately.  The SavedModel export is the portable one; the architecture JSON
# plus .weights.h5 pair is the one that rebuilds in a fresh process without TF's signature
# indirection.  (Keras 3's single-file `.keras` archive writer stalls in this environment even for a
# three-layer Sequential model, so it is avoided rather than depended on.)
DL_WEIGHTS = MODEL_DIR / "best_deep_learning.weights.h5"
DL_CONFIG = MODEL_DIR / "best_deep_learning_config.json"
DL_SAVEDMODEL = MODEL_DIR / "best_deep_learning_savedmodel"

best_dl_model.save_weights(str(DL_WEIGHTS))
DL_CONFIG.write_text(json.dumps(best_dl_model.get_config(), indent=2), encoding="utf-8")
tf.saved_model.save(best_dl_model, str(DL_SAVEDMODEL))

joblib.dump({"feature_engineer": engineer, "scaler": SCALERS["full"]}, MODEL_DIR / "preprocessor.joblib")

run_metadata = {
    "student": "Crispin Hebert Hirwa",
    "course": "Introduction to Machine Learning - Summative Assignment",
    "dataset": DATASET_META,
    "dataset_sha256": RAW_SHA256,
    "seed": SEED,
    "versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    },
    "split": {
        "train": len(X_train), "validation": len(X_val), "test": len(X_test),
        "train_prevalence": round(float(y_train.mean()), 5),
        "test_prevalence": round(float(y_test.mean()), 5),
    },
    "feature_sets": {"raw": FEATURES, "engineered": ENGINEERED, "full": FULL_FEATURES},
    "class_weights": {str(k): round(v, 4) for k, v in CLASS_WEIGHT.items()},
    "best_traditional": {"exp_id": best_classical_id,
                         "model": exp_df.loc[exp_df.exp_id == best_classical_id, "model"].iloc[0]},
    "best_deep_learning": {"exp_id": best_dl_id,
                           "model": exp_df.loc[exp_df.exp_id == best_dl_id, "model"].iloc[0]},
    "deep_learning_files": {
        "config": DL_CONFIG.name, "weights": DL_WEIGHTS.name, "saved_model": DL_SAVEDMODEL.name,
    },
    "experiments": exp_df.to_dict(orient="records"),
}
with open(MODEL_DIR / "run_metadata.json", "w", encoding="utf-8") as fh:
    json.dump(run_metadata, fh, indent=2)

print(f"Best traditional ({best_classical_id}) -> models/best_traditional.joblib")
print(f"Best deep learning ({best_dl_id}) -> models/best_deep_learning_config.json")
print("                                   -> models/best_deep_learning.weights.h5")
print("                                   -> models/best_deep_learning_savedmodel/")
print("Preprocessor                        -> models/preprocessor.joblib")
print("Run metadata                        -> models/run_metadata.json")

# ---- verify the round trip ---------------------------------------------------------------------
print("\nRound-trip verification on the first 500 test rows:")

bundle = joblib.load(MODEL_DIR / "best_traditional.joblib")
reloaded_proba = predict_with_bundle(bundle, X_test.iloc[:500])
original_proba = TEST_PREDICTIONS[best_classical_id]["proba"][:500]
classical_delta = float(np.abs(reloaded_proba - original_proba).max())
print(f"  traditional : max |delta| = {classical_delta:.2e} -> "
      f"{'identical' if np.allclose(reloaded_proba, original_proba, atol=1e-9) else 'MISMATCH'}")

# Rebuild the network from its saved architecture and weights, in this same process, and confirm it
# reproduces the recorded test probabilities exactly.
rebuilt = keras.Model.from_config(json.loads(DL_CONFIG.read_text(encoding="utf-8")))
rebuilt.load_weights(str(DL_WEIGHTS))

if best_dl_id == "E8":
    dl_input = test_ds_emb
elif best_dl_id == "E9":
    dl_input = test_ds_emb if use_embeddings else test_ds
else:
    dl_input = test_ds

rebuilt_proba = rebuilt.predict(dl_input, verbose=0).ravel()[:500]
dl_delta = float(np.abs(rebuilt_proba - TEST_PREDICTIONS[best_dl_id]["proba"][:500]).max())
print(f"  deep learning: max |delta| = {dl_delta:.2e} -> "
      f"{'identical' if np.allclose(rebuilt_proba, TEST_PREDICTIONS[best_dl_id]['proba'][:500], atol=1e-6) else 'MISMATCH'}")

saved_model_size = sum(p.stat().st_size for p in DL_SAVEDMODEL.rglob("*") if p.is_file())
print(f"  SavedModel export: {saved_model_size/1e6:.2f} MB, loadable with tf.saved_model.load()")

print("\nSaved files:")
for p in sorted(MODEL_DIR.iterdir()):
    size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    print(f"  {p.name:<40} {size/1e6:.2f} MB")
