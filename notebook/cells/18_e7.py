# ----------------------------------------------------------------------------------------------
# E7 -- Deeper MLP with dropout, batch normalisation and early stopping.
# ----------------------------------------------------------------------------------------------
set_global_seed()

def build_regularised_mlp(n_in: int, dropout: float = 0.30, name: str = "E7_MLP_regularised"):
    """128-64-32 with BatchNorm -> ReLU -> Dropout blocks."""
    inputs = keras.Input(shape=(n_in,), name="features")
    x = inputs
    for units, tag in [(128, "1"), (64, "2"), (32, "3")]:
        x = layers.Dense(units, name=f"dense_{tag}")(x)
        x = layers.BatchNormalization(name=f"bn_{tag}")(x)
        x = layers.Activation("relu", name=f"relu_{tag}")(x)
        x = layers.Dropout(dropout, name=f"drop_{tag}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs, outputs, name=name)


mlp_reg = build_regularised_mlp(n_features)
hist_e7, time_e7 = train_network(mlp_reg, "E7", train_ds, val_ds)
mlp_reg.summary()

val_proba_e7 = mlp_reg.predict(val_ds, verbose=0).ravel()
test_proba_e7 = mlp_reg.predict(test_ds, verbose=0).ravel()

record_experiment(
    exp_id="E7",
    name="MLP 128-64-32 + BN/Dropout",
    paradigm="Deep Learning",
    y_true=y_test,
    proba=test_proba_e7,
    fit_time=time_e7,
    hyperparams="Dense(128,64,32) each BN+ReLU+Dropout(0.3), Adam lr=1e-3 + ReduceLROnPlateau(0.5,2), "
                "early stop on val_pr_auc patience=6, batch=1024, no class weight",
    n_params=int(mlp_reg.count_params()),
    notes="Regularised counterpart to E6; identical loss and optimiser so the comparison isolates regularisation.",
    val_proba=val_proba_e7,
    y_val_=y_val,
)

train_proba_e7 = mlp_reg.predict(train_ds, verbose=0).ravel()
print(f"\nE6 gap (train-test PR-AUC): {average_precision_score(y_train, train_proba_e6) - average_precision_score(y_test, test_proba_e6):+.4f}")
print(f"E7 gap (train-test PR-AUC): {average_precision_score(y_train, train_proba_e7) - average_precision_score(y_test, test_proba_e7):+.4f}")
print(f"E7 stopped after {len(hist_e7.history['loss'])} of {EPOCHS} permitted epochs.")
