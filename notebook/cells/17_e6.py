# ----------------------------------------------------------------------------------------------
# E6 -- Sequential API, 2 x 64 ReLU, no regularisation, unweighted loss.
# ----------------------------------------------------------------------------------------------
set_global_seed()

n_features = Xtr_dl.shape[1]
mlp_base = keras.Sequential(
    [
        keras.Input(shape=(n_features,), name="features"),
        layers.Dense(64, activation="relu", name="hidden_1"),
        layers.Dense(64, activation="relu", name="hidden_2"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ],
    name="E6_MLP_baseline",
)

hist_e6, time_e6 = train_network(mlp_base, "E6", train_ds, val_ds)
mlp_base.summary()

val_proba_e6 = mlp_base.predict(val_ds, verbose=0).ravel()
test_proba_e6 = mlp_base.predict(test_ds, verbose=0).ravel()

record_experiment(
    exp_id="E6",
    name="MLP 64-64 (Sequential)",
    paradigm="Deep Learning",
    y_true=y_test,
    proba=test_proba_e6,
    fit_time=time_e6,
    hyperparams=f"Sequential, Dense(64,relu)-Dense(64,relu)-Dense(1,sigmoid), Adam lr=1e-3, "
                f"batch=1024, epochs<={EPOCHS}, no dropout/BN/L2, no class weight",
    n_params=int(mlp_base.count_params()),
    notes="Under-regularised baseline; its train-validation gap motivates E7.",
    val_proba=val_proba_e6,
    y_val_=y_val,
)

train_proba_e6 = mlp_base.predict(train_ds, verbose=0).ravel()
print(f"\nTrain PR-AUC {average_precision_score(y_train, train_proba_e6):.4f} vs "
      f"test PR-AUC {average_precision_score(y_test, test_proba_e6):.4f}")
