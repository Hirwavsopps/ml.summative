# ----------------------------------------------------------------------------------------------
# E8 -- Functional API: embedding branch for ordinal codes + dense branch for measurements.
# ----------------------------------------------------------------------------------------------
set_global_seed()

# Ordinal survey codes -> embedded.  Everything else is a count or a measurement -> dense branch.
CAT_COLS = ["GenHlth", "Age", "Education", "Income", "BMI_band"]
NUM_COLS = [c for c in FULL_FEATURES if c not in CAT_COLS]
CAT_OFFSETS = {"GenHlth": 1, "Age": 1, "Education": 1, "Income": 1, "BMI_band": 0}

cat_frames = {"train": X_train_fe, "val": X_val_fe, "test": X_test_fe}
num_matrix = {"train": Xtr_s, "val": Xval_s, "test": Xte_s}
num_idx = [FULL_FEATURES.index(c) for c in NUM_COLS]

# One int32 tensor per categorical column, so the model needs no slicing layers at all.
# Slicing would require a Lambda layer, and Lambda layers do not survive Keras serialisation cleanly.
cat_arrays, num_arrays = {}, {}
for part, frame in cat_frames.items():
    cat_arrays[part] = {
        c: (frame[c].to_numpy() - CAT_OFFSETS[c]).reshape(-1, 1).astype("int32") for c in CAT_COLS
    }
    num_arrays[part] = num_matrix[part][:, num_idx].astype("float32")

CARDINALITIES = {c: int(cat_arrays["train"][c].max()) + 1 for c in CAT_COLS}
EMB_DIMS = {c: int(min(4, (k + 1) // 2)) for c, k in CARDINALITIES.items()}
print("Embedding table:")
for c in CAT_COLS:
    print(f"    {c:<12} cardinality {CARDINALITIES[c]:>2} -> embedding dim {EMB_DIMS[c]}")
print(f"    {'numeric':<12} {len(NUM_COLS)} columns -> dense branch")


def make_dict_dataset(part: str, training: bool) -> tf.data.Dataset:
    """Dict-input tf.data pipeline: {cat_*: (n,1) int32, num: (n,p) float32} -> label."""
    y = {"train": ytr_dl, "val": yval_dl, "test": yte_dl}[part]
    features = {f"cat_{c}": cat_arrays[part][c] for c in CAT_COLS}
    features["num"] = num_arrays[part]
    ds = tf.data.Dataset.from_tensor_slices((features, y))
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)


train_ds_emb = make_dict_dataset("train", True)
val_ds_emb = make_dict_dataset("val", False)
test_ds_emb = make_dict_dataset("test", False)


def build_embedding_model(name: str = "E8_Functional_Embeddings") -> keras.Model:
    """Two-branch Functional model: embeddings for ordinal codes, dense for measurements.

    Keras 3 takes the *keys of the inputs dict* as the model's input names, so the keys here must
    match the keys produced by `make_dict_dataset` exactly (`cat_<column>`).
    """
    cat_inputs = {
        f"cat_{c}": keras.Input(shape=(1,), dtype="int32", name=f"cat_{c}") for c in CAT_COLS
    }
    embedded = [
        layers.Flatten(name=f"flat_{c}")(
            layers.Embedding(CARDINALITIES[c], EMB_DIMS[c], name=f"emb_{c}")(cat_inputs[f"cat_{c}"])
        )
        for c in CAT_COLS
    ]
    cat_branch = layers.Concatenate(name="cat_concat")(embedded)
    cat_branch = layers.Dense(16, activation="relu", name="cat_dense")(cat_branch)

    num_in = keras.Input(shape=(len(NUM_COLS),), dtype="float32", name="num")
    num_branch = layers.Dense(64, name="num_dense_1")(num_in)
    num_branch = layers.BatchNormalization(name="num_bn_1")(num_branch)
    num_branch = layers.Activation("relu", name="num_relu_1")(num_branch)
    num_branch = layers.Dropout(0.20, name="num_drop_1")(num_branch)
    num_branch = layers.Dense(32, activation="relu", name="num_dense_2")(num_branch)

    merged = layers.Concatenate(name="merge")([num_branch, cat_branch])
    x = layers.Dense(32, activation="relu", name="head_dense")(merged)
    x = layers.Dropout(0.20, name="head_drop")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return keras.Model({**cat_inputs, "num": num_in}, outputs, name=name)


emb_model = build_embedding_model()
hist_e8, time_e8 = train_network(emb_model, "E8", train_ds_emb, val_ds_emb)
emb_model.summary()

val_proba_e8 = emb_model.predict(val_ds_emb, verbose=0).ravel()
test_proba_e8 = emb_model.predict(test_ds_emb, verbose=0).ravel()

record_experiment(
    exp_id="E8",
    name="Emb+MLP (Functional API)",
    paradigm="Deep Learning",
    y_true=y_test,
    proba=test_proba_e8,
    fit_time=time_e8,
    hyperparams=(f"Functional 2-branch: embeddings {EMB_DIMS} for {len(CAT_COLS)} ordinal codes + "
                 f"dense 64-32 for {len(NUM_COLS)} numeric; concat -> Dense(32)->Dropout(0.2)->sigmoid; "
                 f"Adam lr=1e-3, batch=1024, no class weight"),
    n_params=int(emb_model.count_params()),
    notes="Representation change, not just architecture change: ordinal codes are learned, not imposed as linear.",
    val_proba=val_proba_e8,
    y_val_=y_val,
)

# remember validation PR-AUC so E9 can build on the stronger of E7 / E8 without peeking at the test set
VAL_PR_AUC = {
    "E7": average_precision_score(y_val, val_proba_e7),
    "E8": average_precision_score(y_val, val_proba_e8),
}
print("\nValidation PR-AUC: " + " | ".join(f"{k} {v:.4f}" for k, v in VAL_PR_AUC.items()))
