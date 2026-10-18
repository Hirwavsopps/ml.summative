# ----------------------------------------------------------------------------------------------
# tf.data pipelines, class weights, and one training helper reused by E6-E9.
# ----------------------------------------------------------------------------------------------
set_global_seed()
tf.keras.utils.set_random_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1024
SHUFFLE_BUFFER = 20_000
EPOCHS = 40

# ---- tensors: float32 features, float32 targets (Keras needs a single dtype for BCE) ------------
Xtr_dl = Xtr_s.astype("float32")
Xval_dl = Xval_s.astype("float32")
Xte_dl = Xte_s.astype("float32")
ytr_dl = y_train.to_numpy().astype("float32")
yval_dl = y_val.to_numpy().astype("float32")
yte_dl = y_test.to_numpy().astype("float32")

# ---- class weights, computed from the training partition only ---------------------------------
n_pos = float(ytr_dl.sum())
n_neg = float(len(ytr_dl) - n_pos)
CLASS_WEIGHT = {0: len(ytr_dl) / (2.0 * n_neg), 1: len(ytr_dl) / (2.0 * n_pos)}


def make_dataset(X, y, training: bool, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Build a cached, prefetched tf.data pipeline.

    training=True  -> shuffle with a large buffer, never repeat (epochs are driven by fit()).
    training=False -> deterministic order so evaluation is reproducible.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).cache().prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(Xtr_dl, ytr_dl, training=True)
val_ds = make_dataset(Xval_dl, yval_dl, training=False)
test_ds = make_dataset(Xte_dl, yte_dl, training=False)

# ---- one training routine for all four networks -----------------------------------------------
def learning_rate_is_settable(optimizer) -> bool:
    """True when the optimizer's learning rate can be reassigned by a callback.

    Optimizers constructed with a `LearningRateSchedule` raise on assignment, which would make
    `ReduceLROnPlateau` crash on the first epoch it fires.
    """
    try:
        current = optimizer.learning_rate
        optimizer.learning_rate = current
        return True
    except Exception:
        return False


def train_network(model, tag: str, train_data, val_data, epochs: int = EPOCHS,
                  use_class_weight: bool = False, lr_schedule=None, optimizer=None):
    """Compile, fit with early stopping + TensorBoard logging, and return (history, seconds).

    `optimizer` overrides the default Adam; `lr_schedule` accepts a float or a Keras schedule.
    """
    set_global_seed()
    tf.keras.utils.set_random_seed(SEED)

    if optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule if lr_schedule else 1e-3)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.AUC(name="pr_auc", curve="PR"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )

    log_dir = LOG_DIR / "tensorboard" / tag
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0,
                                    write_graph=False, update_freq="epoch"),
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=6,
                                      restore_best_weights=True, verbose=0),
    ]
    # ReduceLROnPlateau *assigns* a new learning rate, which an optimizer built around a
    # LearningRateSchedule refuses. E9 uses an exponential schedule, so the plateau callback is
    # attached only when the optimizer's rate is settable.
    if learning_rate_is_settable(optimizer):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5, verbose=0))

    t0 = time.perf_counter()
    history = model.fit(
        train_data, validation_data=val_data, epochs=epochs, verbose=0,
        callbacks=callbacks, class_weight=CLASS_WEIGHT if use_class_weight else None,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [{tag}] {len(history.history['loss'])} epochs in {elapsed:.1f}s | "
          f"best val PR-AUC {max(history.history['val_pr_auc']):.4f}")
    return history, elapsed


print(f"Train batches        : {len(train_ds)} @ batch={BATCH_SIZE}")
print(f"Validation batches   : {len(val_ds)}")
print(f"Test batches         : {len(test_ds)}")
print(f"Class weights        : neg {CLASS_WEIGHT[0]:.3f} | pos {CLASS_WEIGHT[1]:.3f}  (inverse frequency)")
print(f"Feature tensor dtype : {train_ds.element_spec[0].dtype.name}, shape {Xtr_dl.shape}")
