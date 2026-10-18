# ----------------------------------------------------------------------------------------------
# E9 -- Best architecture + class weighting + LR decay; Adam vs SGD-momentum compared on val.
# ----------------------------------------------------------------------------------------------
set_global_seed()

use_embeddings = VAL_PR_AUC["E8"] >= VAL_PR_AUC["E7"]
print(f"Building on {'E8 (embeddings)' if use_embeddings else 'E7 (dense MLP)'} "
      f"-- higher validation PR-AUC ({max(VAL_PR_AUC.values()):.4f}).")

if use_embeddings:
    build_e9 = build_embedding_model
    tr_e9, va_e9, te_e9 = train_ds_emb, val_ds_emb, test_ds_emb
else:
    build_e9 = lambda: build_regularised_mlp(n_features, name="E9_MLP_weighted")
    tr_e9, va_e9, te_e9 = train_ds, val_ds, test_ds

decay = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-3, decay_steps=1000, decay_rate=0.90, staircase=False
)

candidates = {
    "Adam + exp decay": keras.optimizers.Adam(learning_rate=decay),
    "SGD + Nesterov momentum": keras.optimizers.SGD(learning_rate=0.02, momentum=0.9, nesterov=True),
}

e9_runs, e9_best = {}, None
for label, opt in candidates.items():
    print(f"\n--- E9 candidate: {label} ---")
    set_global_seed()
    net = build_e9()
    tag = "E9_" + label.split()[0].lower()
    hist, secs = train_network(net, tag, tr_e9, va_e9, use_class_weight=True, optimizer=opt)
    vp = net.predict(va_e9, verbose=0).ravel()
    score = average_precision_score(y_val, vp)
    e9_runs[label] = {"val_pr_auc": round(score, 4), "seconds": round(secs, 1),
                      "epochs": len(hist.history["loss"])}
    print(f"    validation PR-AUC {score:.4f}")
    if e9_best is None or score > e9_best[2]:
        e9_best = (net, label, score, hist, secs)

net_e9, e9_label, e9_score, hist_e9, time_e9 = e9_best
print("\nOptimiser comparison (validation PR-AUC):")
print(pd.DataFrame(e9_runs).T.to_string())

val_proba_e9 = net_e9.predict(va_e9, verbose=0).ravel()
test_proba_e9 = net_e9.predict(te_e9, verbose=0).ravel()

record_experiment(
    exp_id="E9",
    name=f"Best DL + class weights ({e9_label.split()[0]})",
    paradigm="Deep Learning",
    y_true=y_test,
    proba=test_proba_e9,
    fit_time=time_e9,
    hyperparams=(f"{'embeddings+dense' if use_embeddings else 'dense 128-64-32'}, "
                 f"class_weight={{0:{CLASS_WEIGHT[0]:.2f}, 1:{CLASS_WEIGHT[1]:.2f}}}, "
                 f"{e9_label}, exponential LR decay (3e-3, x0.90/1000 steps), "
                 f"early stop val_pr_auc patience=6, batch=1024"),
    n_params=int(net_e9.count_params()),
    notes="Only experiment with a re-weighted loss; isolates the effect of class weighting from architecture.",
    val_proba=val_proba_e9,
    y_val_=y_val,
)
