# ----------------------------------------------------------------------------------------------
# One metric function, one logger.  Every experiment below funnels through these.
# ----------------------------------------------------------------------------------------------
EXPERIMENT_LOG: list[dict] = []      # rows of Table II
TEST_PREDICTIONS: dict[str, dict] = {}   # exp_id -> {y_true, proba, paradigm, name}
SCALERS: dict[str, StandardScaler] = {}


def metrics_at(y_true, proba, threshold: float = 0.5) -> dict:
    """All reported metrics for one model at one decision threshold."""
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "mcc": matthews_corrcoef(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
    }


def best_threshold_by_f1(y_true, proba) -> float:
    """Threshold that maximises F1 on the *validation* partition.

    Tuning on validation rather than test keeps the test set untouched until Section 9.
    """
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    return float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5


def record_experiment(exp_id: str, name: str, paradigm: str, y_true, proba,
                      fit_time: float, hyperparams: str, n_params: int, notes: str,
                      val_proba=None, y_val_=None) -> dict:
    """Score a model, append it to Table II, and cache its test probabilities."""
    m = metrics_at(y_true, proba)
    thr = best_threshold_by_f1(y_val_, val_proba) if val_proba is not None else 0.5
    m_tuned = metrics_at(y_true, proba, thr)

    row = {
        "exp_id": exp_id, "model": name, "paradigm": paradigm,
        "hyperparameters": hyperparams, "n_params": n_params,
        "fit_time_s": round(fit_time, 2),
        **{k: round(v, 4) for k, v in m.items()},
        "tuned_threshold": round(thr, 4),
        "recall_tuned": round(m_tuned["recall"], 4),
        "f1_tuned": round(m_tuned["f1"], 4),
        "precision_tuned": round(m_tuned["precision"], 4),
        "notes": notes,
    }
    EXPERIMENT_LOG.append(row)
    TEST_PREDICTIONS[exp_id] = {
        "y_true": np.asarray(y_true),
        "proba": np.asarray(proba, dtype=float),
        "name": name, "paradigm": paradigm,
    }

    print(f"[{exp_id}] {name}")
    print(f"    acc {m['accuracy']:.4f} | prec {m['precision']:.4f} | rec {m['recall']:.4f} | "
          f"F1 {m['f1']:.4f} | ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} | "
          f"MCC {m['mcc']:.4f} | Brier {m['brier']:.4f}")
    print(f"    tuned threshold {thr:.3f} -> rec {m_tuned['recall']:.4f}, prec {m_tuned['precision']:.4f}, "
          f"F1 {m_tuned['f1']:.4f}   (fit {fit_time:.1f}s)")
    return row


COLUMN_SETS = {"raw": FEATURES, "full": FULL_FEATURES}


def scaled_versions(feature_set: str):
    """Fit a StandardScaler on the training partition and return scaled copies of all three."""
    if feature_set not in SCALERS:
        SCALERS[feature_set] = StandardScaler().fit(TRAIN_FRAMES[feature_set][COLUMN_SETS[feature_set]])
    scaler = SCALERS[feature_set]
    COLUMNS = COLUMN_SETS[feature_set]
    return (
        scaler.transform(TRAIN_FRAMES[feature_set][COLUMNS]),
        scaler.transform(VAL_FRAMES[feature_set][COLUMNS]),
        scaler.transform(TEST_FRAMES[feature_set][COLUMNS]),
    )


TRAIN_FRAMES = {"raw": X_train, "full": X_train_fe}
VAL_FRAMES = {"raw": X_val, "full": X_val_fe}
TEST_FRAMES = {"raw": X_test, "full": X_test_fe}

print("Evaluation harness ready.")
print(f"Reference floor: predicting the majority class scores accuracy "
      f"{max(y_test.mean(), 1-y_test.mean()):.4f} and MCC {0.0:.4f}.")
print(f"Test set size: {len(y_test):,} ({int(y_test.sum()):,} positives)")
