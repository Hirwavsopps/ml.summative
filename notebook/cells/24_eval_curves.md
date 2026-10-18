### 11.4 Figure 5 — Training and validation curves for the four networks

For a neural network the learning curve has a different job from the scikit-learn curves in Figure 4. It is
the record of a single optimisation run, and the *distance between the training and validation curves over
epochs* is the direct read-out of variance.

Three diagnostic signatures are worth looking for:

- **Validation loss turning upward while training loss keeps falling.** This is overfitting in real time: the
  model is now fitting training noise. Every network here is protected by early stopping with
  `restore_best_weights`, so the reported model is the one from the turning point, not the end of the run.
- **Both curves flattening at a high loss.** Underfitting. The remedy is capacity or features, not more epochs.
- **Jagged validation curves.** Variance in the estimate, usually from too small a validation partition or too
  large a learning rate. `ReduceLROnPlateau` damps this, and the effect is visible as the curve settling after
  a drop in the learning rate.

The PR-AUC row is the more sensitive indicator here: loss is dominated by the majority class, so it keeps
improving after the positive class has stopped being learned. That divergence alone is a reason to monitor
PR-AUC during training rather than loss.
