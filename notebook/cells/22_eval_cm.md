### 11.2 Figure 6 — Confusion matrices

A confusion matrix is where aggregate metrics stop being abstract. ROC-AUC and PR-AUC summarise the whole
ranking; the matrix reports what the model actually does at the threshold that will be deployed, expressed in
patients rather than in fractions.

The matrices below are **row-normalised**, so each cell reads as "of the respondents who truly are in this
class, what share did the model put here". Row normalisation is used rather than column normalisation because
the clinically relevant question is *how many cases were missed*, which is a per-true-class question. Raw counts
are printed in each cell so the scale of the test set stays visible.

Four models are shown: the linear baseline (E1), the best classical model, the unregularised network (E6), and
the best-performing deep model. Comparing E1 with E6 isolates what non-linearity buys; comparing E6 with the
best deep model isolates what regularisation and weighting buy.
