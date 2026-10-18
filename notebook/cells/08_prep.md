## 6. Shared evaluation machinery

Every one of the nine experiments is scored by the same function. This is deliberate: when results are
computed by different code paths, small differences in metric definition quietly become model differences,
and the whole comparison becomes an artefact of implementation drift.

### 6.1 Which metrics, and why this set

Accuracy alone cannot discriminate between a model that detects diabetes and a model that ignores it, because
the majority class is 86 % of the data. Six metrics are therefore recorded for every experiment:

| Metric | What it answers | Why it is included here |
|---|---|---|
| **ROC-AUC** | Ranking quality across all thresholds | Threshold-free, the fairest single comparison between paradigms; invariant to class balance |
| **PR-AUC** | Ranking quality *for the positive class* | The honest metric under 6:1 imbalance — it does not reward correct "no diabetes" predictions |
| **Recall** (sensitivity) | Share of true diabetics caught | The clinically decisive number: a missed case is an untreated patient |
| **Precision** | Share of flagged respondents who are diabetic | Governs the cost of follow-up testing |
| **F1** | Harmonic mean of the two | Single summary when both errors matter, as they do in triage |
| **MCC** | Correlation between prediction and truth over all four cells | Robust to imbalance; a value near 0 means no better than chance |
| **Brier** | Mean squared error of the probabilities | Separates *discrimination* (AUC) from *calibration*, which AUC ignores |

Precision, recall and F1 are computed at a fixed 0.5 threshold for comparability across the table, and again
at a validation-tuned threshold in Section 9.4. Reporting both is the only way to show that a model's
apparent weakness is sometimes a threshold artefact rather than a discrimination failure.

### 6.2 How models will be compared

Two comparison rules are fixed before any result is seen, to prevent the temptation to rationalise afterwards:

1. **Primary criterion: PR-AUC**, because the operating context is a screening programme where the positive
   class is the one that matters. ROC-AUC is reported alongside and any disagreement between the two is
   discussed rather than hidden.
2. **Ties broken by recall at the tuned threshold**, since a screening test that misses cases has failed at
   its stated purpose regardless of its precision.
