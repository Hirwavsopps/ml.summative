### 2.5 The variable dictionary and the class imbalance

**Table I** below is the dictionary the report cites. It is generated programmatically from `VARIABLE_SPEC`
so that the report table and the notebook cannot disagree.

**Figure 1** quantifies the problem the rest of the notebook has to solve. 86.07 % of respondents are
non-diabetic and 13.93 % are diabetic or pre-diabetic — a ratio of roughly 6.2 to 1. Two consequences follow
immediately and they shape every design decision that comes after this cell:

1. **Accuracy is a misleading headline metric.** A model that predicts "no diabetes" for every respondent
   scores 0.8607 accuracy while detecting nobody. The classifier that achieves it has learned nothing and
   would be worthless clinically, yet it would top an accuracy-only leaderboard. Every model in this study is
   therefore evaluated on **ROC-AUC, PR-AUC, recall and F1** alongside accuracy.
2. **The default decision threshold is not the right threshold.** At 0.5, a model trained on 6:1 data
   systematically under-predicts the minority class. Section 9.4 shows what happens when the threshold is
   chosen on the validation split instead of assumed.
