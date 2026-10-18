## 11. Consolidated results

### 11.1 Table II — the experiment table

Every experiment recorded by `record_experiment` is consolidated below and written to
`results/table2_experiments.csv`. This table is the single source of truth: the numbers in the report are read
from this file, not retyped, so the notebook and the paper cannot drift apart.

Three columns deserve explanation before the table is read:

- **`fit_time_s`** includes hyperparameter search where one was run. It is therefore a measure of *total cost
  to reach the reported model*, not of a single `.fit()` call. E3 and E4 pay for their search; E5 pays for four
  SVM fits.
- **`n_params`** is not comparable across paradigms and is not meant to be. For the ensembles it counts leaves;
  for the networks it counts weights. It is included to show the order-of-magnitude difference in model
  complexity behind nearly identical scores, which is the capacity half of the bias–variance discussion.
- **`recall_tuned`** and **`f1_tuned`** are measured at the F1-optimal threshold selected on the validation
  partition, not at 0.5. The gap between `recall` and `recall_tuned` is information: it shows how much of each
  model's apparent weakness is a threshold artefact.
