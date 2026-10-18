## 4. Splitting: three disjoint partitions, stratified

The split is **70 / 15 / 15** and it is asserted, not assumed, because every result in this notebook depends
on it being exactly this.

The three-way split exists for a specific reason. With two partitions one is forced to choose between using
the held-out data to early-stop a network — which leaks it — or not early-stopping at all. A validation split
resolves that: **early stopping, hyperparameter selection and threshold tuning all happen on `val`**, and
`test` is evaluated once, in Section 9, after every model is frozen.

Stratification preserves the 13.93 % prevalence in all three partitions. Without it, random chance would move
the positive rate by roughly ±0.35 percentage points per partition, and a comparison between two models
scoring within a point of each other would be measuring split noise rather than model quality.

The `ID` column is dropped before splitting. It is a row index, not a measurement; leaving it in would allow a
model to memorise respondents and would trivially inflate any tree model that split on it.
